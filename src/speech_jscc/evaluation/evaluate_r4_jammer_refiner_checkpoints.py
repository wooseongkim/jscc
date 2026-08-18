"""Paired fixed-held-out evaluation for trained R4 jammer refiner checkpoints.

This module deliberately does not train or select a model.  It materializes a
deterministic validation condition plan once, then applies every saved
estimator/refiner checkpoint to precisely that plan.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import replace
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml

from channels.multipath import exponential_pdp
from channels.physical_ofdm import active_grid_masks
from channels.temporal_multipath import correlated_tap_trajectory, jakes_slot_correlation
from channels.physical_ofdm import active_grid_masks
from channels.re_risk import (
    oracle_jammer_grid_to_interference_report,
    oracle_jamming_mask_to_risk_report,
)
from channels.r4_uep_allocator import UEPProfile
from speech_jscc.evaluation.r4_jammer_baseline import build_r4_jammer
from models.adaptive_latent_refiner import load_adaptive_latent_refiner_checkpoint
from models.jammer_estimator import load_jammer_estimator_checkpoint
from speech_jscc.config import resolve_device
from speech_jscc.experiment import build_components
from speech_jscc.training.channel_free_revalidation import per_layer_nmse
from speech_jscc.training.r4_waveform_finetune import (
    R4ForwardCondition,
    R4WaveformForward,
    freeze_codec_for_input_gradient,
)
from speech_jscc.training.si_sdr_loss import si_sdr
from speech_jscc.evaluation.speech_quality_metrics import (
    MetricOptions,
    RawSpeechMetricComputer,
    metric_backend_metadata,
)
from speech_jscc.training.train_r4_jammer_refiner import (
    JAMMER_TYPE_TO_INDEX,
    TRAINER_JAMMER_TYPE_CLASSES,
)
from train_channel_free_conv_conformer import fixed_paths, load_batch, sha256


ROOT = Path(__file__).resolve().parents[3]
FIXED_JAMMER_TYPES: tuple[str, ...] = (
    "no_jammer", "broadband", "subband", "burst", "tone",
)
_PHYSICAL_JAMMER_TYPE = {"broadband": "broadband_awgn", **{name: name for name in FIXED_JAMMER_TYPES if name != "broadband"}}


def _stable_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def condition_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """All non-model fields which must be shared across checkpoints."""
    keys = (
        "sample_id", "crop_offset", "snr_db", "jsr_db", "jammer_type",
        "realization_index", "channel_seed", "noise_seed", "jammer_seed",
    )
    return tuple(row[key] for key in keys) + (
        row.get("allocation_risk_mode", "none"), float(row.get("risk_alpha", 0.0)),
    )


_ALLOCATION_COMPARISON_KEY_FIELDS = (
    "checkpoint_name", "sample_id", "crop_offset", "snr_db", "jsr_db",
    "jammer_type", "realization_index", "channel_seed", "noise_seed",
    "jammer_seed", "condition_hash",
)


def allocation_comparison_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Key for strict triplet/CSI-only/interference allocation joins."""
    jsr = row["jsr_db"]
    return (
        str(row["checkpoint_name"]), str(row["sample_id"]), int(row["crop_offset"]),
        float(row["snr_db"]), None if jsr in (None, "") else float(jsr),
        str(row["jammer_type"]), int(row["realization_index"]),
        int(row["channel_seed"]), int(row["noise_seed"]), int(row["jammer_seed"]),
        str(row["condition_hash"]),
    )


def build_three_way_allocation_comparison(
    existing_rows: list[dict[str, Any]],
    csi_only_rows: list[dict[str, Any]],
    interference_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join three allocation runs and calculate raw/refined paired deltas."""
    indexed: list[dict[tuple[Any, ...], dict[str, Any]]] = []
    for label, rows in zip(("existing", "csi_only", "interference"), (existing_rows, csi_only_rows, interference_rows), strict=True):
        table = {allocation_comparison_key(row): row for row in rows}
        if len(table) != len(rows):
            raise ValueError(f"duplicate condition key in {label} allocation run")
        indexed.append(table)
    keys = set(indexed[0])
    if any(set(table) != keys for table in indexed[1:]):
        raise ValueError("condition key mismatch across existing, csi_only, and interference runs")
    output: list[dict[str, Any]] = []
    for key in sorted(keys, key=repr):
        existing, csi_only, interference = (table[key] for table in indexed)
        comparison = {
            "condition_key": json.dumps(key, default=str, separators=(",", ":")),
            "jammer_type": existing["jammer_type"], "jsr_db": existing["jsr_db"],
            "snr_db": existing["snr_db"],
            "si_sdr_existing_raw": float(existing["raw_si_sdr"]),
            "si_sdr_csi_only_raw": float(csi_only["raw_si_sdr"]),
            "si_sdr_interference_raw": float(interference["raw_si_sdr"]),
            "delta_csi_only_minus_existing": float(csi_only["raw_si_sdr"]) - float(existing["raw_si_sdr"]),
            "delta_interference_minus_csi_only": float(interference["raw_si_sdr"]) - float(csi_only["raw_si_sdr"]),
            "delta_interference_minus_existing": float(interference["raw_si_sdr"]) - float(existing["raw_si_sdr"]),
            "si_sdr_existing_refined": float(existing["refined_si_sdr"]),
            "si_sdr_csi_only_refined": float(csi_only["refined_si_sdr"]),
            "si_sdr_interference_refined": float(interference["refined_si_sdr"]),
            "mapping_hash_existing": existing["mapping_hash"],
            "mapping_hash_csi_only": csi_only["mapping_hash"],
            "mapping_hash_interference": interference["mapping_hash"],
        }
        # Quality backends are optional for ordinary fixed validation.  When
        # enabled for all three runs, carry them through the same strict join.
        for metric in ("raw_estoi", "raw_wer", "raw_visqol_mos_lqo"):
            present = [metric in row and row[metric] not in (None, "") for row in (existing, csi_only, interference)]
            if any(present) and not all(present):
                raise ValueError(f"incomplete {metric} values across allocation runs")
            if all(present):
                comparison[f"{metric}_existing"] = float(existing[metric])
                comparison[f"{metric}_csi_only"] = float(csi_only[metric])
                comparison[f"{metric}_interference"] = float(interference[metric])
        output.append(comparison)
    return output


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def build_three_way_allocation_comparison_from_csv(
    existing_path: Path, csi_only_path: Path, interference_path: Path,
) -> list[dict[str, Any]]:
    """Load the three distinct completed runs before strict condition joining."""
    return build_three_way_allocation_comparison(
        _read_csv_rows(existing_path),
        _read_csv_rows(csi_only_path),
        _read_csv_rows(interference_path),
    )


def _p5(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute percentile of an empty set")
    return sorted(values)[max(0, math.ceil(0.05 * len(values)) - 1)]


def summarize_three_way_allocation_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize all-SNR and individual-SNR allocation deltas, including tails."""
    groups: dict[tuple[str, str | None, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        jsr = None if row["jsr_db"] in (None, "") else str(row["jsr_db"])
        groups[(str(row["jammer_type"]), jsr, "all")].append(row)
        groups[(str(row["jammer_type"]), jsr, str(row["snr_db"]))].append(row)
    output: list[dict[str, Any]] = []
    for (jammer_type, jsr_db, snr_db), grouped in sorted(groups.items(), key=repr):
        values = {field: [float(row[field]) for row in grouped] for field in (
            "si_sdr_existing_raw", "si_sdr_csi_only_raw", "si_sdr_interference_raw",
            "delta_csi_only_minus_existing", "delta_interference_minus_csi_only",
            "delta_interference_minus_existing",
        )}
        output.append({
            "jammer_type": jammer_type, "jsr_db": "no_jammer" if jsr_db is None else jsr_db,
            "snr_db": snr_db, "rows": len(grouped),
            **{field: _mean([{field: value} for value in field_values], field) for field, field_values in values.items()},
            "p5_delta_csi_minus_existing": _p5(values["delta_csi_only_minus_existing"]),
            "p5_delta_interference_minus_csi": _p5(values["delta_interference_minus_csi_only"]),
            "p5_delta_interference_minus_existing": _p5(values["delta_interference_minus_existing"]),
            "frac_delta_csi_minus_existing_lt_minus3": sum(value < -3.0 for value in values["delta_csi_only_minus_existing"]) / len(grouped),
            "frac_delta_interference_minus_csi_lt_minus3": sum(value < -3.0 for value in values["delta_interference_minus_csi_only"]) / len(grouped),
            "frac_delta_interference_minus_existing_lt_minus3": sum(value < -3.0 for value in values["delta_interference_minus_existing"]) / len(grouped),
            "frac_si_sdr_existing_lt_minus10": sum(value < -10.0 for value in values["si_sdr_existing_raw"]) / len(grouped),
            "frac_si_sdr_csi_only_lt_minus10": sum(value < -10.0 for value in values["si_sdr_csi_only_raw"]) / len(grouped),
            "frac_si_sdr_interference_lt_minus10": sum(value < -10.0 for value in values["si_sdr_interference_raw"]) / len(grouped),
        })
        if "raw_estoi_existing" in grouped[0]:
            for metric in ("raw_estoi", "raw_wer", "raw_visqol_mos_lqo"):
                for mode in ("existing", "csi_only", "interference"):
                    field = f"{metric}_{mode}"
                    output[-1][field] = _mean(grouped, field)
    return output


def _verify_allocation_comparison_artifacts(run_dirs: list[Path]) -> None:
    """Reject a three-way table unless all completed runs share fixed inputs."""
    manifests = [json.loads((directory / "checkpoint_manifest.json").read_text()) for directory in run_dirs]
    integrity = [json.loads((directory / "validation_integrity.json").read_text()) for directory in run_dirs]
    if any(manifest != manifests[0] for manifest in manifests[1:]):
        raise ValueError("checkpoint SHA mismatch across allocation runs")
    fixed = [(item["source_checkpoint_sha256"], item["conditions_per_checkpoint"], item["total_rows"]) for item in integrity]
    if any(item != fixed[0] for item in fixed[1:]):
        raise ValueError("validation integrity mismatch across allocation runs")


def build_fixed_condition_plan(
    *,
    sample_ids: list[str],
    crop_offsets: list[int],
    snr_db: list[float | int],
    jsr_db: list[float | int],
    realizations: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Return a stable ordered set of source/channel/jammer conditions."""
    if len(sample_ids) != len(crop_offsets):
        raise ValueError("sample_ids and crop_offsets must have equal length")
    if realizations <= 0:
        raise ValueError("realizations must be positive")
    rows: list[dict[str, Any]] = []
    ordinal = 0
    for sample_index, (sample_id, crop_offset) in enumerate(zip(sample_ids, crop_offsets)):
        for realization_index in range(realizations):
            for snr in snr_db:
                for jammer_type in FIXED_JAMMER_TYPES:
                    targets: Iterable[float | None] = (None,) if jammer_type == "no_jammer" else jsr_db
                    for jsr in targets:
                        row = {
                            "sample_id": str(sample_id),
                            "sample_index": sample_index,
                            "crop_offset": int(crop_offset),
                            "snr_db": float(snr),
                            "jsr_db": None if jsr is None else float(jsr),
                            "jammer_type": jammer_type,
                            "realization_index": realization_index,
                            "channel_seed": int(seed + 100_000 + realization_index * 10_000 + ordinal),
                            "noise_seed": int(seed + 200_000 + ordinal),
                            "jammer_seed": int(seed + 300_000 + ordinal),
                        }
                        row["condition_hash"] = _stable_hash(row)
                        rows.append(row)
                        ordinal += 1
    return rows


def verify_paired_row_keys(rows: list[dict[str, Any]], checkpoint_names: list[str]) -> None:
    expected: set[tuple[Any, ...]] | None = None
    for name in checkpoint_names:
        observed = [condition_key(row) for row in rows if row["checkpoint_name"] == name]
        if len(observed) != len(set(observed)):
            raise ValueError(f"duplicate fixed validation row key for {name}")
        keys = set(observed)
        if expected is None:
            expected = keys
        elif keys != expected:
            raise ValueError("paired validation row keys differ across checkpoints")


def mask_classification_metrics(*, predicted, target) -> dict[str, float]:
    predicted = torch.as_tensor(predicted, dtype=torch.bool)
    target = torch.as_tensor(target, dtype=torch.bool)
    if predicted.shape != target.shape:
        raise ValueError("predicted and target mask shapes differ")
    tp = (predicted & target).sum().item()
    fp = (predicted & ~target).sum().item()
    fn = (~predicted & target).sum().item()
    tn = (~predicted & ~target).sum().item()
    union = tp + fp + fn
    predicted_positive = tp + fp
    target_positive = tp + fn
    return {
        "iou": 1.0 if union == 0 else tp / union,
        "f1": 1.0 if predicted_positive + target_positive == 0 else 2.0 * tp / (predicted_positive + target_positive),
        "false_positive_rate": 0.0 if fp + tn == 0 else fp / (fp + tn),
        "false_negative_rate": 0.0 if target_positive == 0 else fn / target_positive,
    }


def _deterministic_crop_offset(path: Path, *, sample_rate: int, samples: int) -> int:
    """Mirror ``load_waveform_segment``'s center crop and expose its offset."""
    if path.suffix.lower() in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        waveform = payload["waveform"] if isinstance(payload, dict) else payload
        source_rate = int(payload.get("sample_rate", sample_rate)) if isinstance(payload, dict) else sample_rate
        length = int(torch.as_tensor(waveform).shape[-1])
    else:
        try:
            import soundfile as sf
        except ImportError as error:  # pragma: no cover - covered by the normal environment
            raise RuntimeError("soundfile is required to record deterministic crop offsets") from error
        info = sf.info(str(path))
        length, source_rate = int(info.frames), int(info.samplerate)
    scaled = round(length * sample_rate / source_rate)
    return max(0, (scaled - samples) // 2)


def _load_source_model(checkpoint_payload: dict, device: torch.device):
    source_checkpoint = Path(checkpoint_payload["source_checkpoint"])
    if not source_checkpoint.is_absolute():
        source_checkpoint = ROOT / source_checkpoint
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"source JSCC checkpoint does not exist: {source_checkpoint}")
    source_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    model_config = copy.deepcopy(source_payload["config"])
    for key in ("config_path", "checkpoint_path"):
        if key in model_config["codec"] and not Path(model_config["codec"][key]).is_absolute():
            model_config["codec"][key] = str(ROOT / model_config["codec"][key])
    model_config["device"] = str(device)
    codec, model = build_components(model_config, device)
    model.load_state_dict(source_payload["model"], strict=True)
    model.eval()
    freeze_codec_for_input_gradient(codec)
    return source_checkpoint, source_payload, model_config, codec, model


def _tap_coefficients(engine: R4WaveformForward, physical_config: dict, *, seed: int, device: torch.device) -> torch.Tensor:
    pdp = exponential_pdp(physical_config["channel"]["num_taps"], physical_config["channel"]["pdp_decay"])
    rho = jakes_slot_correlation(
        physical_config["physical"]["user_speed_mps"],
        physical_config["physical"]["carrier_frequency_hz"], engine.profile.tti_duration_s,
    )
    return correlated_tap_trajectory(slots=1, batch_size=1, pdp=pdp, rho=rho, seed=seed)[0].to(device)


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / max(1, len(rows))


def summarize_perceptual_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate raw speech-quality metrics by condition and over all SNRs.

    ``frac_raw_si_sdr_lt_minus10`` deliberately remains a row-level fraction:
    it answers how often an evaluated waveform enters the severe-distortion
    region, rather than pretending that an average SI-SDR preserves that fact.
    """
    groups: dict[tuple[str, float | None, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        jsr = row.get("jsr_db")
        key = (str(row["jammer_type"]), None if jsr in (None, "") else float(jsr))
        groups[(*key, "all")].append(row)
        groups[(*key, f"{float(row['snr_db']):g}")].append(row)
    summaries: list[dict[str, Any]] = []
    for (jammer_type, jsr_db, snr_db), grouped in sorted(groups.items(), key=repr):
        required = ("raw_estoi", "raw_wer", "raw_visqol_mos_lqo")
        if any(field not in row for row in grouped for field in required):
            raise ValueError("perceptual summary requires ESTOI, WER, and ViSQOL on every row")
        summaries.append({
            "jammer_type": jammer_type,
            "jsr_db": "no_jammer" if jsr_db is None else jsr_db,
            "snr_db": snr_db,
            "rows": len(grouped),
            **{field: _mean(grouped, field) for field in required},
            "frac_raw_si_sdr_lt_minus10": sum(float(row["raw_si_sdr"]) < -10.0 for row in grouped) / len(grouped),
        })
    return summaries


def _resolve_fixed_uep_profile(config: dict[str, Any]) -> tuple[str, UEPProfile | None]:
    """Accept a named fixed profile or an explicit frozen optimizer result."""
    artifact_value = config.get("uep_profile_selection_artifact")
    if artifact_value:
        artifact = Path(artifact_value)
        if not artifact.is_absolute():
            artifact = ROOT / artifact
        if not artifact.is_file():
            raise FileNotFoundError(f"fixed UEP selection artifact does not exist: {artifact}")
        selection_key = str(config.get("uep_profile_selection_key", "x_best"))
        payload = json.loads(artifact.read_text())
        selected = payload["selected"][selection_key]
        if selected.get("status") != "SELECTED":
            raise ValueError(f"fixed UEP selection {selection_key} is not selected")
        candidate = selected["candidate"]["candidate"]
        profile = UEPProfile(
            f"{selection_key}_{candidate['profile_id']}",
            tuple(int(value) for value in candidate["repetition"]),
            power_share=tuple(float(value) for value in candidate["power_share"]),
        )
        return profile.name, profile
    definition = config.get("uep_profile_definition")
    if definition is None:
        return str(config.get("uep_profile", "U0")), None
    profile = UEPProfile(
        str(definition["name"]), tuple(int(value) for value in definition["repetition"]),
        power_share=tuple(float(value) for value in definition["power_share"]),
    )
    return profile.name, profile


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval_r4_jammer_refiner_fixed.yaml")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-names", nargs="+", default=["last.pt", "best_validation_si_sdr.pt", "best_validation_latent.pt"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-utterances", type=int, default=None)
    parser.add_argument("--max-realizations", type=int, default=None)
    parser.add_argument("--snr-db", nargs="*", type=float, default=None)
    parser.add_argument("--jsr-db", nargs="*", type=float, default=None)
    parser.add_argument("--allocation-risk-mode", choices=("none", "oracle_jamming", "rx_residual", "delayed_rx_residual"), default=None)
    parser.add_argument("--risk-alpha", type=float, default=None)
    parser.add_argument("--risk-alpha-sweep", nargs="*", type=float, default=None)
    parser.add_argument(
        "--jammer-aware-allocation-mode",
        choices=("none", "csi_only", "delayed_rx_interference", "oracle_jamming_interference"),
        default=None,
    )
    parser.add_argument("--compare-existing-dir", default=None)
    parser.add_argument("--compare-interference-dir", default=None)
    parser.add_argument(
        "--allocation-sequence-mode",
        choices=("plan_row_sequence", "same_condition_warmup_tti_pair"),
        default=None,
        help="Use the same TTI-0 receiver observation for every scored TTI-1 condition.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enable-estoi", action="store_true")
    parser.add_argument("--enable-wer", action="store_true")
    parser.add_argument("--enable-visqol", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    checkpoint_paths = [checkpoint_dir / name for name in args.checkpoint_names]
    missing = [str(path) for path in checkpoint_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"missing jammer-refiner checkpoint(s): {missing}")
    output = Path(args.output_dir or config["output_root"])
    if not output.is_absolute():
        output = ROOT / output
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"refusing existing output directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=False)
    device = resolve_device(args.device or config["device"])
    metric_options = MetricOptions.from_config(config)
    metric_options = replace(
        metric_options,
        estoi_enabled=metric_options.estoi_enabled or args.enable_estoi,
        wer_enabled=metric_options.wer_enabled or args.enable_wer,
        visqol_enabled=metric_options.visqol_enabled or args.enable_visqol,
    )
    payloads = {path.name: torch.load(path, map_location="cpu", weights_only=False) for path in checkpoint_paths}
    source_values = {(payload["source_checkpoint_sha256"], payload["source_checkpoint"]) for payload in payloads.values()}
    if len(source_values) != 1:
        raise ValueError("all refiner checkpoints must share the same source JSCC checkpoint")
    source_checkpoint, source_payload, model_config, codec, model = _load_source_model(next(iter(payloads.values())), device)
    risk_config = dict(config.get("risk_aware_allocation", {}))
    if args.allocation_risk_mode is not None:
        risk_config["risk_mode"] = args.allocation_risk_mode
    if args.risk_alpha is not None:
        risk_config["risk_alpha"] = args.risk_alpha
    risk_config.setdefault("enabled", False)
    risk_config.setdefault("risk_mode", "none")
    risk_config.setdefault("risk_alpha", 0.0)
    risk_config.setdefault("risk_delay_ttis", 1)
    risk_config.setdefault("normalize_risk", True)
    risk_config["enabled"] = bool(risk_config["enabled"]) or risk_config["risk_mode"] != "none"
    jammer_aware_config = dict(config.get("jammer_aware_allocation", {}))
    if args.jammer_aware_allocation_mode is not None:
        jammer_aware_config["mode"] = args.jammer_aware_allocation_mode
    jammer_aware_config.setdefault("enabled", False)
    jammer_aware_config.setdefault("mode", "none")
    jammer_aware_config.setdefault("delay_ttis", 1)
    jammer_aware_config["enabled"] = bool(jammer_aware_config["enabled"]) or jammer_aware_config["mode"] != "none"
    sequence_mode = args.allocation_sequence_mode or config.get("allocation_sequence_mode")
    if sequence_mode is None:
        sequence_mode = "same_condition_warmup_tti_pair" if jammer_aware_config["mode"] != "none" else "plan_row_sequence"
    risk_alphas = args.risk_alpha_sweep or [float(risk_config["risk_alpha"])]
    validation_seed = int(config["validation_seed"])
    _, validation_paths = fixed_paths(model_config, int(model_config["seed"]))
    max_utterances = int(args.max_utterances or config["max_utterances"])
    selected_paths = validation_paths[:max_utterances]
    if not selected_paths:
        raise ValueError("fixed validation manifest is empty")
    sample_rate = int(model_config["codec"]["sample_rate"])
    waveform_samples = int(model_config["codec"]["waveform_samples"])
    manifest = [
        {"sample_id": str(path), "crop_offset": _deterministic_crop_offset(path, sample_rate=sample_rate, samples=waveform_samples)}
        for path in selected_paths
    ]
    plan = build_fixed_condition_plan(
        sample_ids=[item["sample_id"] for item in manifest],
        crop_offsets=[item["crop_offset"] for item in manifest],
        snr_db=args.snr_db or config["snr_db"], jsr_db=args.jsr_db or config["jsr_db"],
        realizations=int(args.max_realizations or config["realizations"]), seed=validation_seed,
    )
    _write_csv(output / "fixed_validation_manifest.csv", manifest, ["sample_id", "crop_offset"])
    (output / "fixed_condition_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    config["risk_aware_allocation"] = risk_config
    config["jammer_aware_allocation"] = jammer_aware_config
    config["allocation_sequence_mode"] = sequence_mode
    config["risk_alpha_sweep"] = risk_alphas
    config["speech_metrics"] = {
        "estoi_enabled": metric_options.estoi_enabled,
        "wer_enabled": metric_options.wer_enabled,
        "visqol_enabled": metric_options.visqol_enabled,
        "asr_model": metric_options.asr_model,
        "visqol_binary": metric_options.visqol_binary,
    }
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "speech_metric_backends.json").write_text(
        json.dumps(metric_backend_metadata(metric_options), indent=2, sort_keys=True) + "\n"
    )
    (output / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n")
    (output / "checkpoint_manifest.json").write_text(json.dumps({
        name: {"path": str(path), "sha256": sha256(path), "step": int(payloads[name]["global_step"])}
        for name, path in zip(payloads, checkpoint_paths)
    }, indent=2, sort_keys=True) + "\n")
    if args.dry_run:
        (output / "dry_run.json").write_text(json.dumps({"conditions": len(plan), "checkpoints": list(payloads), "device": str(device)}, indent=2) + "\n")
        return
    speech_metrics = RawSpeechMetricComputer(metric_options, device=str(device))
    metric_workspace = output / "metric_workspace"
    physical_config = yaml.safe_load((ROOT / "configs/ofdm_nr_like_r4.yaml").read_text())
    all_rows: list[dict[str, Any]] = []
    mapping_by_condition: dict[str, str] = {}
    profile_name, profile_definition = _resolve_fixed_uep_profile(config)
    for risk_alpha in risk_alphas:
        variant_risk_config = {**risk_config, "risk_alpha": float(risk_alpha)}
        engine = R4WaveformForward(
            codec, model, uep_profile_name=profile_name, uep_profile=profile_definition,
            risk_aware_allocation=variant_risk_config,
            jammer_aware_allocation=jammer_aware_config,
        )
        masks = active_grid_masks(engine.profile, device=device)
        for checkpoint_name, checkpoint_payload in payloads.items():
            estimator = load_jammer_estimator_checkpoint(checkpoint_payload["estimator"], device).eval()
            refiner = load_adaptive_latent_refiner_checkpoint(checkpoint_payload["adaptive_refiner"], device).eval()
            if tuple(estimator.jammer_type_classes) != TRAINER_JAMMER_TYPE_CLASSES:
                raise ValueError(f"{checkpoint_name} has incompatible jammer vocabulary")
            report = None
            delayed_re_risk = None
            delayed_re_interference = None
            with torch.no_grad():
                for tti, base_condition_row in enumerate(plan):
                    condition_row = {**base_condition_row, "allocation_risk_mode": variant_risk_config["risk_mode"], "risk_alpha": float(risk_alpha)}
                    waveform = load_batch([selected_paths[int(condition_row["sample_index"])]], model_config, device)
                    target = codec.encode_waveform(waveform)
                    # A delayed allocation report must come from an actual
                    # preceding receiver observation of the *same* stationary
                    # condition, not from the unrelated preceding JSONL row.
                    # The warm-up packet is not scored; its sole purpose is
                    # causal LS-CSI/interference feedback for the scored TTI.
                    paired_warmup = sequence_mode == "same_condition_warmup_tti_pair"
                    scored_tti = 1 if paired_warmup else tti
                    if paired_warmup:
                        warmup = R4ForwardCondition(
                            snr_db=float(condition_row["snr_db"]), tti=0,
                            tap_coefficients=_tap_coefficients(engine, physical_config, seed=int(condition_row["channel_seed"]), device=device),
                            noise_seed=int(condition_row["noise_seed"]),
                        )
                        warmup_output = engine.forward(
                            target, channel_condition=warmup, training=False,
                            jammer_type=_PHYSICAL_JAMMER_TYPE[condition_row["jammer_type"]],
                            jammer_jsr_db=condition_row["jsr_db"], jammer_seed=int(condition_row["jammer_seed"]),
                            jammer_subband_fraction=float(config["jammer"]["subband_fraction"]),
                            jammer_burst_fraction=float(config["jammer"]["burst_fraction"]),
                            jammer_tone_count=int(config["jammer"]["tone_count"]),
                        )
                        report = warmup_output.next_delayed_csi
                        delayed_re_interference = warmup_output.next_re_interference_report
                    condition = R4ForwardCondition(
                        snr_db=float(condition_row["snr_db"]), tti=scored_tti,
                        tap_coefficients=_tap_coefficients(engine, physical_config, seed=int(condition_row["channel_seed"]), device=device),
                        noise_seed=int(condition_row["noise_seed"]),
                    )
                    oracle_risk = None
                    oracle_interference = None
                    if variant_risk_config["risk_mode"] == "oracle_jamming" and float(risk_alpha) != 0.0:
                        # Upper-bound only: the geometry is explicitly labeled oracle.
                        dummy = torch.ones((1, engine.profile.active_subcarriers, engine.profile.n_ofdm_symbols), dtype=torch.complex64, device=device)
                        oracle = build_r4_jammer(dummy, masks.candidate_data, jammer_type=_PHYSICAL_JAMMER_TYPE[condition_row["jammer_type"]], jsr_db=condition_row["jsr_db"], seed=int(condition_row["jammer_seed"]), subband_fraction=float(config["jammer"]["subband_fraction"]), burst_fraction=float(config["jammer"]["burst_fraction"]), tone_count=int(config["jammer"]["tone_count"]))
                        oracle_risk = oracle_jamming_mask_to_risk_report(oracle.mask, masks.candidate_data, generated_tti=scored_tti - 1)
                    if jammer_aware_config["mode"] == "oracle_jamming_interference":
                        dummy = torch.ones((1, engine.profile.active_subcarriers, engine.profile.n_ofdm_symbols), dtype=torch.complex64, device=device)
                        oracle = build_r4_jammer(dummy, masks.candidate_data, jammer_type=_PHYSICAL_JAMMER_TYPE[condition_row["jammer_type"]], jsr_db=condition_row["jsr_db"], seed=int(condition_row["jammer_seed"]), subband_fraction=float(config["jammer"]["subband_fraction"]), burst_fraction=float(config["jammer"]["burst_fraction"]), tone_count=int(config["jammer"]["tone_count"]))
                        oracle_interference = oracle_jammer_grid_to_interference_report(
                            oracle.grid, masks.candidate_data, generated_tti=scored_tti - 1,
                            noise_power=1.0,
                        )
                    output_row = engine.forward(
                        target, channel_condition=condition, delayed_csi=report, training=False,
                        jammer_type=_PHYSICAL_JAMMER_TYPE[condition_row["jammer_type"]],
                        jammer_jsr_db=condition_row["jsr_db"], jammer_seed=int(condition_row["jammer_seed"]),
                        jammer_subband_fraction=float(config["jammer"]["subband_fraction"]),
                        jammer_burst_fraction=float(config["jammer"]["burst_fraction"]),
                        jammer_tone_count=int(config["jammer"]["tone_count"]),
                        delayed_re_risk=delayed_re_risk, oracle_jamming_risk=oracle_risk,
                        delayed_re_interference=delayed_re_interference,
                        oracle_jamming_interference=oracle_interference,
                    )
                    if not paired_warmup:
                        report = output_row.next_delayed_csi
                        delayed_re_risk = output_row.next_re_risk_report
                        delayed_re_interference = output_row.next_re_interference_report
                    estimate = estimator(output_row.received_resources, output_row.pilots, masks.pilot, output_row.estimated_channel, output_row.noise_variance)
                    refined = refiner(output_row.raw_reconstruction, output_row.decoder_state, estimate.mask_prob, estimate.posterior)
                    raw_waveform = codec.decode_representation(output_row.raw_reconstruction)
                    refined_waveform = codec.decode_representation(refined)
                    raw_sisdr = si_sdr(raw_waveform, waveform).mean()
                    refined_sisdr = si_sdr(refined_waveform, waveform).mean()
                    raw_nmse = per_layer_nmse(output_row.raw_reconstruction, target).mean()
                    refined_nmse = per_layer_nmse(refined, target).mean()
                    predicted = estimate.mask_prob >= float(config["mask_threshold"])
                    mask_metrics = mask_classification_metrics(predicted=predicted, target=output_row.jammer_mask.bool())
                    expected_index = JAMMER_TYPE_TO_INDEX[_PHYSICAL_JAMMER_TYPE[condition_row["jammer_type"]]]
                    mapping_hash = hashlib.sha256(output_row.mapping_indices.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                    metric_values = speech_metrics.evaluate(
                        reference=waveform.detach().cpu().numpy(),
                        estimate=raw_waveform.detach().cpu().numpy(),
                        sample_rate=sample_rate,
                        sample_id=str(condition_row["sample_id"]),
                        workspace=metric_workspace,
                    )
                    mapping_key = f"{condition_row['condition_hash']}:{risk_alpha}"
                    known = mapping_by_condition.setdefault(mapping_key, mapping_hash)
                    if known != mapping_hash:
                        raise ValueError(f"mapping differs across checkpoints for condition {condition_row['condition_hash']}")
                    row = {
                        **condition_row,
                        "jammer_aware_allocation_mode": jammer_aware_config["mode"],
                        "allocation_sequence_mode": "same_condition_warmup_tti_pair" if paired_warmup else "plan_row_sequence",
                        "checkpoint_name": checkpoint_name,
                        "checkpoint_step": int(checkpoint_payload["global_step"]),
                        "raw_si_sdr": float(raw_sisdr),
                        "raw_si_sdr_lt_minus10": bool(float(raw_sisdr) < -10.0),
                        "refined_si_sdr": float(refined_sisdr),
                        "delta_si_sdr_vs_raw": float(refined_sisdr - raw_sisdr),
                        "raw_latent_nmse": float(raw_nmse),
                        "refined_latent_nmse": float(refined_nmse),
                        "delta_latent_nmse": float(refined_nmse - raw_nmse),
                        "estimator_type_prediction": estimator.jammer_type_classes[int(estimate.posterior.argmax(dim=-1)[0])],
                        "estimator_type_accuracy": float((estimate.posterior.argmax(dim=-1) == expected_index).float().mean()),
                        "mask_iou": mask_metrics["iou"], "mask_f1": mask_metrics["f1"],
                        "mask_false_positive_rate": mask_metrics["false_positive_rate"],
                        "mask_false_negative_rate": mask_metrics["false_negative_rate"],
                        "mapping_hash": mapping_hash,
                        "allocation_selected_re_count": int(output_row.allocation.selected_re_count if hasattr(output_row.allocation, "selected_re_count") else output_row.mapping_indices.numel()),
                        "allocation_subband_count": int(getattr(output_row.allocation, "subband_count", 3)),
                        "allocation_distinct_subband_per_source": bool(getattr(output_row.allocation, "distinct_subband_per_source", True)),
                        "allocation_mean_assignment_sinr": float(getattr(output_row.allocation, "assignment_sinr", torch.zeros(1))[getattr(output_row.allocation, "assignment_sinr", torch.zeros(1)).gt(0)].mean()) if hasattr(output_row.allocation, "assignment_sinr") else None,
                        "raw_estoi": metric_values.get("raw_estoi"),
                        "raw_wer": metric_values.get("raw_wer"),
                        "raw_visqol_mos_lqo": metric_values.get("raw_visqol_mos_lqo"),
                    }
                    for index, label in enumerate(TRAINER_JAMMER_TYPE_CLASSES):
                        output_label = "broadband" if label == "broadband_awgn" else label
                        row[f"posterior_{output_label}"] = float(estimate.posterior[0, index])
                    if not all(torch.isfinite(torch.tensor(value)) for key, value in row.items() if key in {"raw_si_sdr", "refined_si_sdr", "raw_latent_nmse", "refined_latent_nmse"}):
                        raise FloatingPointError(f"nonfinite metric for {checkpoint_name}, {condition_row['condition_hash']}")
                    all_rows.append(row)
    checkpoint_names = list(payloads)
    verify_paired_row_keys(all_rows, checkpoint_names)
    fields = [
        "checkpoint_name", "checkpoint_step", "sample_id", "crop_offset", "snr_db", "jsr_db", "jammer_type", "realization_index", "channel_seed", "noise_seed", "jammer_seed", "allocation_risk_mode", "risk_alpha", "jammer_aware_allocation_mode", "allocation_sequence_mode",
        "raw_si_sdr", "raw_si_sdr_lt_minus10", "raw_estoi", "raw_wer", "raw_visqol_mos_lqo", "refined_si_sdr", "delta_si_sdr_vs_raw", "raw_latent_nmse", "refined_latent_nmse", "delta_latent_nmse",
        "estimator_type_prediction", "estimator_type_accuracy", "mask_iou", "mask_f1", "mask_false_positive_rate", "mask_false_negative_rate",
        "posterior_no_jammer", "posterior_broadband", "posterior_subband", "posterior_burst", "posterior_tone", "condition_hash", "mapping_hash", "allocation_selected_re_count", "allocation_subband_count", "allocation_distinct_subband_per_source", "allocation_mean_assignment_sinr",
    ]
    _write_csv(output / "per_condition_metrics.csv", all_rows, fields)
    summaries: dict[str, dict[str, float | str]] = {}
    type_rows: list[dict[str, Any]] = []
    confusion: dict[tuple[str, str, str], int] = defaultdict(int)
    for name in checkpoint_names:
        for risk_alpha in risk_alphas:
            scoped = [row for row in all_rows if row["checkpoint_name"] == name and float(row["risk_alpha"]) == float(risk_alpha)]
            summary_id = (
                f"{name}|{risk_config['risk_mode']}|alpha={float(risk_alpha):g}"
                f"|jammer_aware={jammer_aware_config['mode']}"
            )
            summary_metrics = (
                "raw_si_sdr", "refined_si_sdr", "delta_si_sdr_vs_raw", "raw_latent_nmse", "refined_latent_nmse", "delta_latent_nmse", "estimator_type_accuracy", "mask_iou", "mask_f1", "mask_false_positive_rate", "mask_false_negative_rate",
            )
            if metric_options.estoi_enabled:
                summary_metrics += ("raw_estoi",)
            if metric_options.wer_enabled:
                summary_metrics += ("raw_wer",)
            if metric_options.visqol_enabled:
                summary_metrics += ("raw_visqol_mos_lqo",)
            summary_metrics += ("raw_si_sdr_lt_minus10",)
            summaries[summary_id] = {field: _mean(scoped, field) for field in summary_metrics}
            summaries[summary_id]["checkpoint_step"] = int(payloads[name]["global_step"])
            summaries[summary_id]["checkpoint_name"] = name
            summaries[summary_id]["allocation_risk_mode"] = str(risk_config["risk_mode"])
            summaries[summary_id]["jammer_aware_allocation_mode"] = str(jammer_aware_config["mode"])
            summaries[summary_id]["risk_alpha"] = float(risk_alpha)
            for jammer_type in FIXED_JAMMER_TYPES:
                subset = [row for row in scoped if row["jammer_type"] == jammer_type]
                metric_fields = ("raw_si_sdr", "refined_si_sdr", "delta_si_sdr_vs_raw", "raw_latent_nmse", "refined_latent_nmse", "delta_latent_nmse", "estimator_type_accuracy", "mask_iou", "mask_f1", "mask_false_positive_rate", "mask_false_negative_rate")
                if metric_options.estoi_enabled:
                    metric_fields += ("raw_estoi",)
                if metric_options.wer_enabled:
                    metric_fields += ("raw_wer",)
                if metric_options.visqol_enabled:
                    metric_fields += ("raw_visqol_mos_lqo",)
                metric_fields += ("raw_si_sdr_lt_minus10",)
                type_rows.append({"checkpoint_name": name, "checkpoint_step": int(payloads[name]["global_step"]), "allocation_risk_mode": risk_config["risk_mode"], "jammer_aware_allocation_mode": jammer_aware_config["mode"], "risk_alpha": float(risk_alpha), "jammer_type": jammer_type, "rows": len(subset), **{field: _mean(subset, field) for field in metric_fields}})
                for row in subset:
                    confusion[(summary_id, jammer_type, row["estimator_type_prediction"])] += 1
    (output / "summary_by_checkpoint.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    summary_fields = ("raw_si_sdr", "refined_si_sdr", "delta_si_sdr_vs_raw", "raw_latent_nmse", "refined_latent_nmse", "delta_latent_nmse", "estimator_type_accuracy", "mask_iou", "mask_f1", "mask_false_positive_rate", "mask_false_negative_rate")
    if metric_options.estoi_enabled:
        summary_fields += ("raw_estoi",)
    if metric_options.wer_enabled:
        summary_fields += ("raw_wer",)
    if metric_options.visqol_enabled:
        summary_fields += ("raw_visqol_mos_lqo",)
    summary_fields += ("raw_si_sdr_lt_minus10",)
    _write_csv(output / "summary_by_jammer_type.csv", type_rows, ["checkpoint_name", "checkpoint_step", "allocation_risk_mode", "jammer_aware_allocation_mode", "risk_alpha", "jammer_type", "rows", *summary_fields])
    _write_csv(output / "confusion_matrix.csv", [
        {"checkpoint_variant": key[0], "true_jammer_type": key[1], "predicted_jammer_type": key[2], "count": value}
        for key, value in sorted(confusion.items())
    ], ["checkpoint_variant", "true_jammer_type", "predicted_jammer_type", "count"])
    if all((metric_options.estoi_enabled, metric_options.wer_enabled, metric_options.visqol_enabled)):
        perceptual_rows = summarize_perceptual_metrics(all_rows)
        _write_csv(
            output / "speech_quality_summary.csv", perceptual_rows,
            ["jammer_type", "jsr_db", "snr_db", "rows", "raw_estoi", "raw_wer", "raw_visqol_mos_lqo", "frac_raw_si_sdr_lt_minus10"],
        )
    (output / "validation_integrity.json").write_text(json.dumps({
        "paired_row_keys_equal": True,
        "mapping_hash_equal": True,
        "conditions_per_checkpoint": len(plan),
        "checkpoint_count": len(checkpoint_names),
        "total_rows": len(all_rows),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": sha256(source_checkpoint),
        "jammer_aware_allocation_mode": jammer_aware_config["mode"],
    }, indent=2, sort_keys=True) + "\n")
    if args.compare_existing_dir is not None or args.compare_interference_dir is not None:
        if args.compare_existing_dir is None or args.compare_interference_dir is None:
            raise ValueError("three-way comparison requires both existing and interference directories")
        existing_dir = Path(args.compare_existing_dir).resolve()
        interference_dir = Path(args.compare_interference_dir).resolve()
        comparison_dirs = [existing_dir, output, interference_dir]
        required = [directory / "per_condition_metrics.csv" for directory in comparison_dirs]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing allocation comparison metrics: {missing}")
        _verify_allocation_comparison_artifacts(comparison_dirs)
        comparison_rows = build_three_way_allocation_comparison_from_csv(*required)
        comparison_fields = [
            "condition_key", "jammer_type", "jsr_db", "snr_db",
            "si_sdr_existing_raw", "si_sdr_csi_only_raw", "si_sdr_interference_raw",
            "delta_csi_only_minus_existing", "delta_interference_minus_csi_only", "delta_interference_minus_existing",
            "si_sdr_existing_refined", "si_sdr_csi_only_refined", "si_sdr_interference_refined",
            "mapping_hash_existing", "mapping_hash_csi_only", "mapping_hash_interference",
        ]
        quality_comparison_fields: list[str] = []
        if comparison_rows and "raw_estoi_existing" in comparison_rows[0]:
            quality_comparison_fields = [
                f"{metric}_{mode}"
                for metric in ("raw_estoi", "raw_wer", "raw_visqol_mos_lqo")
                for mode in ("existing", "csi_only", "interference")
            ]
            comparison_fields.extend(quality_comparison_fields)
        _write_csv(output / "compare_existing_vs_csi_only_vs_interference.csv", comparison_rows, comparison_fields)
        summary_rows = summarize_three_way_allocation_comparison(comparison_rows)
        summary_fields = [
            "jammer_type", "jsr_db", "snr_db", "rows",
            "si_sdr_existing_raw", "si_sdr_csi_only_raw", "si_sdr_interference_raw",
            "delta_csi_only_minus_existing", "delta_interference_minus_csi_only", "delta_interference_minus_existing",
            "p5_delta_csi_minus_existing", "p5_delta_interference_minus_csi", "p5_delta_interference_minus_existing",
            "frac_delta_csi_minus_existing_lt_minus3", "frac_delta_interference_minus_csi_lt_minus3", "frac_delta_interference_minus_existing_lt_minus3",
            "frac_si_sdr_existing_lt_minus10", "frac_si_sdr_csi_only_lt_minus10", "frac_si_sdr_interference_lt_minus10",
        ]
        summary_fields.extend(quality_comparison_fields)
        _write_csv(output / "summary_by_condition.csv", summary_rows, summary_fields)


if __name__ == "__main__":
    main()


__all__ = [
    "FIXED_JAMMER_TYPES", "build_fixed_condition_plan", "condition_key",
    "allocation_comparison_key", "build_three_way_allocation_comparison",
    "build_three_way_allocation_comparison_from_csv",
    "mask_classification_metrics", "summarize_three_way_allocation_comparison",
    "verify_paired_row_keys",
]
