"""Evaluate the fixed SI-SDR-medium R4 checkpoint under deterministic jammers."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import torch
import yaml

from evaluate_r4_expanded_validation import _cache_codec_inputs, _conditions, _make_engine, _report_hash, _tensor_hash
from evaluate_r4_si_sdr_finetune_checkpoints import _checkpoint_payload, _repository_path, _source_rows
from speech_jscc.config import resolve_device
from speech_jscc.evaluation.expanded_validation import ManifestEntry, file_sha256
from speech_jscc.evaluation.r4_jammer_baseline import repetition_overlap_diagnostics, tensor_hash
from speech_jscc.evaluation.r4_si_sdr_checkpoint_selection import paired_statistics, statistical_tests
from speech_jscc.experiment import build_components
from speech_jscc.training.channel_free_revalidation import per_layer_nmse
from speech_jscc.training.r4_waveform_finetune import freeze_codec_for_input_gradient
from src.evaluation.waveform_metrics import waveform_metrics


ROOT = Path(__file__).resolve().parent


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _jsonl(path: Path, rows) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _parse_jsr(values: list[str]) -> list[float | None]:
    result = []
    for value in values:
        result.append(None if value == "no_jammer" else float(value))
    return result


def _seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _condition_hash(condition, report, mapping_hash: str, codec_hash: str, *, jammer_type: str, jsr_db, jammer_seed: int, jammer_mask_hash: str | None = None) -> str:
    value = {
        "snr_db": float(condition.snr_db), "tti": int(condition.tti), "noise_seed": int(condition.noise_seed),
        "taps": _tensor_hash(condition.tap_coefficients), "input_report": _report_hash(report),
        "mapping": mapping_hash, "codec": codec_hash, "jammer_type": jammer_type,
        "jsr_db": jsr_db, "jammer_seed": int(jammer_seed), "jammer_mask": jammer_mask_hash,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _finite(row: dict) -> bool:
    # ``-inf`` is the explicit no-jammer JSR sentinel, not a numerical model
    # failure.  Finite validation applies to decoded/channel quality metrics.
    ignored = {"target_jsr_db", "measured_pre_channel_jsr_db", "measured_post_channel_inr_db"}
    values = [value for key, value in row.items() if key not in ignored | {"per_layer_nmse", "jammer_statistics"} and isinstance(value, (int, float))]
    return all(math.isfinite(float(value)) for value in values)


def _scenario_rows(*, codec, medium_model, cache, conditions, csi_rows, specification, device, jammer_type: str, jsr_db: float | None, realization_index: int, realization_seed: int, jammer_config: dict, subband_fraction: float, burst_fraction: float) -> list[dict]:
    engine = _make_engine(codec, medium_model, specification)
    sample_rate = int(medium_model._expanded_validation_config["codec"]["sample_rate"])
    rows = []
    with torch.no_grad():
        for cached, condition, csi in zip(cache, conditions, csi_rows, strict=True):
            waveform, target = cached["waveform"].to(device), cached["target"].to(device)
            jammer_seed = _seed(cached["entry"].utterance_id, realization_seed, condition.snr_db, jammer_type, jsr_db, subband_fraction, burst_fraction)
            result = engine.forward(
                target, waveform, condition, csi["input_report"], training=False,
                jammer_type=jammer_type, jammer_jsr_db=jsr_db, jammer_seed=jammer_seed,
                jammer_subband_fraction=subband_fraction,
                jammer_burst_fraction=burst_fraction,
                jammer_tone_count=int(jammer_config["tone_count"]),
            )
            mapping_hash = _tensor_hash(result.mapping_indices)
            if mapping_hash != csi["mapping_hash"]:
                raise AssertionError("jammer condition changed the balanced-triplet mapping")
            metrics = waveform_metrics(waveform, result.decoded_waveform, sample_rate)
            layer = per_layer_nmse(result.reconstruction, target)
            jammer = result.jammer_statistics or {}
            if result.jammer_mask is None or result.jammer_grid is None:
                raise RuntimeError("R4 forward did not return jammer diagnostics")
            overlap = repetition_overlap_diagnostics(result.allocation, result.jammer_mask)
            copy_counts = torch.tensor(overlap["source_copy_count"], device=device, dtype=torch.long)
            symbol_error = (result.combined_symbols[0] - result.tx_symbols[0]).abs().square()
            symbol_signal = result.tx_symbols[0].abs().square().clamp_min(1e-12)
            by_copy = {str(count): (float(symbol_error[copy_counts == count].sum() / symbol_signal[copy_counts == count].sum()) if bool((copy_counts == count).any()) else None) for count in range(4)}
            signal_power = result.signal_received_time.abs().square().mean().clamp_min(1e-12)
            jammer_power = result.jammer_received_time.abs().square().mean()
            post_jsr = float(10 * torch.log10((jammer_power / signal_power).clamp_min(1e-12))) if jammer_type != "no_jammer" else None
            row = {
                "split": "legacy_final", "utterance_id": cached["entry"].utterance_id, "speaker_id": cached["entry"].speaker_id,
                "realization_index": realization_index, "realization_seed": realization_seed, "channel_seed": int(realization_seed),
                "awgn_seed": int(condition.noise_seed), "snr_db": float(condition.snr_db),
                "jammer_type": jammer_type, "target_jsr_db": jsr_db, "jammer_seed": jammer_seed,
                "subband_fraction": subband_fraction, "burst_fraction": burst_fraction,
                "mapping_hash": mapping_hash, "csi_report_hash": _report_hash(csi["input_report"]),
                "codec_input_hash": cached["codec_baseline_hash"], "target_waveform_hash": _tensor_hash(waveform),
                "jammer_mask_hash": jammer["jammer_mask_hash"], "jammer_tensor_hash": jammer["jammer_tensor_hash"],
                "jammer_active_re_fraction": jammer["active_re_fraction"], "jammer_active_subcarrier_fraction": jammer["active_subcarrier_fraction"],
                "jammer_active_time_fraction": jammer["active_time_fraction"],
                "measured_pre_channel_jsr_db": jammer["measured_pre_channel_jsr_db"], "measured_post_channel_inr_db": post_jsr,
                "effective_sinr_db": float(10 * torch.log10(result.effective_sinr)),
                "si_sdr_db": float(metrics["si_sdr_db"]), "waveform_snr_db": float(metrics["waveform_snr_db"]),
                "stft_l1": float(metrics["stft_l1"]), "aggregate_latent_nmse": float(layer.mean()),
                "per_layer_nmse": [float(x) for x in layer], "waveform_rms": float(metrics["output_rms"]),
                "silence_ratio": float((waveform.abs() < 1e-4).float().mean()),
                "copy_count_histogram": overlap["copy_count_histogram"],
                "copy_count_fraction": overlap["copy_count_fraction"],
                "copy_count_ge_2_fraction": overlap["copy_count_ge_2_fraction"],
                "per_layer_copy_count_histogram": overlap["per_layer_copy_count_histogram"],
                "post_mrc_symbol_nmse_by_jammed_copy_count": by_copy,
                "pairwise_frequency_separation": overlap["pairwise_frequency_separation"],
                "pairwise_time_separation": overlap["pairwise_time_separation"],
            }
            row["condition_hash"] = _condition_hash(condition, csi["input_report"], mapping_hash, cached["codec_baseline_hash"], jammer_type=jammer_type, jsr_db=jsr_db, jammer_seed=jammer_seed, jammer_mask_hash=row["jammer_mask_hash"])
            row["finite"] = _finite(row) and all(math.isfinite(x) for x in row["per_layer_nmse"])
            if not row["finite"]:
                raise FloatingPointError(f"nonfinite jammer evaluation result: {row['utterance_id']}")
            rows.append(row)
    return rows


def _group(rows, keys):
    result = {}
    for row in rows:
        result.setdefault(tuple(row[key] for key in keys), []).append(row)
    return result


def _summaries(rows, baseline_rows, bootstrap_samples: int, bootstrap_seed: int):
    baseline = {(r["utterance_id"], r["realization_index"], r["snr_db"]): r for r in baseline_rows}
    summaries, tails, worst = [], {}, {}
    for key, members in _group(rows, ("jammer_type", "target_jsr_db", "subband_fraction", "burst_fraction", "snr_db")).items():
        jammer_type, jsr_db, subband_fraction, burst_fraction, snr = key
        if jammer_type == "no_jammer":
            deltas = [0.0] * len(members)
        else:
            deltas = [r["si_sdr_db"] - baseline[(r["utterance_id"], r["realization_index"], r["snr_db"])]["si_sdr_db"] for r in members]
        utterance = {}
        for row, delta in zip(members, deltas, strict=True):
            utterance.setdefault(row["utterance_id"], []).append(delta)
        values = [sum(value) / len(value) for value in utterance.values()]
        stats = paired_statistics(values, samples=bootstrap_samples, seed=bootstrap_seed + round(float(snr) * 100))
        for row, delta in zip(members, deltas, strict=True):
            row["delta_si_sdr_vs_no_jammer_db"] = delta
        summary = {
            "jammer_type": jammer_type, "target_jsr_db": jsr_db, "subband_fraction": subband_fraction, "burst_fraction": burst_fraction, "snr_db": snr,
            "rows": len(members), "mean_si_sdr_db": sum(r["si_sdr_db"] for r in members) / len(members),
            "mean_waveform_snr_db": sum(r["waveform_snr_db"] for r in members) / len(members),
            "mean_stft_l1": sum(r["stft_l1"] for r in members) / len(members),
            "mean_latent_nmse": sum(r["aggregate_latent_nmse"] for r in members) / len(members),
            "mean_effective_sinr_db": sum(r["effective_sinr_db"] for r in members) / len(members),
            "paired_vs_no_jammer": stats,
        }
        summaries.append(summary); label = str(key); tails[label] = stats
        row_deltas = sorted(zip(members, deltas, strict=True), key=lambda item: item[1])
        worst[label] = [{**row, "delta_si_sdr_vs_no_jammer_db": delta} for row, delta in row_deltas[:5]]
    return summaries, tails, worst


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = ["jammer_type", "target_jsr_db", "subband_fraction", "burst_fraction", "snr_db", "rows", "mean_si_sdr_db", "mean_waveform_snr_db", "mean_stft_l1", "mean_latent_nmse", "mean_effective_sinr_db"]
    path.write_text(",".join(columns) + "\n" + "".join(",".join(str(row.get(column, "")) for column in columns) + "\n" for row in rows))


def clean_anchor_status(*, observed: float, reference: float, tolerance: float, device: str, utterances: int, realizations: int, full_utterances: int, full_realizations: int) -> dict:
    """Evaluate the historical clean anchor only at its original sample scope."""
    result = {
        "reference_si_sdr_db": reference, "observed_si_sdr_db": observed,
        "tolerance_db": tolerance, "device": device,
        "evaluated_utterances": utterances, "evaluated_realizations": realizations,
    }
    if utterances != full_utterances or realizations != full_realizations:
        return {**result, "pass": None, "status": "NOT_APPLICABLE_SUBSET"}
    if device != "cuda":
        return {**result, "pass": None, "status": "NOT_COMPARABLE_CPU_PROTOCOL"}
    passed = abs(observed - reference) <= tolerance
    return {**result, "pass": passed, "status": "PASS_CUDA_REFERENCE" if passed else "FAIL_REFERENCE_MISMATCH"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval_r4_jammer_baseline.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=("legacy_final",), default="legacy_final")
    parser.add_argument("--jammer-types", nargs="+", default=None)
    parser.add_argument("--jsr-db", nargs="+", default=None)
    parser.add_argument("--snr-db", nargs="+", type=float, default=None)
    parser.add_argument("--subband-fractions", nargs="+", type=float, default=None)
    parser.add_argument("--burst-fractions", nargs="+", type=float, default=None)
    parser.add_argument("--minimum-copy-time-separation-symbols", type=int, default=None,
                        help="Opt-in time-interleaved triplet mapping; 0 preserves the clean baseline mapping.")
    parser.add_argument("--max-utterances", type=int)
    parser.add_argument("--max-realizations", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir")
    parser.add_argument("--allow-long-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); config = yaml.safe_load(_repository_path(args.config).read_text())
    checkpoint = _repository_path(args.checkpoint or config["medium_checkpoint"])
    if not checkpoint.is_file(): raise SystemExit(f"medium baseline checkpoint does not exist: {checkpoint}")
    payload = _checkpoint_payload(checkpoint)
    if payload.get("diagnostic_type") != "r4_si_sdr_finetune" or int(payload.get("local_step", -1)) != 3000:
        raise SystemExit("jammer baseline requires the fixed si_sdr_medium local_step_003000 checkpoint")
    if args.dry_run:
        print(json.dumps({"checkpoint": str(checkpoint), "split": args.split, "jammer_types": args.jammer_types or config["jammer_types"], "jsr_db": args.jsr_db or config["jsr_db"], "minimum_copy_time_separation_symbols": args.minimum_copy_time_separation_symbols if args.minimum_copy_time_separation_symbols is not None else config.get("mapping", {}).get("minimum_copy_time_separation_symbols", 0), "cuda_execution_required_for_full_run": args.device == "cuda"}, indent=2)); return
    utterances = args.max_utterances or int(config["evaluation"]["utterances"])
    realizations = args.max_realizations or int(config["evaluation"]["realizations"])
    snrs = args.snr_db or [float(x) for x in config["evaluation"]["snr_db"]]
    if utterances * realizations * len(snrs) > int(config["evaluation"]["smoke_max_rows"]) and not args.allow_long_run:
        raise SystemExit("full jammer baseline requires --allow-long-run")
    output = _repository_path(args.output_dir or config["output_root"])
    if output.exists():
        if not args.overwrite: raise SystemExit(f"refusing existing output directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    model_config = copy.deepcopy(payload["config"])
    for key in ("config_path", "checkpoint_path"):
        path = Path(model_config["codec"][key]); model_config["codec"][key] = str(_repository_path(path)) if not path.is_absolute() else str(path)
    model_config["device"] = args.device; device = resolve_device(args.device)
    codec, medium = build_components(model_config, device); medium.load_state_dict(payload["model"], strict=True)
    source_path = _repository_path(config["csi_reference_checkpoint"]); source_payload = _checkpoint_payload(source_path)
    source = copy.deepcopy(medium); source.load_state_dict(source_payload["model"], strict=True)
    freeze_codec_for_input_gradient(codec); medium.eval().requires_grad_(False); source.eval().requires_grad_(False)
    medium._expanded_validation_config = model_config; source._expanded_validation_config = model_config
    manifest = _repository_path(config["legacy_final_manifest"])
    entries = [ManifestEntry(**json.loads(line)) for line in manifest.read_text().splitlines() if line.strip()][:utterances]
    if len(entries) != utterances: raise ValueError("legacy manifest shorter than requested")
    physical = copy.deepcopy(config["physical"])
    minimum_time_separation = (
        int(args.minimum_copy_time_separation_symbols)
        if args.minimum_copy_time_separation_symbols is not None
        else int(config.get("mapping", {}).get("minimum_copy_time_separation_symbols", 0))
    )
    physical["minimum_copy_time_separation_symbols"] = minimum_time_separation
    specification = {"physical_profile_config": str(_repository_path(config["physical_profile_config"])), "physical": physical, "pairing": {"csi_report_tolerance": 1e-5}}
    cache = _cache_codec_inputs(codec, entries, model_config, device)
    all_rows=[]; csi_engine=_make_engine(codec, source, specification)
    jammer_types=args.jammer_types or list(config["jammer_types"]); jsrs=_parse_jsr(args.jsr_db or list(config["jsr_db"]))
    subband_fractions=args.subband_fractions or list(config["jammer"]["subband_fractions"])
    burst_fractions=args.burst_fractions or list(config["jammer"]["burst_fractions"])
    scenarios=[("no_jammer", None, float(config["jammer"]["subband_fraction"]), float(config["jammer"]["burst_fraction"]))]
    for kind in jammer_types:
        fractions = [(sub, float(config["jammer"]["burst_fraction"])) for sub in subband_fractions] if kind == "subband" else [(float(config["jammer"]["subband_fraction"]), burst) for burst in burst_fractions] if kind == "burst" else [(sub, burst) for sub in subband_fractions for burst in burst_fractions] if kind == "block" else [(float(config["jammer"]["subband_fraction"]), float(config["jammer"]["burst_fraction"]))]
        scenarios.extend((kind, jsr, sub, burst) for jsr in jsrs if jsr is not None for sub, burst in fractions)
    base_rows=[]
    seeds=[int(value) for value in config["realization_seeds"]][:realizations]
    for realization_index, realization_seed in enumerate(seeds):
        for snr in snrs:
            conditions = _conditions(specification, csi_engine, snr=snr, seed=realization_seed, count=len(cache))
            csi_rows = _source_rows(codec=codec, model=source, cache=cache, conditions=conditions, specification=specification, device=device)
            for jammer_type, jsr, subband_fraction, burst_fraction in scenarios:
                rows = _scenario_rows(codec=codec, medium_model=medium, cache=cache, conditions=conditions, csi_rows=csi_rows, specification=specification, device=device, jammer_type=jammer_type, jsr_db=jsr, realization_index=realization_index, realization_seed=realization_seed, jammer_config=config["jammer"], subband_fraction=subband_fraction, burst_fraction=burst_fraction)
                all_rows.extend(rows)
                if jammer_type == "no_jammer": base_rows.extend(rows)
    summaries, tails, worst = _summaries(all_rows, base_rows, int(config["bootstrap_samples"]), int(config["bootstrap_seed"]))
    anchor = next(x for x in summaries if x["jammer_type"] == "no_jammer" and x["snr_db"] == 5.0)
    reference = json.loads(_repository_path(config["clean_reference_summary"]).read_text())["si_sdr_medium"]["5"]
    anchor_check=clean_anchor_status(observed=anchor["mean_si_sdr_db"], reference=reference, tolerance=float(config["clean_anchor_tolerance_db"]), device=args.device, utterances=utterances, realizations=realizations, full_utterances=int(config["evaluation"]["utterances"]), full_realizations=int(config["evaluation"]["realizations"]))
    if minimum_time_separation > 0:
        anchor_check = {
            **anchor_check,
            "pass": None,
            "status": "NOT_APPLICABLE_MAPPING_CHANGED",
            "minimum_copy_time_separation_symbols": minimum_time_separation,
        }
    if anchor_check["status"] == "FAIL_REFERENCE_MISMATCH":
        raise RuntimeError(f"no_jammer clean anchor failed: {anchor_check}")
    _jsonl(output/"per_sample_jammer_results.jsonl", all_rows); _write_csv(output/"per_condition_summary.csv", summaries)
    _json(output/"clean_anchor_check.json",anchor_check); _json(output/"paired_statistics.json",tails); _json(output/"tail_risk_summary.json",tails); _json(output/"worst_samples.json",worst)
    _json(output/"statistical_tests.json", statistical_tests({key:[r["delta_si_sdr_vs_no_jammer_db"] for r in all_rows if r["jammer_type"] != "no_jammer" and str((r["jammer_type"],r["target_jsr_db"],r["subband_fraction"],r["burst_fraction"],r["snr_db"]))==key] for key in tails if not key.startswith("('no_jammer'")}))
    _json(output/"jammer_mask_summary.json", [{key:value for key,value in row.items() if key.startswith("jammer_") or key in ("target_jsr_db","snr_db")} for row in all_rows])
    _json(output/"jsr_calibration_summary.json", [{key:row[key] for key in ("jammer_type","target_jsr_db","measured_pre_channel_jsr_db","measured_post_channel_inr_db","effective_sinr_db","jammer_mask_hash","jammer_tensor_hash")} for row in all_rows])
    layer={}
    for key,members in _group(all_rows,("jammer_type","target_jsr_db","subband_fraction","burst_fraction","snr_db")).items(): layer[str(key)]={"per_layer_nmse":[sum(r["per_layer_nmse"][i] for r in members)/len(members) for i in range(8)]}
    _json(output/"layer_damage_summary.json",layer)
    repetition={}
    for key,members in _group(all_rows,("jammer_type","target_jsr_db","subband_fraction","burst_fraction","snr_db")).items():
        total={str(count):sum(int(r["copy_count_histogram"][str(count)]) for r in members) for count in range(4)}
        buckets={"zero":[], "low_le_0.25":[], "medium_le_0.50":[], "high_gt_0.50":[]}
        for row in members:
            exposure=float(row["copy_count_ge_2_fraction"])
            name="zero" if exposure == 0.0 else "low_le_0.25" if exposure <= .25 else "medium_le_0.50" if exposure <= .50 else "high_gt_0.50"
            buckets[name].append(row)
        conditional={name:{"rows":len(values),"mean_si_sdr_db":sum(x["si_sdr_db"] for x in values)/len(values),"mean_aggregate_latent_nmse":sum(x["aggregate_latent_nmse"] for x in values)/len(values),"mean_delta_si_sdr_vs_no_jammer_db":sum(x["delta_si_sdr_vs_no_jammer_db"] for x in values)/len(values)} for name,values in buckets.items() if values}
        repetition[str(key)]={"copy_count_histogram":total,"copy_count_fraction":{name:value/(len(members)*1920.0) for name,value in total.items()},"copy_count_ge_2_fraction":(total["2"]+total["3"])/(len(members)*1920.0),"mean_post_mrc_symbol_nmse_by_jammed_copy_count":{str(count):sum(r["post_mrc_symbol_nmse_by_jammed_copy_count"][str(count)] for r in members if r["post_mrc_symbol_nmse_by_jammed_copy_count"][str(count)] is not None)/max(1,sum(r["post_mrc_symbol_nmse_by_jammed_copy_count"][str(count)] is not None for r in members)) for count in range(4)},"conditional_waveform_by_copy_ge_2_exposure":conditional,"pairwise_frequency_separation":members[0]["pairwise_frequency_separation"],"pairwise_time_separation":members[0]["pairwise_time_separation"]}
    _json(output/"repetition_damage_summary.json",repetition)
    _json(output/"effective_sinr_summary.json",[{key:row[key] for key in ("jammer_type","target_jsr_db","snr_db","mean_effective_sinr_db")} for row in summaries])
    _json(output/"checkpoint_manifest.json",{"role":"medium_baseline","path":str(checkpoint),"local_step":payload.get("local_step",3000),"source_global_step":payload.get("source_global_step"),"target_si_sdr_weight":0.02,"checkpoint_type":payload.get("diagnostic_type")})
    codec_checkpoint = Path(model_config["codec"]["checkpoint_path"])
    _json(output/"checkpoint_hashes.json",{"medium_checkpoint_sha256":file_sha256(checkpoint),"csi_reference_checkpoint_sha256":file_sha256(source_path),"speech_tokenizer_checkpoint_sha256":file_sha256(codec_checkpoint) if codec_checkpoint.is_file() else None})
    resolved_config = copy.deepcopy(config)
    resolved_config["mapping"] = {
        **resolved_config.get("mapping", {}),
        "minimum_copy_time_separation_symbols": minimum_time_separation,
    }
    _json(output/"jammer_config_resolved.json", {**config["jammer"], "mapping": resolved_config["mapping"]}); (output/"resolved_config.yaml").write_text(yaml.safe_dump(resolved_config,sort_keys=False)); (output/"command.txt").write_text(" ".join(sys.argv)+"\n"); _json(output/"environment.json",{"device":args.device,"torch":torch.__version__})
    (output/"jammer_baseline_report.md").write_text("# R4 uniform RE/uniform-power jammer baseline\n\n"+json.dumps({"clean_anchor":anchor_check,"rows":len(all_rows),"scenarios":scenarios},indent=2)+"\n")
    print(json.dumps({"output":str(output),"rows":len(all_rows),"clean_anchor":anchor_check},indent=2))


if __name__ == "__main__": main()
