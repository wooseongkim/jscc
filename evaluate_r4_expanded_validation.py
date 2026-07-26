from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

from channels.multipath import exponential_pdp
from channels.temporal_multipath import correlated_tap_trajectory, jakes_slot_correlation
from speech_jscc.config import resolve_device
from speech_jscc.evaluation.expanded_validation import (
    ManifestEntry,
    audit_protocol_overlap,
    build_seed_manifest,
    build_selection_manifest,
    checkpoint_gate,
    discover_candidates,
    entries_from_paths,
    explicit_metric_row,
    file_sha256,
    paired_statistics,
    prepare_final_test_directory,
    prepare_output_directory,
    rank_candidates,
    shared_report_for_candidate,
    utterance_level_rows,
    write_selected_checkpoint,
)
from speech_jscc.experiment import build_components
from speech_jscc.training.channel_free_revalidation import (
    per_layer_nmse,
    summed_latent_statistics,
)
from speech_jscc.training.r4_waveform_finetune import (
    R4ForwardCondition,
    R4WaveformForward,
    freeze_codec_for_input_gradient,
    validate_initial_checkpoint_metadata,
)
from src.evaluation.waveform_metrics import waveform_metrics
from train_channel_free_conv_conformer import fixed_paths, load_batch


EXPLICIT_DELTA_METRICS = (
    "delta_si_sdr_vs_initial_r4_db",
    "delta_waveform_snr_vs_initial_r4_db",
    "delta_stft_ratio_vs_initial_r4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/eval_r4_expanded_validation.yaml"
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--max-utterances", type=int)
    parser.add_argument("--max-realizations", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-final-test", action="store_true")
    parser.add_argument("--allow-long-run", action="store_true")
    return parser.parse_args()


def _git_metadata() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=False, capture_output=True, text=True
        ).stdout
    )
    return {"commit": commit, "working_tree_dirty": dirty}


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _manifest_metadata(entries: list[ManifestEntry]) -> dict:
    encoded = "\n".join(
        json.dumps(entry.to_dict(), sort_keys=True) for entry in entries
    ).encode()
    return {
        "utterances": len(entries),
        "speakers": len({entry.speaker_id for entry in entries}),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "source_split": sorted({entry.source_split for entry in entries}),
        "assigned_evaluation_role": sorted(
            {entry.assigned_evaluation_role for entry in entries}
        ),
    }


def _checkpoint_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in payload:
        raise ValueError(f"checkpoint has no model state: {path}")
    return payload


def _report_hash(report) -> str | None:
    if report is None:
        return None
    tensor = report.reliability.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _cache_codec_inputs(codec, entries, model_config, device):
    cache = []
    sample_rate = int(model_config["codec"]["sample_rate"])
    with torch.no_grad():
        for entry in entries:
            waveform = load_batch([Path(entry.source_path)], model_config, device)
            target = codec.encode_waveform(waveform)
            clean_waveform = codec.decode_representation(target)
            clean_metrics = waveform_metrics(
                waveform, clean_waveform, sample_rate
            )
            cache.append(
                {
                    "entry": entry,
                    "waveform": waveform.cpu(),
                    "target": target.cpu(),
                    "clean_metrics": clean_metrics,
                    "codec_baseline_hash": _tensor_hash(clean_waveform),
                }
            )
    return cache


def _make_engine(codec, model, specification):
    return R4WaveformForward(
        codec,
        model,
        estimator_ridge_lambda=float(
            specification["physical"]["estimator_ridge_lambda"]
        ),
        epsilon=float(specification["physical"]["epsilon"]),
    )


def _conditions(specification, engine, *, snr, seed, count):
    physical = yaml.safe_load(
        Path(specification["physical_profile_config"]).read_text()
    )
    pdp = exponential_pdp(
        physical["channel"]["num_taps"], physical["channel"]["pdp_decay"]
    )
    rho = jakes_slot_correlation(
        physical["physical"]["user_speed_mps"],
        physical["physical"]["carrier_frequency_hz"],
        engine.profile.tti_duration_s,
    )
    trajectory = correlated_tap_trajectory(
        slots=count,
        batch_size=1,
        pdp=pdp,
        rho=rho,
        seed=int(seed) + round(float(snr) * 100),
    )
    variance = 10.0 ** (-float(snr) / 10.0)
    return [
        R4ForwardCondition(
            snr_db=float(snr),
            tti=tti,
            tap_coefficients=trajectory[tti],
            noise_seed=int(seed) + tti + round(float(snr) * 1000),
            noise_variance_override=variance,
        )
        for tti in range(count)
    ]


def _initial_trajectory(
    codec,
    initial_model,
    cache,
    conditions,
    specification,
    device,
):
    engine = _make_engine(codec, initial_model, specification)
    input_reports = []
    rows = []
    report = None
    sample_rate = int(initial_model._expanded_validation_config["codec"]["sample_rate"])
    with torch.no_grad():
        for cached, condition in zip(cache, conditions):
            input_reports.append(report)
            waveform = cached["waveform"].to(device)
            target = cached["target"].to(device)
            result = engine.forward(
                target, waveform, condition, report, training=False
            )
            metrics = waveform_metrics(
                waveform, result.decoded_waveform, sample_rate
            )
            rows.append(
                {
                    "metrics": metrics,
                    "input_report_hash": _report_hash(report),
                    "next_report": result.next_delayed_csi,
                    "mapping_hash": _tensor_hash(result.mapping_indices),
                }
            )
            report = result.next_delayed_csi
    return input_reports, rows


def _candidate_rows(
    *,
    codec,
    model,
    candidate_path,
    cache,
    conditions,
    input_reports,
    initial_rows,
    specification,
    device,
    snr,
    realization,
):
    engine = _make_engine(codec, model, specification)
    sample_rate = int(model._expanded_validation_config["codec"]["sample_rate"])
    rows = []
    with torch.no_grad():
        for tti, (cached, condition, shared_report, initial) in enumerate(
            zip(cache, conditions, input_reports, initial_rows)
        ):
            waveform = cached["waveform"].to(device)
            target = cached["target"].to(device)
            result = engine.forward(
                target,
                waveform,
                condition,
                shared_report,
                training=False,
            )
            candidate_report = result.next_delayed_csi
            used_report = shared_report_for_candidate(
                shared_report, candidate_generated_report=candidate_report
            )
            if used_report is not shared_report:
                raise AssertionError("candidate replaced the shared delayed CSI report")
            if _report_hash(shared_report) != initial["input_report_hash"]:
                raise AssertionError("candidate delayed-CSI input differs from initial R4")
            mapping_hash = _tensor_hash(result.mapping_indices)
            if mapping_hash != initial["mapping_hash"]:
                raise AssertionError("paired allocation map differs from initial R4")
            report_difference = float(
                (
                    candidate_report.reliability
                    - initial["next_report"].reliability
                )
                .abs()
                .max()
            )
            if report_difference > float(
                specification["pairing"]["csi_report_tolerance"]
            ):
                raise AssertionError(
                    "candidate receiver CSI report is not channel-observation-only: "
                    f"max_difference={report_difference}"
                )
            current = waveform_metrics(
                waveform, result.decoded_waveform, sample_rate
            )
            metrics = explicit_metric_row(
                candidate=current,
                initial=initial["metrics"],
                clean=cached["clean_metrics"],
            )
            layer = per_layer_nmse(result.reconstruction, target)
            summed = summed_latent_statistics(result.reconstruction, target)
            if not all(math.isfinite(float(value)) for value in metrics.values()):
                raise FloatingPointError(
                    "nonfinite expanded-validation metric "
                    f"checkpoint={candidate_path} utterance={cached['entry'].utterance_id} "
                    f"realization={realization} snr={snr} "
                    f"seed={condition.noise_seed}"
                )
            rows.append(
                {
                    "candidate_checkpoint": str(candidate_path),
                    "utterance_id": cached["entry"].utterance_id,
                    "speaker_id": cached["entry"].speaker_id,
                    "source_path": cached["entry"].source_path,
                    "snr_db": float(snr),
                    "realization": int(realization),
                    "tti": tti,
                    "channel_seed": int(
                        condition.noise_seed
                        - tti
                        - round(float(snr) * 1000)
                        + round(float(snr) * 100)
                    ),
                    "noise_seed": int(condition.noise_seed),
                    "input_delayed_csi_hash": _report_hash(shared_report),
                    "candidate_csi_report_max_abs_difference": report_difference,
                    "mapping_hash": mapping_hash,
                    "codec_baseline_hash": cached["codec_baseline_hash"],
                    **metrics,
                    "aggregate_layer_nmse": float(layer.mean()),
                    "summed_latent_nmse": float(summed["nmse"]),
                    "per_layer_nmse": [float(value) for value in layer],
                    "csi_nmse": float(result.csi_nmse),
                    "pilot_evm": float(result.pilot_evm),
                    "effective_sinr_db": float(
                        10 * torch.log10(result.effective_sinr)
                    ),
                }
            )
    return rows


def _mean(rows, key):
    return sum(float(row[key]) for row in rows) / len(rows)


def summarize_candidate(rows, inventory, specification):
    metric_keys = (
        *EXPLICIT_DELTA_METRICS,
        "si_sdr_absolute_db",
        "delta_si_sdr_vs_clean_codec_db",
        "waveform_snr_absolute_db",
        "delta_waveform_snr_vs_clean_codec_db",
        "stft_l1_absolute",
        "stft_ratio_vs_clean_codec",
        "aggregate_layer_nmse",
        "summed_latent_nmse",
        "csi_nmse",
        "pilot_evm",
        "effective_sinr_db",
    )
    utterance_rows = utterance_level_rows(rows, metric_keys=metric_keys)
    by_snr = {}
    for snr in map(float, specification["full_selection_validation"]["snr_db"]):
        realization_members = [row for row in rows if row["snr_db"] == snr]
        utterance_members = [row for row in utterance_rows if row["snr_db"] == snr]
        realization_statistics = {
            key: paired_statistics(
                [row[key] for row in realization_members],
                bootstrap_samples=int(specification["bootstrap"]["samples"]),
                bootstrap_seed=int(specification["bootstrap"]["seed"])
                + round(snr * 100)
                + index,
            )
            for index, key in enumerate(metric_keys)
        }
        utterance_statistics = {
            key: paired_statistics(
                [row[key] for row in utterance_members],
                bootstrap_samples=int(specification["bootstrap"]["samples"]),
                bootstrap_seed=int(specification["bootstrap"]["seed"])
                + 10_000
                + round(snr * 100)
                + index,
            )
            for index, key in enumerate(metric_keys)
        }
        by_snr[str(snr)] = {
            "realization_level": realization_statistics,
            "utterance_level": utterance_statistics,
            **{key: _mean(utterance_members, key) for key in metric_keys},
        }
    gate = checkpoint_gate(by_snr)
    return {
        **inventory,
        "by_snr": by_snr,
        "clean_gate_pass": gate["passed"],
        "gate_normalized_margins": gate["normalized_margins"],
        "gate_normalized_minimum_margin": gate["normalized_minimum_margin"],
        "pairing_audit": {
            "maximum_candidate_csi_report_difference": max(
                row["candidate_csi_report_max_abs_difference"] for row in rows
            ),
            "unique_mapping_hashes": len({row["mapping_hash"] for row in rows}),
            "candidate_order_independent": True,
        },
    }


def _load_light_score(payload):
    validation = payload.get("validation", {})
    value = validation.get("5db_delta_si_sdr_vs_initial_r4")
    return float(value) if value is not None else None


def evaluate_candidates(
    *,
    candidate_inventory,
    entries,
    specification,
    initial_payload,
    codec,
    initial_model,
    device,
    realization_seeds,
):
    model_config = copy.deepcopy(initial_payload["config"])
    cache = _cache_codec_inputs(codec, entries, model_config, device)
    all_rows = []
    summaries = []
    initial_cache = {}
    for snr in map(float, specification["full_selection_validation"]["snr_db"]):
        for realization, seed in enumerate(realization_seeds):
            conditions = _conditions(
                specification,
                _make_engine(codec, initial_model, specification),
                snr=snr,
                seed=seed,
                count=len(entries),
            )
            initial_cache[(snr, realization)] = (
                conditions,
                *_initial_trajectory(
                    codec,
                    initial_model,
                    cache,
                    conditions,
                    specification,
                    device,
                ),
            )
    for inventory in candidate_inventory:
        candidate_path = Path(inventory["checkpoint_path"])
        payload = _checkpoint_payload(candidate_path)
        model = copy.deepcopy(initial_model)
        model.load_state_dict(payload["model"], strict=True)
        model.eval().requires_grad_(False)
        model._expanded_validation_config = model_config
        rows = []
        for snr in map(float, specification["full_selection_validation"]["snr_db"]):
            for realization, _seed in enumerate(realization_seeds):
                conditions, reports, initial_rows = initial_cache[(snr, realization)]
                rows.extend(
                    _candidate_rows(
                        codec=codec,
                        model=model,
                        candidate_path=candidate_path,
                        cache=cache,
                        conditions=conditions,
                        input_reports=reports,
                        initial_rows=initial_rows,
                        specification=specification,
                        device=device,
                        snr=snr,
                        realization=realization,
                    )
                )
        metadata = {
            **inventory,
            "light_validation_score": _load_light_score(payload),
            "checkpoint_sha256": file_sha256(candidate_path),
        }
        summaries.append(summarize_candidate(rows, metadata, specification))
        all_rows.extend(rows)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return all_rows, summaries


def _summary_csv_rows(summaries):
    rows = []
    for summary in summaries:
        row = {
            "checkpoint_path": summary["checkpoint_path"],
            "global_step": summary["global_step"],
            "training_stage": summary["training_stage"],
            "light_validation_score": summary["light_validation_score"],
            "clean_gate_pass": summary["clean_gate_pass"],
            "gate_normalized_minimum_margin": summary[
                "gate_normalized_minimum_margin"
            ],
        }
        for snr in ("5.0", "10.0", "15.0"):
            stats = summary["by_snr"][snr]["utterance_level"][
                "delta_si_sdr_vs_initial_r4_db"
            ]
            row[f"{snr}_mean_delta_si_sdr"] = stats["mean"]
            row[f"{snr}_ci_low"] = stats["paired_mean_bootstrap_ci95"][0]
            row[f"{snr}_ci_high"] = stats["paired_mean_bootstrap_ci95"][1]
        row["clean_codec_delta_si_sdr_5db"] = summary["by_snr"]["5.0"][
            "delta_si_sdr_vs_clean_codec_db"
        ]
        rows.append(row)
    return rows


def _write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(output, audit, candidates, ranked, decision, final_executed):
    lines = [
        "# R4 Expanded Validation Report",
        "",
        "## Protocol audit",
        "",
        "- Legacy validation used 6 utterances for light validation and 16 for full validation.",
        "- Both historical validation sets overlapped the legacy 64-utterance final suite.",
        "- Historical light and full validation could both update named best checkpoints.",
        f"- New speaker-overlap audit passed: `{audit['passed']}`.",
        "- `test-clean` is assigned only to `selection_validation`; these are not test results.",
        "",
        "## Candidates",
        "",
    ]
    for row in candidates:
        lines.append(
            f"- `{row['checkpoint_path']}`: step={row['global_step']}, "
            f"stage={row['training_stage']}"
        )
    lines.extend(
        [
            "",
            "## Ranking",
            "",
            "| rank | checkpoint | gate | 5 dB paired mean | 5 dB CI |",
            "|---:|---|:---:|---:|---:|",
        ]
    )
    for row in ranked:
        stats = row["by_snr"]["5.0"]["utterance_level"][
            "delta_si_sdr_vs_initial_r4_db"
        ]
        lines.append(
            f"| {row['rank']} | `{row['checkpoint_path']}` | "
            f"{'PASS' if row['clean_gate_pass'] else 'FAIL'} | "
            f"{stats['mean']:.4f} | "
            f"[{stats['paired_mean_bootstrap_ci95'][0]:.4f}, "
            f"{stats['paired_mean_bootstrap_ci95'][1]:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Selected: `{decision['checkpoint_path']}`",
            f"- Status: `{decision['selection_status']}`",
            f"- Clean gate pass: `{decision['clean_gate_pass']}`",
            f"- Legacy final executed: `{final_executed}`",
            "- Jammer and layer-aware follow-on work remain blocked until external expanded validation completes.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines))


def _run_final_test_only(args, specification, output_path: Path) -> None:
    if not args.allow_long_run:
        raise SystemExit("--run-final-test requires --allow-long-run")
    final_dir = prepare_final_test_directory(
        output_path, overwrite=args.overwrite
    )
    selected_path = output_path / "best_expanded_validation.pt"
    selected_payload = _checkpoint_payload(selected_path)
    decision = json.loads((output_path / "selection_decision.json").read_text())
    final_entries = [
        ManifestEntry(**json.loads(line))
        for line in (output_path / "final_test_manifest_reference.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    if len(final_entries) != int(specification["legacy_final"]["utterances"]):
        raise ValueError("legacy final manifest is not the frozen 64-utterance suite")
    model_config = copy.deepcopy(selected_payload["config"])
    model_config["device"] = args.device or specification["device"]
    device = resolve_device(model_config["device"])
    codec, initial_model = build_components(model_config, device)
    freeze_codec_for_input_gradient(codec)
    initial_payload = _checkpoint_payload(Path(specification["initial_checkpoint"]))
    initial_model.load_state_dict(initial_payload["model"], strict=True)
    initial_model.eval().requires_grad_(False)
    initial_model._expanded_validation_config = model_config
    inventory = [
        {
            "checkpoint_path": str(selected_path),
            "global_step": selected_payload.get("global_step"),
            "training_stage": selected_payload.get("curriculum_stage"),
            "candidate_source": "best_expanded_validation",
        }
    ]
    started = time.time()
    with torch.no_grad():
        rows, summaries = evaluate_candidates(
            candidate_inventory=inventory,
            entries=final_entries,
            specification=specification,
            initial_payload=initial_payload,
            codec=codec,
            initial_model=initial_model,
            device=device,
            realization_seeds=list(
                specification["legacy_final"]["realization_seeds"]
            ),
        )
    _write_jsonl(final_dir / "per_sample_metrics.jsonl", rows)
    summary = {
        **summaries[0],
        "evaluation_role": "legacy_final_test",
        "used_for_checkpoint_selection": False,
        "selection_decision_reference": decision,
    }
    (final_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (final_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")
    (final_dir / "runtime_summary.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": time.time() - started,
                "utterances": len(final_entries),
                "realizations": len(
                    specification["legacy_final"]["realization_seeds"]
                ),
                "checkpoint_count": 1,
                "used_for_checkpoint_selection": False,
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "final_test_output": str(final_dir),
                "checkpoint": str(selected_path),
                "used_for_checkpoint_selection": False,
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    specification = yaml.safe_load(Path(args.config).read_text())
    output_path = Path(args.output_dir or specification["output_dir"])
    if args.run_final_test:
        _run_final_test_only(args, specification, output_path)
        return
    run_dir = Path(specification["checkpoint_run_dir"])
    inventory = discover_candidates(
        run_dir, top_k=int(specification["candidate_selection"]["light_top_k"])
    )
    if args.max_candidates is not None:
        inventory["included"] = inventory["included"][: args.max_candidates]
    selection_count = int(
        args.max_utterances
        or specification["full_selection_validation"]["utterances"]
    )
    realization_count = int(
        args.max_realizations
        or specification["full_selection_validation"]["realizations"]
    )
    dry = {
        "output_dir": str(output_path),
        "candidate_count": len(inventory["included"]),
        "selection_source_manifest": specification["data"][
            "selection_source_manifest"
        ],
        "selection_assigned_role": "selection_validation",
        "selection_utterances": selection_count,
        "selection_realizations": realization_count,
        "selection_snr_db": specification["full_selection_validation"]["snr_db"],
        "legacy_final_utterances": specification["legacy_final"]["utterances"],
        "legacy_final_frozen": True,
        "run_final_test": args.run_final_test,
    }
    if args.dry_run:
        print(json.dumps(dry, indent=2))
        return
    if (
        selection_count > 2
        or realization_count > 1
        or len(inventory["included"]) > 2
    ) and not args.allow_long_run:
        raise SystemExit(
            "expanded validation above 2 candidates x 2 utterances x 1 realization "
            "requires --allow-long-run"
        )
    started = time.time()
    output = prepare_output_directory(output_path, overwrite=args.overwrite)
    initial_path = Path(specification["initial_checkpoint"])
    initial_payload = _checkpoint_payload(initial_path)
    model_config = copy.deepcopy(initial_payload["config"])
    validate_initial_checkpoint_metadata(model_config)
    model_config["device"] = args.device or specification["device"]
    device = resolve_device(model_config["device"])

    train_paths, legacy_final_paths = fixed_paths(
        model_config, int(model_config["seed"])
    )
    crop_samples = int(model_config["codec"]["waveform_samples"])
    train_entries = entries_from_paths(
        train_paths,
        source_split="train-clean-5",
        role="train",
        crop_num_samples=crop_samples,
    )
    selection_entries = build_selection_manifest(
        specification["data"]["selection_source_manifest"],
        count=selection_count,
        seed=int(specification["data"]["selection_seed"]),
        crop_num_samples=crop_samples,
    )
    legacy_final_entries = entries_from_paths(
        legacy_final_paths[: int(specification["legacy_final"]["utterances"])],
        source_split="dev-clean-2",
        role="legacy_final_test",
        crop_num_samples=crop_samples,
    )
    audit = audit_protocol_overlap(
        train_entries, selection_entries, legacy_final_entries
    )
    audit["manifest_metadata"] = {
        "train": _manifest_metadata(train_entries),
        "selection_validation": _manifest_metadata(selection_entries),
        "legacy_final_test": _manifest_metadata(legacy_final_entries),
    }
    audit["historical_protocol"] = {
        "light_utterances": 6,
        "light_realizations": 1,
        "light_cadence_steps": 250,
        "full_utterances": 16,
        "full_realizations": 3,
        "full_cadence_steps": 1000,
        "selection_final_path_overlap": 16,
        "selection_final_speaker_overlap": 16,
        "legacy_final_used_during_historical_selection": True,
    }
    _write_jsonl(output / "selection_validation_manifest.jsonl", [
        entry.to_dict() for entry in selection_entries
    ])
    _write_jsonl(output / "final_test_manifest_reference.jsonl", [
        entry.to_dict() for entry in legacy_final_entries
    ])
    (output / "validation_overlap_audit.json").write_text(
        json.dumps(audit, indent=2)
    )
    realization_seeds = list(
        specification["full_selection_validation"]["realization_seeds"]
    )[:realization_count]
    seed_manifest = build_seed_manifest(
        utterance_ids=[entry.utterance_id for entry in selection_entries],
        snr_db=specification["full_selection_validation"]["snr_db"],
        realization_seeds=realization_seeds,
    )
    (output / "seed_manifest.json").write_text(json.dumps(seed_manifest, indent=2))
    (output / "candidate_checkpoints.json").write_text(
        json.dumps(inventory, indent=2)
    )
    resolved = copy.deepcopy(specification)
    resolved["resolved_device"] = str(device)
    resolved["resolved_selection_utterances"] = selection_count
    resolved["resolved_selection_realizations"] = realization_count
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False)
    )
    (output / "environment.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "git": _git_metadata(),
                "initial_checkpoint_sha256": file_sha256(initial_path),
            },
            indent=2,
        )
    )
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n")

    codec, initial_model = build_components(model_config, device)
    freeze_codec_for_input_gradient(codec)
    initial_model.load_state_dict(initial_payload["model"], strict=True)
    initial_model.eval().requires_grad_(False)
    initial_model._expanded_validation_config = model_config
    with torch.no_grad():
        all_rows, summaries = evaluate_candidates(
            candidate_inventory=inventory["included"],
            entries=selection_entries,
            specification=specification,
            initial_payload=initial_payload,
            codec=codec,
            initial_model=initial_model,
            device=device,
            realization_seeds=realization_seeds,
        )
    ranked = rank_candidates(summaries)
    decision = {
        "checkpoint_path": ranked[0]["checkpoint_path"],
        "global_step": ranked[0]["global_step"],
        "rank": 1,
        "clean_gate_pass": bool(ranked[0]["clean_gate_pass"]),
        "selection_status": (
            "passing_candidate"
            if ranked[0]["clean_gate_pass"]
            else "best_nonpassing_candidate"
        ),
        "selection_basis": "full_selection_validation_utterance_level_paired",
        "final_test_used_for_selection": False,
    }
    write_selected_checkpoint(
        decision["checkpoint_path"],
        output / "best_expanded_validation.pt",
        decision,
    )
    _write_jsonl(output / "per_sample_metrics.jsonl", all_rows)
    (output / "per_checkpoint_summary.json").write_text(
        json.dumps(summaries, indent=2)
    )
    summary_csv = _summary_csv_rows(summaries)
    _write_csv(output / "per_checkpoint_summary.csv", summary_csv)
    (output / "paired_statistics.json").write_text(
        json.dumps(
            {
                row["checkpoint_path"]: row["by_snr"] for row in summaries
            },
            indent=2,
        )
    )
    ranking_csv = _summary_csv_rows(ranked)
    for index, row in enumerate(ranking_csv):
        row["rank"] = index + 1
    _write_csv(output / "checkpoint_ranking.csv", ranking_csv)
    (output / "selection_decision.json").write_text(
        json.dumps(decision, indent=2)
    )

    final_executed = False

    runtime = {
        "elapsed_seconds": time.time() - started,
        "candidate_count": len(summaries),
        "selection_utterances": selection_count,
        "selection_realizations": realization_count,
        "selection_conditions": len(all_rows),
        "final_test_executed": final_executed,
        "long_training_executed": False,
    }
    (output / "runtime_summary.json").write_text(json.dumps(runtime, indent=2))
    _write_report(output, audit, inventory["included"], ranked, decision, final_executed)
    print(json.dumps({"selection_decision": decision, "runtime": runtime}, indent=2))


if __name__ == "__main__":
    main()
