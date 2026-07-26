from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import yaml

from speech_jscc.config import resolve_device
from speech_jscc.experiment import build_components
from speech_jscc.evaluation.si_sdr_alignment import (
    active_speech_fraction,
    best_cross_correlation_alignment,
)
from speech_jscc.training.r4_waveform_finetune import (
    R4ForwardCondition,
    R4WaveformForward,
    freeze_codec_for_input_gradient,
)
from src.evaluation.waveform_metrics import waveform_metrics
from train_channel_free_conv_conformer import fixed_paths, load_batch
from evaluate_r4_expanded_validation import (
    _cache_codec_inputs,
    _checkpoint_payload,
    _conditions,
    _initial_trajectory,
    _make_engine,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval_r4_expanded_validation.yaml")
    p.add_argument("--output-dir", default="runs/waveform_aware_wireless/r4_expanded_validation/si_sdr_alignment_diagnostic")
    p.add_argument("--device")
    p.add_argument("--max-lag-ms", type=float, default=5.0)
    p.add_argument("--max-utterances", type=int)
    p.add_argument("--max-realizations", type=int)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _metric_row(ref, estimate, sample_rate, label, snr, realization, entry, max_lag_ms):
    standard = waveform_metrics(ref, estimate, sample_rate)
    aligned = best_cross_correlation_alignment(
        ref, estimate, sample_rate=sample_rate, max_lag_ms=max_lag_ms
    )
    return {
        "model": label,
        "utterance_id": entry.utterance_id,
        "speaker_id": entry.speaker_id,
        "source_path": entry.source_path,
        "snr_db": float(snr),
        "realization": int(realization),
        "reference_samples": int(ref.shape[-1]),
        "estimate_samples": int(estimate.shape[-1]),
        "reference_mean": float(ref.mean()),
        "estimate_mean": float(estimate.mean()),
        "reference_rms": float(ref.square().mean().sqrt()),
        "estimate_rms": float(estimate.square().mean().sqrt()),
        "active_speech_fraction": active_speech_fraction(ref),
        "si_sdr_standard_db": float(standard["si_sdr_db"]),
        "waveform_snr_standard_db": float(standard["waveform_snr_db"]),
        "stft_l1_standard": float(standard["stft_l1"]),
        "aligned_si_sdr_db": float(aligned.si_sdr_db),
        "alignment_shift_samples": int(aligned.shift_samples),
        "alignment_shift_ms": float(aligned.shift_samples * 1000.0 / sample_rate),
        "alignment_overlap_samples": int(aligned.overlap_samples),
        "alignment_correlation": float(aligned.correlation),
        "si_sdr_alignment_gain_db": float(aligned.si_sdr_db - standard["si_sdr_db"]),
    }


def _aggregate(rows):
    out = {}
    for model in sorted({r["model"] for r in rows}):
        out[model] = {}
        for snr in sorted({float(r["snr_db"]) for r in rows}):
            values = [r for r in rows if r["model"] == model and float(r["snr_db"]) == snr]
            def stats(key):
                x = torch.tensor([float(r[key]) for r in values], dtype=torch.float64)
                return {"count": int(x.numel()), "mean": float(x.mean()), "median": float(x.median()), "std": float(x.std(unbiased=x.numel()>1)), "p10": float(torch.quantile(x, 0.10)), "p90": float(torch.quantile(x, 0.90)), "min": float(x.min()), "max": float(x.max())}
            out[model][str(snr)] = {
                "si_sdr_standard_db": stats("si_sdr_standard_db"),
                "aligned_si_sdr_db": stats("aligned_si_sdr_db"),
                "si_sdr_alignment_gain_db": stats("si_sdr_alignment_gain_db"),
                "alignment_shift_ms": stats("alignment_shift_ms"),
                "alignment_correlation": stats("alignment_correlation"),
                "waveform_snr_standard_db": stats("waveform_snr_standard_db"),
                "stft_l1_standard": stats("stft_l1_standard"),
                "length_mismatch_count": sum(r["reference_samples"] != r["estimate_samples"] for r in values),
            }
    return out


def main():
    args = parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise SystemExit(f"refusing existing output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    specification = yaml.safe_load(Path(args.config).read_text())
    selected_path = output.parent / "best_expanded_validation.pt"
    if not selected_path.exists():
        selected_path = Path("runs/waveform_aware_wireless/r4_expanded_validation/best_expanded_validation.pt")
    initial_path = Path(specification["initial_checkpoint"])
    final_manifest = Path(specification["output_dir"]) / "final_test_manifest_reference.jsonl"
    entries = [json.loads(line) for line in final_manifest.read_text().splitlines() if line.strip()]
    entries = entries[: args.max_utterances] if args.max_utterances else entries
    from speech_jscc.evaluation.expanded_validation import ManifestEntry
    entries = [ManifestEntry(**e) for e in entries]
    realization_seeds = list(specification["legacy_final"]["realization_seeds"])
    if args.max_realizations:
        realization_seeds = realization_seeds[: args.max_realizations]
    selected_payload = _checkpoint_payload(selected_path)
    model_config = copy.deepcopy(selected_payload["config"])
    model_config["device"] = args.device or specification["device"]
    device = resolve_device(model_config["device"])
    codec, template = build_components(model_config, device)
    freeze_codec_for_input_gradient(codec)
    initial_payload = _checkpoint_payload(initial_path)
    template.load_state_dict(initial_payload["model"], strict=True)
    template.eval().requires_grad_(False)
    template._expanded_validation_config = model_config
    selected = copy.deepcopy(template)
    selected.load_state_dict(selected_payload["model"], strict=True)
    selected.eval().requires_grad_(False)
    selected._expanded_validation_config = model_config
    cache = _cache_codec_inputs(codec, entries, model_config, device)
    rows = []
    sample_rate = int(model_config["codec"]["sample_rate"])
    started = time.time()
    with torch.no_grad():
        for snr in map(float, specification["full_selection_validation"]["snr_db"]):
            for realization, seed in enumerate(realization_seeds):
                conditions = _conditions(specification, _make_engine(codec, template, specification), snr=snr, seed=seed, count=len(entries))
                initial_engine = _make_engine(codec, template, specification)
                selected_engine = _make_engine(codec, selected, specification)
                report = None
                reports = []
                for cached, condition in zip(cache, conditions):
                    reports.append(report)
                    clean_waveform = codec.decode_representation(cached["target"].to(device))
                    rows.append(_metric_row(cached["waveform"].to(device), clean_waveform, sample_rate, "clean_codec", snr, realization, cached["entry"], args.max_lag_ms))
                    result = initial_engine.forward(cached["target"].to(device), cached["waveform"].to(device), condition, report, training=False)
                    rows.append(_metric_row(cached["waveform"].to(device), result.decoded_waveform, sample_rate, "initial_r4", snr, realization, cached["entry"], args.max_lag_ms))
                    report = result.next_delayed_csi
                for cached, condition, shared_report in zip(cache, conditions, reports):
                    result = selected_engine.forward(cached["target"].to(device), cached["waveform"].to(device), condition, shared_report, training=False)
                    rows.append(_metric_row(cached["waveform"].to(device), result.decoded_waveform, sample_rate, "selected_expanded", snr, realization, cached["entry"], args.max_lag_ms))
    (output / "per_sample_metrics.jsonl").write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
    summary = {"reference": "full frozen legacy-final manifest; diagnostic only", "initial_checkpoint": str(initial_path), "selected_checkpoint": str(selected_path), "max_lag_ms": args.max_lag_ms, "waveform_metric_definition": {"standard": "existing full-crop SI-SDR: per-waveform zero mean, eps=1e-8, no delay correction", "aligned": "diagnostic only: positive normalized cross-correlation over +/- max lag, then SI-SDR on overlapping segments"}, "rows": len(rows), "aggregate": _aggregate(rows), "elapsed_seconds": time.time() - started}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# SI-SDR alignment diagnostic", "", "Production SI-SDR was not changed. The standard score is full-crop, zero-mean per waveform, eps=1e-8, with no delay correction. The aligned score is diagnostic only and searches positive normalized cross-correlation over +/- %.2f ms." % args.max_lag_ms, "", "| model | SNR | standard SI-SDR | aligned SI-SDR | gain | mean shift (ms) |", "|---|---:|---:|---:|---:|---:|"]
    for model, by_snr in summary["aggregate"].items():
        for snr, values in by_snr.items():
            lines.append("| %s | %s | %.4f | %.4f | %.4f | %.4f |" % (model, snr, values["si_sdr_standard_db"]["mean"], values["aligned_si_sdr_db"]["mean"], values["si_sdr_alignment_gain_db"]["mean"], values["alignment_shift_ms"]["mean"]))
    lines += ["", "All evaluated waveforms had equal length; no active-speech-only score was used for the production gate. Alignment results are explanatory diagnostics and must not redefine the clean-channel gate."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    (output / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
