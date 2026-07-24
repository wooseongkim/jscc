from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from eval_codec_only import (
    build_codec_only,
    read_manifest,
    stft_l1,
    waveform_metrics,
)
from speech_jscc.config import load_config, resolve_device
from speech_jscc.data import load_waveform_segment
from speech_jscc.layer_importance import file_sha256
from speech_jscc.metrics import compute_stoi


METRIC_DIRECTIONS = {
    "waveform_snr": "higher",
    "si_sdr": "higher",
    "stft_l1": "lower",
    "stoi": "higher",
}
CSV_METRIC_KEYS = {
    "waveform_snr": "waveform_snr_db",
    "si_sdr": "si_sdr_db",
    "stft_l1": "stft_l1",
    "stoi": "stoi",
}
DEFAULT_IMPORTANCE = {
    "estimator": "combined",
    "source_protocols": {"leave_one_out": 0.7, "prefix_marginal": 0.3},
    "metric_weights": {
        "si_sdr": 1.0,
        "waveform_snr": 0.5,
        "stft_l1": 0.5,
        "stoi": 1.0,
        "speaker_similarity": 0.5,
        "wer": 1.0,
    },
    "normalization": "max",
    "output_weight_normalization": "mean_one",
    "minimum_weight": 0.05,
    "base_layer_cumulative_threshold": 0.70,
    "enforce_prefix_base_layers": True,
}


def _as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return float(math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)))


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "mean": _mean(values),
        "median": _percentile(values, 0.5),
        "std": _std(values),
        "p10": _percentile(values, 0.1),
        "p90": _percentile(values, 0.9),
        "min": float(min(values)) if values else None,
        "max": float(max(values)) if values else None,
        "n": len(values),
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _config_hash(config: dict[str, Any]) -> str:
    encoded = yaml.safe_dump(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def apply_layer_mask(representation: Tensor, mask: Tensor | list[float]) -> Tensor:
    mask_tensor = torch.as_tensor(mask, device=representation.device, dtype=representation.dtype)
    if representation.ndim != 4 or mask_tensor.shape != (representation.shape[1],):
        raise ValueError("representation must be [B,L,T,D] and mask must be [L]")
    return representation.clone() * mask_tensor[None, :, None, None]


def _mask_for(protocol: str, layer_or_k: int, layers: int, device: torch.device) -> Tensor:
    mask = torch.ones(layers, device=device)
    if protocol == "all_layers":
        return mask
    if protocol == "leave_one_out":
        mask[layer_or_k] = 0.0
        return mask
    if protocol == "prefix_keep":
        mask[layer_or_k:] = 0.0
        return mask
    if protocol == "single_layer_only":
        mask.zero_()
        mask[layer_or_k] = 1.0
        return mask
    raise ValueError(f"unsupported ablation protocol: {protocol}")


def _record_metadata(record: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        "utt_id": record.get("utt_id", Path(str(record["audio_path"])).stem),
        "speaker_id": record.get("speaker_id", ""),
        "chapter_id": record.get("chapter_id", ""),
        "split": record.get("split", split),
        "audio_path": record["audio_path"],
        "text": record.get("text", ""),
        "duration_sec": record.get("duration_sec"),
        "sample_rate": record.get("sample_rate"),
        "num_samples": record.get("num_samples"),
    }


def _compute_metrics(
    reference: Tensor,
    estimate: Tensor,
    *,
    sample_rate: int,
    enable_stoi: bool,
    metric_align: str,
    max_lag_samples: int,
    snr_scale_match: bool,
    metric_zero_mean: bool,
) -> dict[str, Any]:
    values = waveform_metrics(
        reference,
        estimate,
        metric_align=metric_align,
        max_lag_samples=max_lag_samples,
        snr_scale_match=snr_scale_match,
        metric_zero_mean=metric_zero_mean,
    )
    values["stoi"] = ""
    values["stoi_available"] = False
    values["stoi_error"] = None
    if enable_stoi:
        try:
            stoi_values = compute_stoi(reference, estimate, sample_rate)
        except Exception as error:
            values["stoi_error"] = str(error)
        else:
            if stoi_values is None:
                values["stoi_error"] = "pystoi is not installed"
            else:
                values["stoi"] = float(stoi_values.mean().detach().cpu().item())
                values["stoi_available"] = True
    return values


def _metric_delta(metric: str, baseline: float, ablated: float) -> float:
    if METRIC_DIRECTIONS[metric] == "higher":
        return baseline - ablated
    return ablated - baseline


def _prefix_marginal(metric: str, previous: float, current: float) -> float:
    if METRIC_DIRECTIONS[metric] == "higher":
        return current - previous
    return previous - current


def _summarize_rows(rows: list[dict[str, Any]], protocol: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["protocol"] != protocol:
            continue
        layer_index = int(row.get("layer_index", -1))
        prefix_k = int(row.get("prefix_k", -1))
        for metric, csv_key in CSV_METRIC_KEYS.items():
            value = row.get(csv_key, "")
            if value == "" or value is None:
                continue
            grouped[(metric, layer_index if protocol != "prefix_keep" else prefix_k, csv_key)].append(
                float(value)
            )
    summary = []
    for (metric, index, _), values in sorted(grouped.items()):
        stats = _distribution(values)
        summary.append(
            {
                "protocol": protocol,
                "metric": metric,
                "layer_index": index if protocol != "prefix_keep" else "",
                "prefix_k": index if protocol == "prefix_keep" else "",
                **stats,
            }
        )
    return summary


def _build_importance(
    *,
    layers: int,
    summary_rows: list[dict[str, Any]],
    prefix_rows: list[dict[str, Any]],
    config: dict[str, Any],
    available_metrics: list[str],
) -> tuple[dict[str, Any], dict[str, dict[str, list[float]]]]:
    options = {**DEFAULT_IMPORTANCE, **(config.get("layer_importance") or {})}
    metric_weights = {
        name: float(weight)
        for name, weight in options.get("metric_weights", {}).items()
        if name in available_metrics and float(weight) > 0.0
    }
    if not metric_weights:
        metric_weights = {"si_sdr": 1.0} if "si_sdr" in available_metrics else {"waveform_snr": 1.0}
    metric_weight_total = sum(metric_weights.values())
    metric_weights = {name: weight / metric_weight_total for name, weight in metric_weights.items()}

    baseline = {
        row["metric"]: float(row["mean"])
        for row in summary_rows
        if row["protocol"] == "all_layers" and row["mean"] not in {"", None}
    }
    leave_values: dict[str, list[float]] = {metric: [0.0] * layers for metric in metric_weights}
    loo_means = {
        (row["metric"], int(row["layer_index"])): float(row["mean"])
        for row in summary_rows
        if row["protocol"] == "leave_one_out" and row["mean"] not in {"", None}
    }
    for metric in metric_weights:
        for layer in range(layers):
            if metric in baseline and (metric, layer) in loo_means:
                leave_values[metric][layer] = _metric_delta(metric, baseline[metric], loo_means[(metric, layer)])

    prefix_means = {
        (row["metric"], int(row["prefix_k"])): float(row["mean"])
        for row in prefix_rows
        if row["mean"] not in {"", None}
    }
    prefix_values: dict[str, list[float]] = {metric: [0.0] * layers for metric in metric_weights}
    for metric in metric_weights:
        previous = None
        for k in range(1, layers + 1):
            current = prefix_means.get((metric, k))
            if current is None:
                continue
            if previous is None:
                if k == 1:
                    zero_value = 0.0 if METRIC_DIRECTIONS[metric] == "higher" else current
                    prefix_values[metric][0] = _prefix_marginal(metric, zero_value, current)
                previous = current
            else:
                prefix_values[metric][k - 1] = _prefix_marginal(metric, previous, current)
                previous = current

    def normalize(values: list[float]) -> list[float]:
        cleaned = [0.0 if -1e-6 < value < 0.0 else max(0.0, value) for value in values]
        if options.get("normalization", "max") == "sum":
            total = sum(cleaned)
            return [value / total for value in cleaned] if total > 0.0 else [0.0] * len(cleaned)
        maximum = max(cleaned) if cleaned else 0.0
        return [value / maximum for value in cleaned] if maximum > 0.0 else [0.0] * len(cleaned)

    leave_score = [0.0] * layers
    prefix_score = [0.0] * layers
    for metric, weight in metric_weights.items():
        leave_norm = normalize(leave_values[metric])
        prefix_norm = normalize(prefix_values[metric])
        leave_score = [score + weight * value for score, value in zip(leave_score, leave_norm)]
        prefix_score = [score + weight * value for score, value in zip(prefix_score, prefix_norm)]
    protocol_weights = {
        key: float(value)
        for key, value in options.get("source_protocols", {}).items()
        if key in {"leave_one_out", "prefix_marginal"} and float(value) > 0.0
    }
    if not protocol_weights:
        protocol_weights = {"leave_one_out": 1.0}
    protocol_total = sum(protocol_weights.values())
    protocol_weights = {key: value / protocol_total for key, value in protocol_weights.items()}
    raw_scores = [
        protocol_weights.get("leave_one_out", 0.0) * leave_score[layer]
        + protocol_weights.get("prefix_marginal", 0.0) * prefix_score[layer]
        for layer in range(layers)
    ]
    minimum = float(options.get("minimum_weight", 0.0))
    if max(raw_scores) <= 0.0:
        raw_scores = [1.0] * layers
    raw_scores = [max(minimum, value) for value in raw_scores]
    mean_score = sum(raw_scores) / layers
    layer_weights_mean_one = [value / mean_score for value in raw_scores]
    total_score = sum(raw_scores)
    layer_weights_sum_one = [value / total_score for value in raw_scores]
    order = sorted(range(layers), key=lambda layer: raw_scores[layer], reverse=True)
    threshold = float(options.get("base_layer_cumulative_threshold", 0.70))
    cumulative = 0.0
    base_layers: list[int] = []
    if bool(options.get("enforce_prefix_base_layers", True)):
        for layer, weight in enumerate(layer_weights_sum_one):
            cumulative += weight
            base_layers.append(layer)
            if cumulative >= threshold:
                break
    else:
        for layer in order:
            cumulative += layer_weights_sum_one[layer]
            base_layers.append(layer)
            if cumulative >= threshold:
                break
        base_layers = sorted(base_layers)
    return (
        {
            "estimator": options.get("estimator", "combined"),
            "metric_weights": metric_weights,
            "protocol_weights": protocol_weights,
            "raw_scores": raw_scores,
            "leave_one_out_scores": leave_score,
            "prefix_marginal_scores": prefix_score,
            "layer_weights_mean_one": layer_weights_mean_one,
            "layer_weights_sum_one": layer_weights_sum_one,
            "layer_importance_order": order,
            "base_layers": base_layers,
            "base_layer_cumulative_threshold": threshold,
        },
        {
            "leave_one_out": leave_values,
            "prefix_marginal": prefix_values,
        },
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(output_dir: Path, importance: dict[str, Any], summary_rows: list[dict[str, Any]], prefix_rows: list[dict[str, Any]]) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        weights = importance["layer_weights_mean_one"]
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.bar(range(len(weights)), weights)
        axis.set(xlabel="Layer", ylabel="Weight (mean=1)", title="Recommended layer importance")
        fig.tight_layout()
        fig.savefig(output_dir / "layer_importance_bar.png", dpi=150)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(7, 4))
        for metric in ("si_sdr", "waveform_snr", "stft_l1", "stoi"):
            values = [
                row
                for row in summary_rows
                if row["protocol"] == "leave_one_out" and row["metric"] == metric
            ]
            if not values:
                continue
            baseline_rows = [
                row for row in summary_rows if row["protocol"] == "all_layers" and row["metric"] == metric
            ]
            if not baseline_rows:
                continue
            baseline = float(baseline_rows[0]["mean"])
            deltas = [
                _metric_delta(metric, baseline, float(row["mean"]))
                for row in sorted(values, key=lambda item: int(item["layer_index"]))
            ]
            axis.plot(range(len(deltas)), deltas, marker="o", label=metric)
        axis.set(xlabel="Removed layer", ylabel="Degradation", title="Metric deltas by layer")
        axis.legend()
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "metric_delta_by_layer.png", dpi=150)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(7, 4))
        for metric in ("si_sdr", "waveform_snr", "stft_l1"):
            values = [row for row in prefix_rows if row["metric"] == metric]
            if not values:
                continue
            values = sorted(values, key=lambda item: int(item["prefix_k"]))
            axis.plot(
                [int(row["prefix_k"]) for row in values],
                [float(row["mean"]) for row in values],
                marker="o",
                label=metric,
            )
        axis.set(xlabel="Prefix layers kept", ylabel="Metric", title="Prefix reconstruction curves")
        axis.legend()
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "prefix_metric_curves.png", dpi=150)
        plt.close(fig)
    except Exception as error:
        print(f"warning: layer ablation plot generation failed: {error}")


def _write_report(path: Path, artifact: dict[str, Any], summary_rows: list[dict[str, Any]], prefix_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# SpeechTokenizer RVQ Layer Ablation Report",
        "",
        f"- codec type: `{artifact['codec']['type']}`",
        f"- checkpoint: `{artifact['codec'].get('checkpoint_path')}`",
        f"- representation shape: `{artifact['codec']['representation_shape']}`",
        f"- manifest: `{artifact['dataset']['manifest']}`",
        f"- split: `{artifact['dataset']['split']}`",
        f"- evaluated samples: `{artifact['dataset']['evaluated_items']}`",
        f"- available metrics: `{artifact['metrics']['available']}`",
        f"- unavailable metrics: `{artifact['metrics']['unavailable']}`",
        "",
        "## All-layer Baseline",
        "",
        "| metric | mean | median | std | p10 | p90 | min | max | n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        if row["protocol"] == "all_layers":
            lines.append(
                f"| {row['metric']} | {row['mean']} | {row['median']} | {row['std']} | "
                f"{row['p10']} | {row['p90']} | {row['min']} | {row['max']} | {row['n']} |"
            )
    lines += [
        "",
        "## Leave-one-out Degradation",
        "",
        "| layer | metric | leave-one-out mean | degradation |",
        "|---:|---|---:|---:|",
    ]
    baseline = {
        row["metric"]: float(row["mean"])
        for row in summary_rows
        if row["protocol"] == "all_layers" and row["mean"] is not None
    }
    for row in summary_rows:
        if row["protocol"] != "leave_one_out":
            continue
        metric = row["metric"]
        degradation = _metric_delta(metric, baseline[metric], float(row["mean"]))
        lines.append(f"| {row['layer_index']} | {metric} | {row['mean']} | {degradation} |")
    lines += [
        "",
        "## Prefix Reconstruction",
        "",
        "| prefix_k | metric | mean | median |",
        "|---:|---|---:|---:|",
    ]
    for row in prefix_rows:
        lines.append(f"| {row['prefix_k']} | {row['metric']} | {row['mean']} | {row['median']} |")
    lines += [
        "",
        "## Recommended Fixed Layer Importance",
        "",
        f"- raw scores: `{artifact['importance']['raw_scores']}`",
        f"- layer_weights_mean_one: `{artifact['importance']['layer_weights_mean_one']}`",
        f"- layer_weights_sum_one: `{artifact['importance']['layer_weights_sum_one']}`",
        f"- layer_importance_order: `{artifact['importance']['layer_importance_order']}`",
        f"- base_layers: `{artifact['importance']['base_layers']}`",
        "",
        "## Interpretation",
        "",
        "The weights `w_l` in this artifact are fixed corpus-level priors from intrinsic codec-only ablation.",
        "They are not the learned gate values `alpha_l(c)`. The learned gate is channel-conditioned and must not be trained or invoked while deriving intrinsic layer importance.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_recommended_snippet(path: Path, artifact_path: Path) -> None:
    snippet = {
        "layer_importance": {
            "path": str(artifact_path),
            "strict_metadata": True,
            "apply_to_loss_weights": True,
            "apply_to_resource_order": True,
            "apply_to_base_layers": True,
        }
    }
    path.write_text(yaml.safe_dump(snippet, sort_keys=False), encoding="utf-8")


def run_layer_ablation(
    *,
    config_path: str | Path,
    manifest_path: str | Path,
    split: str,
    output_dir: str | Path,
    max_items: int | None = None,
    batch_size: int = 4,
    protocols: list[str] | None = None,
    checkpoint: str | Path | None = None,
    mode: str = "codec_only",
    device_name: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    if mode != "codec_only":
        raise NotImplementedError("full_jscc layer ablation is intentionally separate and not implemented")
    protocols = protocols or ["leave_one_out", "prefix_keep"]
    allowed = {"leave_one_out", "prefix_keep", "single_layer_only"}
    unknown = set(protocols) - allowed
    if unknown:
        raise ValueError(f"unsupported protocols: {sorted(unknown)}")
    config = load_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    if checkpoint is not None:
        config.setdefault("codec", {})["checkpoint_path"] = str(checkpoint)
    if seed is None:
        seed = int(config.get("seed", 0))
    torch.manual_seed(int(seed))
    device = resolve_device(device_name or config.get("device", "auto"))
    codec = build_codec_only(config, device)
    codec.eval()
    if any(parameter.requires_grad for parameter in codec.parameters()):
        raise RuntimeError("codec must be frozen for intrinsic layer ablation")
    sample_rate = int(getattr(codec, "sample_rate", config.get("codec", {}).get("sample_rate", 16000)))
    waveform_samples = int(getattr(codec, "waveform_samples", config["codec"]["waveform_samples"]))
    records = read_manifest(manifest_path, max_items=max_items)
    layers, frames, latent_dim = codec.representation_shape
    eval_cfg = config.get("eval", {})
    enable_stoi = bool(eval_cfg.get("enable_stoi", False))
    metric_align = eval_cfg.get("metric_align", "peak_xcorr")
    snr_scale_match = bool(eval_cfg.get("snr_scale_match", True))
    metric_zero_mean = bool(eval_cfg.get("metric_zero_mean", True))
    max_lag_samples = int(eval_cfg.get("max_lag_samples", 1000))
    per_sample_rows: list[dict[str, Any]] = []
    metric_errors: dict[str, str] = {}

    with torch.no_grad():
        for start in range(0, len(records), max(1, int(batch_size))):
            batch_records = records[start : start + max(1, int(batch_size))]
            waveforms = [
                load_waveform_segment(record["audio_path"], sample_rate, waveform_samples)
                for record in batch_records
            ]
            waveform = torch.stack(waveforms).to(device)
            representation = codec.encode_waveform(waveform)
            if not torch.isfinite(representation).all():
                raise RuntimeError("codec representation contains NaN or Inf")
            per_layer_energy = representation.square().mean(dim=(2, 3)).detach().cpu()
            run_specs: list[tuple[str, int, int, Tensor]] = [
                ("all_layers", -1, -1, _mask_for("all_layers", 0, layers, device))
            ]
            if "leave_one_out" in protocols:
                run_specs.extend(
                    ("leave_one_out", layer, -1, _mask_for("leave_one_out", layer, layers, device))
                    for layer in range(layers)
                )
            if "prefix_keep" in protocols:
                run_specs.extend(
                    ("prefix_keep", -1, k, _mask_for("prefix_keep", k, layers, device))
                    for k in range(1, layers + 1)
                )
            if "single_layer_only" in protocols:
                run_specs.extend(
                    ("single_layer_only", layer, -1, _mask_for("single_layer_only", layer, layers, device))
                    for layer in range(layers)
                )
            for protocol, layer_index, prefix_k, mask in run_specs:
                masked = apply_layer_mask(representation, mask)
                estimate = codec.decode_representation(masked)
                for batch_index, record in enumerate(batch_records):
                    metrics = _compute_metrics(
                        waveform[batch_index : batch_index + 1],
                        estimate[batch_index : batch_index + 1],
                        sample_rate=sample_rate,
                        enable_stoi=enable_stoi,
                        metric_align=metric_align,
                        max_lag_samples=max_lag_samples,
                        snr_scale_match=snr_scale_match,
                        metric_zero_mean=metric_zero_mean,
                    )
                    if metrics.get("stoi_error"):
                        metric_errors["stoi"] = str(metrics["stoi_error"])
                    metadata = _record_metadata(record, split)
                    per_sample_rows.append(
                        {
                            **metadata,
                            "protocol": protocol,
                            "layer_index": layer_index,
                            "prefix_k": prefix_k,
                            "mask": json.dumps([float(value) for value in mask.detach().cpu().tolist()]),
                            "latent_energy_per_layer": json.dumps(
                                [float(value) for value in per_layer_energy[batch_index].tolist()]
                            ),
                            **metrics,
                        }
                    )

    summary_rows = _summarize_rows(per_sample_rows, "all_layers")
    summary_rows.extend(_summarize_rows(per_sample_rows, "leave_one_out"))
    if "single_layer_only" in protocols:
        summary_rows.extend(_summarize_rows(per_sample_rows, "single_layer_only"))
    prefix_rows = _summarize_rows(per_sample_rows, "prefix_keep")
    available_metrics = [
        metric
        for metric in ("waveform_snr", "si_sdr", "stft_l1", "stoi")
        if any(row["metric"] == metric for row in summary_rows)
        and not (metric == "stoi" and all(row.get("stoi") in {"", None} for row in per_sample_rows))
    ]
    unavailable_metrics = [
        metric
        for metric in ("stoi", "pesq", "visqol", "speaker_similarity", "wer")
        if metric not in available_metrics
    ]
    importance, deltas = _build_importance(
        layers=layers,
        summary_rows=summary_rows,
        prefix_rows=prefix_rows,
        config=config,
        available_metrics=available_metrics,
    )
    baseline = {
        row["metric"]: row["mean"]
        for row in summary_rows
        if row["protocol"] == "all_layers"
    }
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "codec": {
            "type": config["codec"].get("type", "mock").lower(),
            "representation_shape": [layers, frames, latent_dim],
            "sample_rate": sample_rate,
            "waveform_samples": waveform_samples,
            "n_q": config["codec"].get("n_q", layers),
            "config_path": str(config_path),
            "checkpoint_path": str(config["codec"].get("checkpoint_path", "")),
        },
        "dataset": {
            "manifest": str(manifest_path),
            "split": split,
            "evaluated_items": len(records),
            "seed": int(seed),
        },
        "ablation": {
            "protocols": ["all_layers", *protocols],
            "masking_value": 0.0,
            "mode": mode,
        },
        "metrics": {
            "available": available_metrics,
            "unavailable": unavailable_metrics,
            "errors": metric_errors,
            "baseline": baseline,
            "leave_one_out_deltas": deltas["leave_one_out"],
            "prefix_marginal_deltas": deltas["prefix_marginal"],
        },
        "importance": importance,
        "provenance": {
            "generated_by": "eval_layer_ablation.py",
            "git_commit": _git_commit(),
            "config_hash": _config_hash(config),
            "manifest_hash": file_sha256(manifest_path),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    _write_csv(output / "ablation_per_sample.csv", per_sample_rows)
    _write_csv(output / "ablation_summary.csv", summary_rows)
    _write_csv(output / "prefix_summary.csv", prefix_rows)
    json_path = output / "layer_importance.json"
    yaml_path = output / "layer_importance.yaml"
    json_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    yaml_path.write_text(yaml.safe_dump(artifact, sort_keys=False), encoding="utf-8")
    _write_recommended_snippet(output / "recommended_config_snippet.yaml", yaml_path)
    _write_report(output / "layer_ablation_report.md", artifact, summary_rows, prefix_rows)
    _write_plots(output, importance, summary_rows, prefix_rows)
    print(f"wrote_layer_importance={yaml_path}")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate intrinsic SpeechTokenizer RVQ layer importance")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_items", type=int)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--protocols", nargs="+", default=["leave_one_out", "prefix_keep"])
    parser.add_argument("--checkpoint")
    parser.add_argument("--mode", choices=["codec_only", "full_jscc"], default="codec_only")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    run_layer_ablation(
        config_path=args.config,
        manifest_path=args.manifest,
        split=args.split,
        output_dir=args.output_dir,
        max_items=args.max_items,
        batch_size=args.batch_size,
        protocols=args.protocols,
        checkpoint=args.checkpoint,
        mode=args.mode,
        device_name=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
