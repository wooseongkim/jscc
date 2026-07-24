from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml

from channels.jammer import compute_jsr, make_jammer
from channels.pilot import (
    csi_nmse,
    equalize_with_csi,
    estimate_channel_ls,
    estimate_ofdm_dft_tap_ls,
    insert_pilots,
    make_pilot_mask,
    pilot_evm,
)
from channels.rayleigh import post_channel_jsr, rayleigh_channel
from speech_jscc.config import load_config, resolve_device


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _config_hash(config: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(yaml.safe_dump(config, sort_keys=True).encode("utf-8")).hexdigest()


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None, "p05": None, "p95": None, "n": 0}
    tensor = torch.tensor(finite, dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "median": float(tensor.median().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "p05": float(torch.quantile(tensor, 0.05).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "n": len(finite),
    }


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().real.float().mean().cpu().item())


def _tensor_hash(value: torch.Tensor) -> str:
    cpu = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(cpu.shape)).encode("utf-8"))
    digest.update(str(cpu.dtype).encode("utf-8"))
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()[:16]


def _scenario_id(
    *,
    seed: int,
    snr_db: float,
    jsr_db: float,
    jammer_type: str,
    config: dict[str, Any],
) -> str:
    channel = config["channel"]
    payload = {
        "seed": int(seed),
        "snr_db": float(snr_db),
        "jsr_db": None if jammer_type == "none" else float(jsr_db),
        "jammer_type": jammer_type,
        "fading": channel.get("fading", "multipath_block"),
        "num_taps": int(channel.get("num_taps", 6)),
        "pdp": channel.get("pdp", "exponential"),
        "pdp_decay": float(channel.get("pdp_decay", 0.7)),
        "grid_shape": channel.get("grid_shape", [32, 16]),
        "pilot_spacing": int(channel.get("pilot_spacing", 4)),
        "pilot_time_spacing": channel.get("pilot_time_spacing", 4),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def _masked_mean_power(value: torch.Tensor, data_mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    mask = torch.broadcast_to(data_mask.to(value.device, torch.bool), value.shape)
    count = mask.sum(dim=tuple(range(1, value.ndim))).clamp_min(1)
    power = (value.abs().square() * mask).sum(dim=tuple(range(1, value.ndim)))
    return power / count.to(value.real.dtype).clamp_min(eps)


def _masked_sum_power(value: torch.Tensor, data_mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    mask = torch.broadcast_to(data_mask.to(value.device, torch.bool), value.shape)
    power = (value.abs().square() * mask).sum(dim=tuple(range(1, value.ndim)))
    return power.clamp_min(eps)


def _percentile_power(value: torch.Tensor, data_mask: torch.Tensor, q: float) -> float:
    mask = torch.broadcast_to(data_mask.to(value.device, torch.bool), value.shape)
    selected = value.abs().square()[mask]
    if selected.numel() == 0:
        raise ValueError("data-resource mask contains no data resources")
    return float(torch.quantile(selected.detach().float().cpu(), q).item())


def data_resource_equalization_metrics(
    *,
    channel: dict[str, torch.Tensor],
    transmitted: torch.Tensor,
    pilot_mask: torch.Tensor,
    channel_estimate: torch.Tensor,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Compute post-equalization diagnostics over data resources only."""
    data_mask = ~torch.broadcast_to(pilot_mask.to(transmitted.device, torch.bool), transmitted.shape)
    data_count = int(data_mask.sum().detach().cpu().item())
    pilot_count = int((~data_mask).sum().detach().cpu().item())
    if data_count <= 0:
        raise ValueError("at least one data resource is required")
    gain = channel_estimate.conj() / channel_estimate.abs().square().clamp_min(eps)
    signal_eq = gain * channel["faded_signal"]
    jammer_eq = gain * channel["faded_jammer"]
    noise_eq = gain * channel["noise"]
    received_eq = gain * channel["received"]
    signal_power = _masked_mean_power(signal_eq, data_mask)
    interference_power = _masked_mean_power(jammer_eq, data_mask)
    noise_power = _masked_mean_power(noise_eq, data_mask)
    sinr_linear = signal_power / (interference_power + noise_power).clamp_min(1e-12)
    sinr_db = 10.0 * torch.log10(sinr_linear.clamp_min(1e-12))
    error = received_eq - transmitted
    mse = _masked_mean_power(error, data_mask)
    evm = torch.sqrt(_masked_sum_power(error, data_mask) / _masked_sum_power(transmitted, data_mask))
    gain_abs2 = gain.abs().square()
    return {
        "post_eq_signal_power": _as_float(signal_power),
        "post_eq_interference_power": _as_float(interference_power),
        "post_eq_noise_power": _as_float(noise_power),
        "post_eq_sinr_linear": _as_float(sinr_linear),
        "post_eq_sinr_db": _as_float(sinr_db),
        "equalized_symbol_mse": _as_float(mse),
        "data_evm": _as_float(evm),
        "data_resource_count": data_count,
        "pilot_resource_count": pilot_count,
        "min_h_abs2_data": _percentile_power(channel["signal_fading"], data_mask, 0.0),
        "p01_h_abs2_data": _percentile_power(channel["signal_fading"], data_mask, 0.01),
        "p05_h_abs2_data": _percentile_power(channel["signal_fading"], data_mask, 0.05),
        "max_equalizer_gain": float(gain.abs()[data_mask].detach().float().max().cpu().item()),
        "max_equalizer_gain_abs2": float(gain_abs2[data_mask].detach().float().max().cpu().item()),
        "mean_equalizer_gain_abs2": float(gain_abs2[data_mask].detach().float().mean().cpu().item()),
    }


def _make_realization(
    config: dict[str, Any],
    *,
    seed: int,
    snr_db: float,
    jsr_db: float,
    jammer_type: str,
    device: torch.device,
) -> dict[str, Any]:
    channel = config["channel"]
    generator = torch.Generator(device=device).manual_seed(seed)
    subcarriers, symbols = [int(value) for value in channel.get("grid_shape", [32, 16])]
    target_power = float(channel.get("target_power", 1.0))
    transmitted = torch.full(
        (1, subcarriers, symbols),
        complex(math.sqrt(target_power), 0.0),
        device=device,
        dtype=torch.complex64,
    )
    pilot_mask = make_pilot_mask(
        tuple(transmitted.shape),
        int(channel.get("pilot_spacing", 4)),
        time_spacing=channel.get("pilot_time_spacing", 4),
        device=device,
    )
    transmitted, pilots = insert_pilots(transmitted, pilot_mask)
    if jammer_type == "none":
        jammer = torch.zeros_like(transmitted)
        jammer_mask = torch.zeros_like(pilot_mask)
    else:
        jammer, jammer_mask = make_jammer(
            transmitted,
            jsr_db,
            jammer_type,
            float(channel.get("jammed_fraction", 0.25)),
            pilot_mask=pilot_mask if jammer_type == "pilot" else None,
            pilot_spacing=int(channel.get("pilot_spacing", 4)),
            generator=generator,
        )
    channel_result = rayleigh_channel(
        transmitted,
        jammer,
        snr_db,
        fading="multipath_block",
        num_taps=int(channel.get("num_taps", 6)),
        pdp_decay=float(channel.get("pdp_decay", 0.7)),
        generator=generator,
    )
    scenario = _scenario_id(
        seed=seed,
        snr_db=snr_db,
        jsr_db=jsr_db,
        jammer_type=jammer_type,
        config=config,
    )
    return {
        "scenario_id": scenario,
        "transmitted": transmitted,
        "pilots": pilots,
        "pilot_mask": pilot_mask,
        "jammer": jammer,
        "jammer_mask": jammer_mask,
        "channel": channel_result,
        "signal_fading_hash": _tensor_hash(channel_result["signal_fading"]),
        "jammer_fading_hash": _tensor_hash(channel_result["jammer_fading"]),
        "noise_hash": _tensor_hash(channel_result["noise"]),
        "transmitted_hash": _tensor_hash(transmitted),
        "pilot_mask_hash": _tensor_hash(pilot_mask),
        "jammer_mask_hash": _tensor_hash(jammer_mask),
    }


def _estimate(
    estimator: str,
    realization: dict[str, Any],
    *,
    estimator_num_taps: int,
    ridge_lambda: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    channel = realization["channel"]
    if estimator == "oracle":
        return channel["signal_fading"], {
            "unique_pilot_subcarriers": torch.nonzero(realization["pilot_mask"].any(dim=(0, 2))).flatten(),
            "dft_matrix_condition_number": torch.tensor(float("nan")),
        }
    if estimator == "dft_tap_ls":
        return estimate_ofdm_dft_tap_ls(
            channel["received"],
            realization["pilots"],
            realization["pilot_mask"],
            estimator_num_taps,
            ridge_lambda,
            return_diagnostics=True,
        )
    estimate = estimate_channel_ls(
        channel["received"],
        realization["pilots"],
        realization["pilot_mask"],
        fading="multipath_block",
        channel_estimator=estimator,
    )
    return estimate, {
        "unique_pilot_subcarriers": torch.nonzero(realization["pilot_mask"].any(dim=(0, 2))).flatten(),
        "dft_matrix_condition_number": torch.tensor(float("nan")),
    }


def _row_for_estimator(
    *,
    seed: int,
    snr_db: float,
    jsr_db: float,
    jammer_type: str,
    estimator: str,
    realization: dict[str, Any],
    estimator_num_taps: int,
    ridge_lambda: float,
) -> dict[str, Any]:
    channel = realization["channel"]
    try:
        estimated, diagnostics = _estimate(
            estimator,
            realization,
            estimator_num_taps=estimator_num_taps,
            ridge_lambda=ridge_lambda,
        )
        oracle_metrics = data_resource_equalization_metrics(
            channel=channel,
            transmitted=realization["transmitted"],
            pilot_mask=realization["pilot_mask"],
            channel_estimate=channel["signal_fading"],
        )
        estimated_metrics = data_resource_equalization_metrics(
            channel=channel,
            transmitted=realization["transmitted"],
            pilot_mask=realization["pilot_mask"],
            channel_estimate=estimated,
        )
        nmse = csi_nmse(channel["signal_fading"], estimated)
        nmse_linear = _as_float(nmse)
        finite = all(
            math.isfinite(value)
            for value in (
                nmse_linear,
                estimated_metrics["post_eq_sinr_linear"],
                oracle_metrics["post_eq_sinr_linear"],
            )
        )
        status = "finite" if finite else "non_finite"
        error = ""
    except Exception as exc:
        estimated = None
        diagnostics = {}
        oracle_metrics = {}
        estimated_metrics = {}
        nmse_linear = float("nan")
        status = "error"
        error = str(exc)
    post_jsr = None if jammer_type == "none" else post_channel_jsr(
        channel["faded_signal"], channel["faded_jammer"]
    )
    post_jsr_db = None if jammer_type == "none" else post_channel_jsr(
        channel["faded_signal"], channel["faded_jammer"], db=True
    )
    pilot_evm_value = (
        float("nan")
        if estimated is None
        else _as_float(pilot_evm(channel["received"], realization["pilots"], realization["pilot_mask"], estimated))
    )
    unique = diagnostics.get("unique_pilot_subcarriers")
    condition = diagnostics.get("dft_matrix_condition_number")
    oracle_db = float(oracle_metrics.get("post_eq_sinr_db", float("nan")))
    estimated_db = float(estimated_metrics.get("post_eq_sinr_db", float("nan")))
    estimated_minus_oracle = estimated_db - oracle_db
    oracle_minus_estimated = oracle_db - estimated_db
    return {
        "seed": seed,
        "scenario_id": realization["scenario_id"],
        "channel_model": "multipath_block",
        "true_num_taps": int(channel.get("signal_taps", torch.empty(1, 0)).shape[1]),
        "estimator_num_taps": estimator_num_taps,
        "estimator": estimator,
        "ridge_lambda": ridge_lambda,
        "unique_pilot_subcarriers": int(unique.numel()) if isinstance(unique, torch.Tensor) else "",
        "dft_matrix_condition_number": _as_float(condition) if isinstance(condition, torch.Tensor) and condition.numel() else "",
        "requested_snr_db": snr_db,
        "requested_jsr_db": "" if jammer_type == "none" else jsr_db,
        "jammer_type": jammer_type,
        "csi_nmse_linear": nmse_linear,
        "csi_nmse_db": 10.0 * math.log10(max(nmse_linear, 1e-12)) if math.isfinite(nmse_linear) else float("nan"),
        "pilot_evm": pilot_evm_value,
        "post_channel_jsr_linear": "" if post_jsr is None else _as_float(post_jsr),
        "post_channel_jsr_db": "" if post_jsr_db is None else _as_float(post_jsr_db),
        "oracle_post_eq_signal_power": oracle_metrics.get("post_eq_signal_power", float("nan")),
        "oracle_post_eq_interference_power": oracle_metrics.get("post_eq_interference_power", float("nan")),
        "oracle_post_eq_noise_power": oracle_metrics.get("post_eq_noise_power", float("nan")),
        "oracle_post_eq_sinr_linear": oracle_metrics.get("post_eq_sinr_linear", float("nan")),
        "oracle_post_eq_sinr_db": oracle_db,
        "estimated_post_eq_signal_power": estimated_metrics.get("post_eq_signal_power", float("nan")),
        "estimated_post_eq_interference_power": estimated_metrics.get("post_eq_interference_power", float("nan")),
        "estimated_post_eq_noise_power": estimated_metrics.get("post_eq_noise_power", float("nan")),
        "estimated_post_eq_sinr_linear": estimated_metrics.get("post_eq_sinr_linear", float("nan")),
        "estimated_post_eq_sinr_db": estimated_db,
        "estimated_minus_oracle_sinr_db": estimated_minus_oracle,
        "oracle_minus_estimated_sinr_db": oracle_minus_estimated,
        "equalized_symbol_mse": estimated_metrics.get("equalized_symbol_mse", float("nan")),
        "data_evm": estimated_metrics.get("data_evm", float("nan")),
        "oracle_equalized_symbol_mse": oracle_metrics.get("equalized_symbol_mse", float("nan")),
        "oracle_data_evm": oracle_metrics.get("data_evm", float("nan")),
        "data_resource_count": estimated_metrics.get("data_resource_count", ""),
        "pilot_resource_count": estimated_metrics.get("pilot_resource_count", ""),
        "min_h_abs2_data": oracle_metrics.get("min_h_abs2_data", float("nan")),
        "p01_h_abs2_data": oracle_metrics.get("p01_h_abs2_data", float("nan")),
        "p05_h_abs2_data": oracle_metrics.get("p05_h_abs2_data", float("nan")),
        "max_oracle_equalizer_gain": oracle_metrics.get("max_equalizer_gain", float("nan")),
        "max_estimated_equalizer_gain": estimated_metrics.get("max_equalizer_gain", float("nan")),
        "max_oracle_equalizer_gain_abs2": oracle_metrics.get("max_equalizer_gain_abs2", float("nan")),
        "max_estimated_equalizer_gain_abs2": estimated_metrics.get("max_equalizer_gain_abs2", float("nan")),
        "mean_oracle_equalizer_gain_abs2": oracle_metrics.get("mean_equalizer_gain_abs2", float("nan")),
        "mean_estimated_equalizer_gain_abs2": estimated_metrics.get("mean_equalizer_gain_abs2", float("nan")),
        "signal_fading_hash": realization["signal_fading_hash"],
        "jammer_fading_hash": realization["jammer_fading_hash"],
        "noise_hash": realization["noise_hash"],
        "transmitted_hash": realization["transmitted_hash"],
        "pilot_mask_hash": realization["pilot_mask_hash"],
        "jammer_mask_hash": realization["jammer_mask_hash"],
        "finite_status": status,
        "status": status,
        "error": error,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["estimator"], float(row["requested_snr_db"]), row["jammer_type"])].append(row)
    metrics = (
        "csi_nmse_linear",
        "csi_nmse_db",
        "pilot_evm",
        "estimated_post_eq_sinr_db",
        "estimated_post_eq_sinr_linear",
        "oracle_post_eq_sinr_db",
        "oracle_post_eq_sinr_linear",
        "estimated_minus_oracle_sinr_db",
        "oracle_minus_estimated_sinr_db",
        "equalized_symbol_mse",
        "data_evm",
        "oracle_equalized_symbol_mse",
        "oracle_data_evm",
    )
    output = []
    for (estimator, snr, jammer), selected in sorted(grouped.items()):
        row: dict[str, Any] = {
            "estimator": estimator,
            "requested_snr_db": snr,
            "jammer_type": jammer,
            "sample_count": len(selected),
        }
        for metric in metrics:
            row[metric] = _distribution([float(item[metric]) for item in selected])
        row["estimated_post_eq_sinr_db"]["mean_per_seed_sinr_db"] = row[
            "estimated_post_eq_sinr_db"
        ]["mean"]
        row["oracle_post_eq_sinr_db"]["mean_per_seed_sinr_db"] = row[
            "oracle_post_eq_sinr_db"
        ]["mean"]
        row["estimated_post_eq_sinr_db"]["db_of_mean_linear_sinr"] = (
            10.0
            * math.log10(
                max(
                    row["estimated_post_eq_sinr_linear"]["mean"]
                    if row["estimated_post_eq_sinr_linear"]["mean"] is not None
                    else float("nan"),
                    1e-12,
                )
            )
        )
        row["oracle_post_eq_sinr_db"]["db_of_mean_linear_sinr"] = (
            10.0
            * math.log10(
                max(
                    row["oracle_post_eq_sinr_linear"]["mean"]
                    if row["oracle_post_eq_sinr_linear"]["mean"] is not None
                    else float("nan"),
                    1e-12,
                )
            )
        )
        output.append(row)
    return output


def _assert_row_consistency(row: dict[str, Any], *, tol: float = 1e-5) -> None:
    if row["finite_status"] != "finite":
        return
    estimated_linear = float(row["estimated_post_eq_sinr_linear"])
    oracle_linear = float(row["oracle_post_eq_sinr_linear"])
    if not math.isfinite(estimated_linear) or estimated_linear <= 0:
        raise ValueError("estimated SINR linear must be finite and positive")
    if not math.isfinite(oracle_linear) or oracle_linear <= 0:
        raise ValueError("oracle SINR linear must be finite and positive")
    estimated_db = float(row["estimated_post_eq_sinr_db"])
    oracle_db = float(row["oracle_post_eq_sinr_db"])
    if abs(estimated_db - 10.0 * math.log10(estimated_linear)) > tol:
        raise ValueError("estimated SINR dB is inconsistent with linear SINR")
    if abs(oracle_db - 10.0 * math.log10(oracle_linear)) > tol:
        raise ValueError("oracle SINR dB is inconsistent with linear SINR")
    estimated_minus = float(row["estimated_minus_oracle_sinr_db"])
    oracle_minus = float(row["oracle_minus_estimated_sinr_db"])
    if abs(estimated_minus - (estimated_db - oracle_db)) > tol:
        raise ValueError("estimated-minus-oracle SINR field is inconsistent")
    if abs(oracle_minus - (oracle_db - estimated_db)) > tol:
        raise ValueError("oracle-minus-estimated SINR field is inconsistent")
    if abs(estimated_minus + oracle_minus) > tol:
        raise ValueError("SINR difference fields must be exact negatives")
    if int(row["data_resource_count"]) <= 0:
        raise ValueError("data_resource_count must be positive")


def validate_pairing_and_consistency(rows: list[dict[str, Any]], estimators: list[str]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        _assert_row_consistency(row)
        grouped[str(row["scenario_id"])].append(row)
    for scenario_id, selected in grouped.items():
        seen = [row["estimator"] for row in selected]
        if sorted(seen) != sorted(estimators):
            raise ValueError(f"scenario {scenario_id} has missing or duplicated estimator rows")
        keys = (
            "signal_fading_hash",
            "jammer_fading_hash",
            "noise_hash",
            "transmitted_hash",
            "pilot_mask_hash",
            "jammer_mask_hash",
            "requested_snr_db",
            "requested_jsr_db",
            "oracle_post_eq_sinr_db",
            "oracle_post_eq_sinr_linear",
        )
        for key in keys:
            values = {str(row[key]) for row in selected}
            if len(values) != 1:
                raise ValueError(f"scenario {scenario_id} is not paired for {key}")


def validate_aggregate_consistency(rows: list[dict[str, Any]], aggregate: list[dict[str, Any]], *, tol: float = 1e-5) -> None:
    for agg in aggregate:
        selected = [
            row
            for row in rows
            if row["estimator"] == agg["estimator"]
            and float(row["requested_snr_db"]) == float(agg["requested_snr_db"])
            and row["jammer_type"] == agg["jammer_type"]
            and row["finite_status"] == "finite"
        ]
        if not selected:
            continue
        mean_est = sum(float(row["estimated_post_eq_sinr_db"]) for row in selected) / len(selected)
        mean_oracle = sum(float(row["oracle_post_eq_sinr_db"]) for row in selected) / len(selected)
        mean_diff = sum(float(row["estimated_minus_oracle_sinr_db"]) for row in selected) / len(selected)
        if abs(mean_diff - (mean_est - mean_oracle)) > tol:
            raise ValueError("aggregate mean SINR difference is inconsistent with paired dB means")
        if abs(mean_diff - float(agg["estimated_minus_oracle_sinr_db"]["mean"])) > tol:
            raise ValueError("aggregate report disagrees with per-seed SINR differences")


def _write_report_md(path: Path, report: dict[str, Any]) -> None:
    lines = ["# Channel Estimator Comparison", ""]
    lines.append(f"- channel model: `{report['metadata']['channel_model']}`")
    lines.append(f"- estimators: `{report['metadata']['estimators']}`")
    lines.append("")
    lines += [
        "",
        "## SINR Definition",
        "",
        "Post-equalization SINR is computed over data resources only. Pilot positions are excluded before power averaging, so zero-filled or pilot resources do not dilute the metric.",
        "For an equalizer `G`, the diagnostic computes `G H_s X`, `G H_j J`, and `G W`, averages their powers over data resources, then reports `10 log10(P_signal / (P_interference + P_noise))`.",
        "Both `estimated_minus_oracle_sinr_db` and `oracle_minus_estimated_sinr_db` are stored without clamping. Positive estimated-minus-oracle means the estimated ZF equalizer has higher aggregate data-resource SINR than oracle ZF for that realization.",
        "Mean per-seed SINR in dB and dB of mean linear SINR are different and are stored separately in JSON aggregates.",
        "Estimated ZF can occasionally exceed oracle ZF in this aggregate diagnostic because zero-forcing equalizer gain changes the weighting of noise/interference across deep fades; this report does not claim oracle ZF maximizes every aggregate weighting.",
        "",
    ]
    lines.append("| estimator | SNR | jammer | median NMSE | mean NMSE | mean est SINR dB | mean oracle SINR dB | mean est-oracle dB |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for row in report["aggregate"]:
        lines.append(
            f"| {row['estimator']} | {row['requested_snr_db']} | {row['jammer_type']} | "
            f"{row['csi_nmse_linear']['median']} | {row['csi_nmse_linear']['mean']} | "
            f"{row['estimated_post_eq_sinr_db']['mean']} | "
            f"{row['oracle_post_eq_sinr_db']['mean']} | "
            f"{row['estimated_minus_oracle_sinr_db']['mean']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _manual_audit_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["jammer_type"] == "none"
        and float(row["requested_snr_db"]) == 30.0
        and row["estimator"] == "dft_tap_ls"
        and row["finite_status"] == "finite"
    ]
    selected = sorted(selected, key=lambda row: int(row["seed"]))[:3]
    audit = []
    for row in selected:
        _assert_row_consistency(row)
        audit.append(
            {
                "seed": int(row["seed"]),
                "scenario_id": row["scenario_id"],
                "oracle_post_eq_signal_power": row["oracle_post_eq_signal_power"],
                "oracle_post_eq_interference_power": row["oracle_post_eq_interference_power"],
                "oracle_post_eq_noise_power": row["oracle_post_eq_noise_power"],
                "estimated_post_eq_signal_power": row["estimated_post_eq_signal_power"],
                "estimated_post_eq_interference_power": row["estimated_post_eq_interference_power"],
                "estimated_post_eq_noise_power": row["estimated_post_eq_noise_power"],
                "oracle_post_eq_sinr_linear": row["oracle_post_eq_sinr_linear"],
                "oracle_post_eq_sinr_db": row["oracle_post_eq_sinr_db"],
                "estimated_post_eq_sinr_linear": row["estimated_post_eq_sinr_linear"],
                "estimated_post_eq_sinr_db": row["estimated_post_eq_sinr_db"],
                "direct_estimated_minus_oracle_sinr_db": row[
                    "estimated_post_eq_sinr_db"
                ]
                - row["oracle_post_eq_sinr_db"],
                "stored_estimated_minus_oracle_sinr_db": row[
                    "estimated_minus_oracle_sinr_db"
                ],
                "stored_oracle_minus_estimated_sinr_db": row[
                    "oracle_minus_estimated_sinr_db"
                ],
                "data_resource_count": row["data_resource_count"],
                "pilot_resource_count": row["pilot_resource_count"],
                "max_oracle_equalizer_gain": row["max_oracle_equalizer_gain"],
                "max_estimated_equalizer_gain": row["max_estimated_equalizer_gain"],
            }
        )
    return audit


def _save_plots(output: Path, aggregate: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_dir = output / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        clean = [row for row in aggregate if row["jammer_type"] == "none"]
        for metric_key, filename, ylabel in (
            ("csi_nmse_linear", "csi_nmse_vs_snr.png", "CSI NMSE"),
            (
                "oracle_minus_estimated_sinr_db",
                "oracle_minus_estimated_sinr_vs_snr.png",
                "Oracle minus estimated SINR (dB)",
            ),
        ):
            fig, axis = plt.subplots(figsize=(6, 4))
            for estimator in sorted({row["estimator"] for row in clean}):
                selected = sorted(
                    [row for row in clean if row["estimator"] == estimator],
                    key=lambda item: float(item["requested_snr_db"]),
                )
                if not selected:
                    continue
                axis.plot(
                    [float(row["requested_snr_db"]) for row in selected],
                    [row[metric_key]["median"] for row in selected],
                    marker="o",
                    label=estimator,
                )
            axis.set(xlabel="Requested SNR (dB)", ylabel=ylabel, title=ylabel)
            axis.grid(alpha=0.25)
            if axis.get_legend_handles_labels()[0]:
                axis.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / filename, dpi=150)
            plt.close(fig)
    except Exception as exc:
        print(f"warning: estimator comparison plot generation failed: {exc}")


def _warn_if_dft_not_better(aggregate: list[dict[str, Any]]) -> None:
    by_key = {
        (row["estimator"], float(row["requested_snr_db"]), row["jammer_type"]): row
        for row in aggregate
    }
    for _, snr, jammer in list(by_key):
        block = by_key.get(("block_frequency_ls", snr, jammer))
        dft = by_key.get(("dft_tap_ls", snr, jammer))
        if not block or not dft:
            continue
        if dft["csi_nmse_linear"]["median"] >= block["csi_nmse_linear"]["median"]:
            print(
                "warning: dft_tap_ls did not improve median CSI NMSE over "
                f"block_frequency_ls for snr={snr:g} jammer={jammer}"
            )


def run_estimator_comparison(
    config_path: str | Path,
    *,
    snr_values: list[float],
    jsr_values: list[float],
    jammer_types: list[str],
    num_seeds: int,
    estimators: list[str],
    output_dir: str | Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    device = resolve_device(config.get("device", "auto"))
    channel = config["channel"]
    estimator_num_taps = int(channel.get("estimator_num_taps", channel.get("num_taps", 6)))
    ridge_lambda = float(channel.get("estimator_ridge_lambda", 1e-6))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    rows = []
    base_seed = int(config.get("seed", 0))
    for snr in snr_values:
        for jammer_type in jammer_types:
            jsr_iter = [0.0] if jammer_type == "none" else jsr_values
            for jsr in jsr_iter:
                for offset in range(int(num_seeds)):
                    realization = _make_realization(
                        config,
                        seed=base_seed + offset,
                        snr_db=float(snr),
                        jsr_db=float(jsr),
                        jammer_type=jammer_type,
                        device=device,
                    )
                    for estimator in estimators:
                        rows.append(
                            _row_for_estimator(
                                seed=base_seed + offset,
                                snr_db=float(snr),
                                jsr_db=float(jsr),
                                jammer_type=jammer_type,
                                estimator=estimator,
                                realization=realization,
                                estimator_num_taps=estimator_num_taps,
                                ridge_lambda=ridge_lambda,
                            )
                        )
    aggregate = _aggregate(rows)
    validate_pairing_and_consistency(rows, estimators)
    validate_aggregate_consistency(rows, aggregate)
    report = {
        "metadata": {
            "channel_model": "multipath_block",
            "true_num_taps": int(channel.get("num_taps", 6)),
            "pdp": channel.get("pdp", "exponential"),
            "pdp_decay": float(channel.get("pdp_decay", 0.7)),
            "grid_shape": channel.get("grid_shape", [32, 16]),
            "pilot_spacing": int(channel.get("pilot_spacing", 4)),
            "pilot_time_spacing": channel.get("pilot_time_spacing", 4),
            "ideal_cp": bool(channel.get("assume_ideal_cp", True)),
            "block_fading": bool(channel.get("block_fading_over_time", True)),
            "estimators": estimators,
            "channel_estimator": {
                "name": "dft_tap_ls",
                "estimator_num_taps": estimator_num_taps,
                "ridge_lambda": ridge_lambda,
                "pilot_time_averaging": True,
                "reconstruction_domain": "tap",
                "dft_convention": "torch_fft_forward",
                "uses_true_channel": False,
            },
            "git_commit": _git_commit(),
            "config_hash": _config_hash(config),
            "sinr_audit_root_cause": (
                "previous reports averaged a clamped per-seed oracle-minus-estimated "
                "field named sinr_loss_vs_oracle_db; that field was not the signed "
                "difference between aggregate mean oracle and estimated SINR. The old "
                "calculation also used all grid resources instead of data resources only."
            ),
        },
        "aggregate": aggregate,
        "manual_audit_clean_30db": _manual_audit_records(rows),
    }
    _write_csv(output / "per_seed_results.csv", rows)
    flat_aggregate = []
    for row in aggregate:
        flattened = {k: v for k, v in row.items() if not isinstance(v, dict)}
        for metric, stats in row.items():
            if isinstance(stats, dict):
                for stat_name, value in stats.items():
                    flattened[f"{metric}_{stat_name}"] = value
        flat_aggregate.append(flattened)
    _write_csv(output / "aggregate_results.csv", flat_aggregate)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report_md(output / "report.md", report)
    _warn_if_dft_not_better(aggregate)
    _save_plots(output, aggregate)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare OFDM channel estimators on shared multipath realizations")
    parser.add_argument("--config", default="configs/eval_multipath_channel.yaml")
    parser.add_argument("--snr_values", nargs="+", type=float, default=[30.0, 20.0, 10.0, 5.0])
    parser.add_argument("--jsr_values", nargs="+", type=float, default=[0.0])
    parser.add_argument("--jammer_types", nargs="+", default=["none"])
    parser.add_argument("--num_seeds", type=int, default=100)
    parser.add_argument(
        "--estimators",
        nargs="+",
        default=["inverse_distance_2d", "block_frequency_ls", "dft_tap_ls", "oracle"],
    )
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    run_estimator_comparison(
        args.config,
        snr_values=args.snr_values,
        jsr_values=args.jsr_values,
        jammer_types=args.jammer_types,
        num_seeds=args.num_seeds,
        estimators=args.estimators,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
