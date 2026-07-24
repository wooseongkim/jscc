from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any

import torch
from torch import Tensor

from speech_jscc.diagnostics.random_distribution import SeedDeriver


ORACLE_CSI_COMPARISON_VERSION = "oracle_csi_comparison_v1"
ZF_EPSILON = 1.0e-6


def paired_evaluation_grid(
    root_seed: int,
    *,
    utterance_count: int,
    realizations: int,
    snr_values: tuple[float, ...] = (5.0, 10.0, 15.0),
) -> list[dict[str, Any]]:
    derive = SeedDeriver(root_seed)
    rows = []
    for utterance in range(utterance_count):
        for snr in snr_values:
            for realization in range(realizations):
                rows.append({
                    "utterance_index": utterance,
                    "realization": realization,
                    "snr_db": float(snr),
                    "seed": derive.seed(
                        "oracle_csi_pair", utterance,
                        f"{float(snr)}|{realization}",
                    ),
                })
    return rows


def tensor_hash(value: Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def equalizer_coefficient(channel: Tensor, eps: float = ZF_EPSILON) -> Tensor:
    return channel.conj() / channel.abs().square().clamp_min(eps)


def _complex_correlation(estimate: Tensor, target: Tensor, eps: float = 1e-12) -> float:
    numerator = (estimate * target.conj()).sum().abs()
    denominator = (
        estimate.abs().square().sum().sqrt()
        * target.abs().square().sum().sqrt()
    ).clamp_min(eps)
    return float(numerator / denominator)


def empirical_symbol_metrics(
    transmitted: Tensor,
    equalized: Tensor,
    *,
    h: Tensor,
    requested_noise_power: float,
    oracle: bool,
    eps: float = 1e-12,
) -> dict[str, float]:
    signal_energy = transmitted.abs().square().sum().clamp_min(eps)
    residual_energy = (equalized - transmitted).abs().square().sum().clamp_min(eps)
    empirical = signal_energy / residual_energy
    output = {
        "transmit_symbol_power": float(transmitted.abs().square().mean()),
        "equalized_symbol_nmse": float(residual_energy / signal_energy),
        "equalized_symbol_correlation": _complex_correlation(equalized, transmitted),
        "post_eq_sinr_empirical_linear": float(empirical),
        "post_eq_sinr_empirical_db": float(10 * torch.log10(empirical)),
    }
    if oracle:
        theory_noise = (
            float(requested_noise_power) / h.abs().square().clamp_min(eps)
        ).sum()
        theory = signal_energy / theory_noise.clamp_min(eps)
        output.update({
            "post_eq_sinr_oracle_theory_linear": float(theory),
            "post_eq_sinr_oracle_theory_db": float(10 * torch.log10(theory)),
            "oracle_empirical_minus_theory_db": float(
                10 * torch.log10(empirical) - 10 * torch.log10(theory)
            ),
        })
    return output


def residual_decomposition(
    transmitted: Tensor,
    true_channel: Tensor,
    estimated_channel: Tensor,
    noise: Tensor,
    *,
    eps: float = ZF_EPSILON,
) -> dict[str, Any]:
    coefficient = equalizer_coefficient(estimated_channel, eps)
    noise_component = noise * coefficient
    csi_component = (true_channel * coefficient - 1.0) * transmitted
    total = noise_component + csi_component
    signal_energy = transmitted.abs().square().sum().clamp_min(1e-12)
    noise_energy = noise_component.abs().square().sum()
    csi_energy = csi_component.abs().square().sum()
    cross = 2.0 * (noise_component * csi_component.conj()).real.sum()
    total_energy = total.abs().square().sum()
    return {
        "noise_component": noise_component,
        "csi_distortion_component": csi_component,
        "total_residual": total,
        "noise_component_energy_ratio": float(noise_energy / signal_energy),
        "csi_distortion_energy_ratio": float(csi_energy / signal_energy),
        "cross_term_energy_ratio": float(cross / signal_energy),
        "total_residual_energy_ratio": float(total_energy / signal_energy),
    }


def channel_power_statistics(channel: Tensor) -> dict[str, float]:
    power = channel.abs().square().flatten()
    return {
        "mean_h_power": float(power.mean()),
        "median_h_power": float(power.median()),
        "minimum_h_power": float(power.min()),
        "h_power_p05": float(torch.quantile(power, 0.05)),
        "h_power_p10": float(torch.quantile(power, 0.10)),
        "h_power_below_0_1_fraction": float((power < 0.1).float().mean()),
        "h_power_below_0_01_fraction": float((power < 0.01).float().mean()),
    }


def summarize_distribution(
    values: list[float],
    *,
    higher_is_better: bool,
    bootstrap_seed: int,
    bootstrap_samples: int = 1000,
) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    rng = random.Random(bootstrap_seed)
    means = []
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(len(values)) for _ in values]
        means.append(sum(values[index] for index in indices) / len(indices))
    means.sort()
    lower = means[int(0.025 * (bootstrap_samples - 1))]
    upper = means[int(0.975 * (bootstrap_samples - 1))]
    tail_q = 0.1 if higher_is_better else 0.9
    return {
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
        "median": float(tensor.median()),
        "worst_decile": float(torch.quantile(tensor, tail_q)),
        "bootstrap_mean_ci95_lower": float(lower),
        "bootstrap_mean_ci95_upper": float(upper),
    }


def suite_hash(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def db(value: float, eps: float = 1e-12) -> float:
    return 10.0 * math.log10(max(float(value), eps))
