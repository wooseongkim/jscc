from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import Tensor


def _validate_latents(reconstruction: Tensor, target: Tensor) -> None:
    if reconstruction.shape != target.shape or target.ndim != 4:
        raise ValueError("reconstruction and target must match [B,L,T,D]")
    if not torch.is_floating_point(reconstruction) or not torch.is_floating_point(target):
        raise TypeError("reconstruction and target must be floating-point tensors")


def normalized_layer_loss(reconstruction: Tensor, target: Tensor, epsilon: float) -> Tensor:
    """Return per-layer power-normalized MSE reduced over batch, time, and latent axes."""
    _validate_latents(reconstruction, target)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    raw_mse = (reconstruction - target).square().mean(dim=(0, 2, 3))
    target_power = target.square().mean(dim=(0, 2, 3)).detach()
    return raw_mse / target_power.clamp_min(float(epsilon))


def zero_predictor_loss(target: Tensor, weights: Tensor, epsilon: float) -> tuple[Tensor, Tensor]:
    """Return weighted and per-layer normalized losses for an all-zero reconstruction."""
    if target.ndim != 4:
        raise ValueError("target must have shape [B,L,T,D]")
    weights = weights.to(device=target.device, dtype=target.dtype)
    if weights.shape != (target.shape[1],) or torch.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must contain L nonnegative values with positive sum")
    layers = normalized_layer_loss(torch.zeros_like(target), target, epsilon)
    return (layers * weights).sum() / weights.sum(), layers


def _safe_cosine(reconstruction: Tensor, target: Tensor, epsilon: float) -> float:
    denominator = reconstruction.norm() * target.norm()
    if float(denominator) <= epsilon:
        return 0.0
    return float(torch.dot(reconstruction, target) / denominator)


def _safe_correlation(reconstruction: Tensor, target: Tensor, epsilon: float) -> tuple[float, bool]:
    reconstruction_centered = reconstruction - reconstruction.mean()
    target_centered = target - target.mean()
    denominator = reconstruction_centered.norm() * target_centered.norm()
    if float(denominator) <= epsilon:
        return 0.0, True
    return float(torch.dot(reconstruction_centered, target_centered) / denominator), False


def latent_metric_rows(
    reconstruction: Tensor,
    target: Tensor,
    *,
    epsilon: float,
    predictor: str,
    scenario: str,
    sample_ids: Sequence[str] | None = None,
    near_zero_threshold: float = 1.0e-4,
) -> list[dict[str, Any]]:
    """Calculate the required Stage-1 reconstruction metrics per sample and layer."""
    _validate_latents(reconstruction, target)
    if epsilon <= 0 or near_zero_threshold < 0:
        raise ValueError("epsilon must be positive and near_zero_threshold nonnegative")
    if sample_ids is None:
        sample_ids = [str(index) for index in range(target.shape[0])]
    if len(sample_ids) != target.shape[0]:
        raise ValueError("sample_ids must contain one value per batch item")

    rows: list[dict[str, Any]] = []
    for sample_index, sample_id in enumerate(sample_ids):
        for layer in range(target.shape[1]):
            current_target = target[sample_index, layer].detach().flatten().float()
            current_reconstruction = reconstruction[sample_index, layer].detach().flatten().float()
            target_power = current_target.square().mean()
            reconstruction_power = current_reconstruction.square().mean()
            raw_mse = (current_reconstruction - current_target).square().mean()
            denominator = target_power.clamp_min(float(epsilon))
            normalized_mse = raw_mse / denominator
            zero_mse = target_power / denominator
            correlation, correlation_degenerate = _safe_correlation(
                current_reconstruction, current_target, epsilon
            )
            values = torch.stack(
                (
                    target_power,
                    reconstruction_power,
                    reconstruction_power / denominator,
                    raw_mse,
                    normalized_mse,
                    zero_mse,
                    normalized_mse - zero_mse,
                    current_target.mean(),
                    current_reconstruction.mean(),
                    current_target.std(unbiased=False),
                    current_reconstruction.std(unbiased=False),
                    (current_reconstruction - current_target).mean(),
                    (current_reconstruction.abs() < near_zero_threshold).float().mean(),
                )
            )
            cosine = _safe_cosine(current_reconstruction, current_target, epsilon)
            numeric = [float(value) for value in values]
            numeric.extend((cosine, correlation))
            rows.append(
                {
                    "sample_index": sample_index,
                    "sample_id": str(sample_id),
                    "scenario": scenario,
                    "predictor": predictor,
                    "layer": layer,
                    "target_power": float(target_power),
                    "reconstruction_power": float(reconstruction_power),
                    "power_ratio": float(reconstruction_power / denominator),
                    "raw_mse": float(raw_mse),
                    "normalized_mse": float(normalized_mse),
                    "zero_normalized_mse": float(zero_mse),
                    "trained_minus_zero_normalized_mse": float(normalized_mse - zero_mse),
                    "cosine_similarity": cosine,
                    "pearson_correlation": correlation,
                    "correlation_degenerate": correlation_degenerate,
                    "target_mean": float(current_target.mean()),
                    "reconstruction_mean": float(current_reconstruction.mean()),
                    "target_std": float(current_target.std(unbiased=False)),
                    "reconstruction_std": float(current_reconstruction.std(unbiased=False)),
                    "reconstruction_bias": float((current_reconstruction - current_target).mean()),
                    "near_zero_fraction": float(
                        (current_reconstruction.abs() < near_zero_threshold).float().mean()
                    ),
                    "finite": all(torch.isfinite(torch.tensor(numeric)).tolist()),
                }
            )
    return rows


def aggregate_latent_rows(
    rows: Iterable[dict[str, Any]],
    group_keys: Sequence[str],
) -> list[dict[str, Any]]:
    """Mean numeric diagnostic fields without collapsing requested group boundaries."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    output: list[dict[str, Any]] = []
    for group, members in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        aggregate: dict[str, Any] = dict(zip(group_keys, group, strict=True))
        aggregate["count"] = len(members)
        shared_keys = set.intersection(*(set(member) for member in members))
        for key in sorted(shared_keys - set(group_keys)):
            values = [member[key] for member in members]
            if all(isinstance(value, bool) for value in values):
                aggregate[key] = all(values)
            elif all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
                aggregate[key] = sum(float(value) for value in values) / len(values)
        output.append(aggregate)
    return output
