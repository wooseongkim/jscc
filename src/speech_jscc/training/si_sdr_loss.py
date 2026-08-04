"""Differentiable, unaligned full-crop zero-mean SI-SDR training loss."""
from __future__ import annotations

import torch
from torch import Tensor


def si_sdr(estimate: Tensor, target: Tensor, *, eps: float = 1e-8) -> Tensor:
    if estimate.shape != target.shape or estimate.ndim < 2:
        raise ValueError("estimate and target must share [B,...,samples] shape")
    estimate_zm = estimate - estimate.mean(dim=-1, keepdim=True)
    target_zm = target - target.mean(dim=-1, keepdim=True)
    target_energy = target_zm.square().sum(dim=-1, keepdim=True)
    alpha = (estimate_zm * target_zm).sum(dim=-1, keepdim=True) / (target_energy + eps)
    projection = alpha * target_zm
    error = estimate_zm - projection
    ratio = (projection.square().sum(dim=-1) + eps) / (error.square().sum(dim=-1) + eps)
    return 10.0 * torch.log10(ratio)


def negative_si_sdr_loss(estimate: Tensor, target: Tensor, *, eps: float = 1e-8, clip_db: float | None = 30.0) -> tuple[Tensor, dict[str, Tensor]]:
    raw = si_sdr(estimate, target, eps=eps)
    used = raw.clamp(-float(clip_db), float(clip_db)) if clip_db is not None else raw
    if not torch.isfinite(used).all():
        raise FloatingPointError("nonfinite SI-SDR loss")
    diagnostics = {"raw_si_sdr_db": raw, "loss_si_sdr_db": used, "target_rms": target.square().mean(-1).sqrt(), "reconstructed_rms": estimate.square().mean(-1).sqrt(), "finite_fraction": torch.isfinite(raw).float().mean(), "low_energy_fraction": (target.square().mean(-1).sqrt() < 1e-4).float().mean()}
    return -used.mean(), diagnostics
