from __future__ import annotations

import torch
from torch import Tensor


def _broadcast_batch(value: float | Tensor, reference: Tensor) -> Tensor:
    result = torch.as_tensor(value, device=reference.device, dtype=reference.real.dtype)
    if result.ndim == 0:
        result = result.repeat(reference.shape[0])
    if result.shape == (reference.shape[0],):
        result = result.reshape(reference.shape[0], *([1] * (reference.ndim - 1)))
    try:
        return torch.broadcast_to(result, reference.shape)
    except RuntimeError as error:
        raise ValueError("reliability component is not broadcastable to channel grid") from error


def compute_resource_reliability(
    channel_estimate: Tensor,
    jammer_power: float | Tensor,
    noise_power: float | Tensor,
    csi_confidence: float | Tensor = 1.0,
    *,
    eps: float = 1e-12,
) -> Tensor:
    """Compute `|H_hat|^2 confidence / (P_jammer + P_noise)` per resource."""
    if not torch.is_complex(channel_estimate) or channel_estimate.ndim not in (2, 3):
        raise ValueError("channel_estimate must be complex [B,M] or [B,K,N]")
    jammer = _broadcast_batch(jammer_power, channel_estimate).clamp_min(0.0)
    noise = _broadcast_batch(noise_power, channel_estimate).clamp_min(0.0)
    confidence = _broadcast_batch(csi_confidence, channel_estimate).clamp(0.0, 1.0)
    return channel_estimate.abs().square() * confidence / (jammer + noise).clamp_min(eps)


def estimate_unreliable_mask(reliability: Tensor, unreliable_fraction: float) -> Tensor:
    """Mark the lowest-reliability fraction independently for each batch item."""
    if reliability.ndim not in (2, 3):
        raise ValueError("reliability must be [B,M] or [B,K,N]")
    if not 0.0 <= unreliable_fraction <= 1.0:
        raise ValueError("unreliable_fraction must be in [0,1]")
    flat = reliability.flatten(1)
    count = round(flat.shape[1] * unreliable_fraction)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    if count > 0:
        indices = flat.argsort(dim=1)[:, :count]
        mask.scatter_(1, indices, True)
    return mask.reshape_as(reliability)


__all__ = ["compute_resource_reliability", "estimate_unreliable_mask"]

