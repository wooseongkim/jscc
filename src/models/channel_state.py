from __future__ import annotations

import torch
from torch import Tensor


JAMMER_TYPES = ("barrage", "narrowband", "burst", "pilot")
CHANNEL_STATE_DIM = 8
SINR_INDEX = 0
JSR_INDEX = 1
CSI_NMSE_INDEX = 2
JAMMER_POSTERIOR_SLICE = slice(3, 7)
MASK_RATIO_INDEX = 7


def rule_based_jammer_posterior(
    jammer_type: str,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return a deterministic one-hot posterior in the documented jammer order."""
    if jammer_type not in JAMMER_TYPES:
        raise ValueError(f"jammer_type must be one of {JAMMER_TYPES}")
    posterior = torch.zeros(batch_size, len(JAMMER_TYPES), device=device, dtype=dtype)
    posterior[:, JAMMER_TYPES.index(jammer_type)] = 1.0
    return posterior


def _batch_values(value: float | Tensor, batch_size: int, reference: Tensor) -> Tensor:
    result = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if result.ndim == 0:
        result = result.repeat(batch_size)
    if result.shape != (batch_size,):
        raise ValueError(f"state component must be scalar or [B], got {tuple(result.shape)}")
    return result


def build_channel_state(
    effective_sinr: Tensor,
    jsr: Tensor,
    csi_nmse: Tensor,
    jammer_posterior: Tensor,
    mask_ratio: float | Tensor,
    *,
    eps: float = 1e-12,
) -> Tensor:
    """Build `[SINR_dB/20, JSR_dB/20, NMSE, posterior(4), mask_ratio]`.

    SINR and JSR inputs are linear power ratios. Log scaling keeps the MLP input
    numerically comparable to the probability and mask features.
    """
    if effective_sinr.ndim != 1 or jsr.ndim != 1 or csi_nmse.ndim != 1:
        raise ValueError("effective_sinr, jsr, and csi_nmse must have shape [B]")
    batch_size = effective_sinr.shape[0]
    if jsr.shape != (batch_size,) or csi_nmse.shape != (batch_size,):
        raise ValueError("all scalar channel measurements must share batch size")
    if jammer_posterior.shape != (batch_size, len(JAMMER_TYPES)):
        raise ValueError("jammer_posterior must have shape [B,4]")
    posterior = jammer_posterior.to(effective_sinr)
    mask = _batch_values(mask_ratio, batch_size, effective_sinr).clamp(0.0, 1.0)
    state = effective_sinr.new_empty((batch_size, CHANNEL_STATE_DIM))
    state[:, SINR_INDEX] = 10.0 * torch.log10(effective_sinr.clamp_min(eps)) / 20.0
    state[:, JSR_INDEX] = 10.0 * torch.log10(jsr.clamp_min(eps)) / 20.0
    state[:, CSI_NMSE_INDEX] = csi_nmse.clamp_min(0.0)
    state[:, JAMMER_POSTERIOR_SLICE] = posterior
    state[:, MASK_RATIO_INDEX] = mask
    return state


def nominal_channel_state(
    snr_db: Tensor,
    jsr_db: Tensor,
    jammer_type: str,
    mask_ratio: float | Tensor,
) -> Tensor:
    """Bootstrap state before pilot feedback is available."""
    if snr_db.shape != jsr_db.shape or snr_db.ndim != 1:
        raise ValueError("snr_db and jsr_db must have matching [B] shapes")
    posterior = rule_based_jammer_posterior(
        jammer_type,
        snr_db.shape[0],
        device=snr_db.device,
        dtype=snr_db.dtype,
    )
    snr = torch.pow(10.0, snr_db / 10.0)
    jsr = torch.pow(10.0, jsr_db / 10.0)
    effective = 1.0 / (snr.reciprocal() + jsr).clamp_min(1e-12)
    return build_channel_state(effective, jsr, torch.zeros_like(snr_db), posterior, mask_ratio)


__all__ = [
    "CHANNEL_STATE_DIM",
    "CSI_NMSE_INDEX",
    "JAMMER_POSTERIOR_SLICE",
    "JAMMER_TYPES",
    "JSR_INDEX",
    "MASK_RATIO_INDEX",
    "SINR_INDEX",
    "build_channel_state",
    "nominal_channel_state",
    "rule_based_jammer_posterior",
]
