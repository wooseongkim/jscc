from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _as_batch_waveform(waveform: Tensor, name: str) -> Tensor:
    if not isinstance(waveform, Tensor):
        raise TypeError(f"{name} must be a torch Tensor")
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 2:
        pass
    elif waveform.ndim == 3 and waveform.shape[1] == 1:
        waveform = waveform.squeeze(1)
    else:
        raise ValueError(f"{name} must have shape [S], [B,S], or [B,1,S]")
    if waveform.shape[-1] <= 0:
        raise ValueError(f"{name} must contain at least one sample")
    if not torch.is_floating_point(waveform):
        waveform = waveform.float()
    else:
        waveform = waveform.to(dtype=waveform.dtype)
    if not torch.isfinite(waveform).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return waveform


def align_waveforms(reference: Tensor, estimate: Tensor) -> tuple[Tensor, Tensor]:
    """Return finite mono batched waveforms cropped to the shared length."""
    reference_batch = _as_batch_waveform(reference, "reference")
    estimate_batch = _as_batch_waveform(estimate, "estimate")
    if reference_batch.device != estimate_batch.device:
        raise ValueError("reference and estimate must be on the same device")
    if reference_batch.shape[0] != estimate_batch.shape[0]:
        raise ValueError("reference and estimate batch sizes must match")
    length = min(reference_batch.shape[-1], estimate_batch.shape[-1])
    return reference_batch[..., :length], estimate_batch[..., :length]


def compute_si_sdr(reference: Tensor, estimate: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute per-example scale-invariant SDR in dB."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    reference_batch, estimate_batch = align_waveforms(reference, estimate)
    dtype = torch.promote_types(reference_batch.dtype, estimate_batch.dtype)
    reference_batch = reference_batch.to(dtype=dtype)
    estimate_batch = estimate_batch.to(dtype=dtype)

    reference_zero_mean = reference_batch - reference_batch.mean(dim=-1, keepdim=True)
    estimate_zero_mean = estimate_batch - estimate_batch.mean(dim=-1, keepdim=True)
    reference_energy = reference_zero_mean.square().sum(dim=-1, keepdim=True).clamp_min(eps)
    projection_scale = (
        estimate_zero_mean * reference_zero_mean
    ).sum(dim=-1, keepdim=True) / reference_energy
    projection = projection_scale * reference_zero_mean
    noise = estimate_zero_mean - projection

    signal_power = projection.square().sum(dim=-1).clamp_min(eps)
    noise_power = noise.square().sum(dim=-1).clamp_min(eps)
    ratio = (signal_power + eps) / (noise_power + eps)
    score = 10.0 * torch.log10(ratio.clamp_min(eps))
    return torch.nan_to_num(score, nan=0.0, posinf=300.0, neginf=-300.0)


def compute_stoi(
    reference: Tensor,
    estimate: Tensor,
    sample_rate: int,
    extended: bool = False,
) -> Tensor | None:
    """Compute per-example STOI if pystoi is installed."""
    try:
        from pystoi.stoi import stoi
    except ImportError:
        return None

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    reference_batch, estimate_batch = align_waveforms(reference, estimate)
    values: list[float] = []
    for ref, est in zip(reference_batch, estimate_batch, strict=True):
        ref_np = ref.detach().clamp(-1.0, 1.0).cpu().numpy().astype("float64", copy=False)
        est_np = est.detach().clamp(-1.0, 1.0).cpu().numpy().astype("float64", copy=False)
        values.append(float(stoi(ref_np, est_np, int(sample_rate), extended=extended)))
    return torch.tensor(values, dtype=reference_batch.dtype, device=reference_batch.device)


def _float_list(values: Tensor) -> list[float]:
    return [float(value) for value in values.detach().cpu().tolist()]


def summarize_audio_metrics(
    reference: Tensor,
    estimate: Tensor,
    sample_rate: int,
    enable_stoi: bool = True,
) -> dict[str, Any]:
    si_sdr = compute_si_sdr(reference, estimate)
    summary: dict[str, Any] = {
        "si_sdr_db": float(si_sdr.detach().mean().cpu().item()),
        "si_sdr_db_per_example": _float_list(si_sdr),
        "stoi": None,
        "stoi_per_example": None,
        "stoi_available": False,
        "stoi_error": None,
    }
    if not enable_stoi:
        return summary
    try:
        stoi_values = compute_stoi(reference, estimate, sample_rate)
    except Exception as error:  # pystoi can reject too-short or degenerate signals.
        summary["stoi_error"] = str(error)
        return summary
    if stoi_values is None:
        summary["stoi_error"] = "pystoi is not installed"
        return summary
    summary["stoi"] = float(stoi_values.detach().mean().cpu().item())
    summary["stoi_per_example"] = _float_list(stoi_values)
    summary["stoi_available"] = True
    return summary
