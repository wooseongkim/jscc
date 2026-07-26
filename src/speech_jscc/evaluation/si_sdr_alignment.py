"""Diagnostics for waveform alignment effects on SI-SDR.

The production metric is intentionally not changed by this module.  It reports
the existing full-crop SI-SDR and a diagnostic score after selecting a small
cross-correlation lag window.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from speech_jscc.metrics.audio_quality import compute_si_sdr


@dataclass(frozen=True)
class AlignmentResult:
    shift_samples: int
    overlap_samples: int
    correlation: float
    si_sdr_db: float


def _segments(reference: Tensor, estimate: Tensor, shift: int) -> tuple[Tensor, Tensor]:
    length = min(reference.shape[-1], estimate.shape[-1])
    if shift >= 0:
        if shift >= length:
            raise ValueError("alignment shift leaves no overlap")
        return reference[..., shift:length], estimate[..., : length - shift]
    offset = -shift
    if offset >= length:
        raise ValueError("alignment shift leaves no overlap")
    return reference[..., : length - offset], estimate[..., offset:length]


def best_cross_correlation_alignment(
    reference: Tensor,
    estimate: Tensor,
    *,
    sample_rate: int,
    max_lag_ms: float = 5.0,
) -> AlignmentResult:
    """Find the positive-correlation lag and score the resulting overlap.

    A positive shift means that the reference is advanced relative to the
    estimate under the segment convention above.  This sign is only
    diagnostic; callers should rely on the reported magnitude and score.
    """
    if sample_rate <= 0 or max_lag_ms < 0:
        raise ValueError("sample_rate must be positive and max_lag_ms nonnegative")
    if reference.ndim == 1:
        reference = reference.unsqueeze(0)
    if estimate.ndim == 1:
        estimate = estimate.unsqueeze(0)
    if reference.ndim != 2 or estimate.ndim != 2 or reference.shape[0] != estimate.shape[0]:
        raise ValueError("waveforms must have matching [B,S] shapes")
    length = min(reference.shape[-1], estimate.shape[-1])
    if length <= 1:
        raise ValueError("waveforms must contain at least two samples")
    max_lag = min(int(round(float(max_lag_ms) * sample_rate / 1000.0)), length - 1)
    best: tuple[float, int, float] | None = None
    for shift in range(-max_lag, max_lag + 1):
        ref_seg, est_seg = _segments(reference, estimate, shift)
        ref_zero = ref_seg - ref_seg.mean(dim=-1, keepdim=True)
        est_zero = est_seg - est_seg.mean(dim=-1, keepdim=True)
        denom = (
            ref_zero.square().sum(dim=-1).sqrt()
            * est_zero.square().sum(dim=-1).sqrt()
        ).clamp_min(1e-12)
        corr = (ref_zero * est_zero).sum(dim=-1) / denom
        score = float(corr.mean().detach().cpu())
        if best is None or score > best[0]:
            best = (score, shift, float(compute_si_sdr(ref_seg, est_seg).mean().detach().cpu()))
    assert best is not None
    return AlignmentResult(
        shift_samples=int(best[1]),
        overlap_samples=int(length - abs(best[1])),
        correlation=float(best[0]),
        si_sdr_db=float(best[2]),
    )


def active_speech_fraction(reference: Tensor, threshold_ratio: float = 0.01) -> float:
    """Report the fraction above a conservative RMS-relative activity threshold."""
    if threshold_ratio <= 0:
        raise ValueError("threshold_ratio must be positive")
    value = reference.detach().float()
    if value.ndim > 1:
        value = value.reshape(-1)
    threshold = float(value.abs().max().cpu()) * float(threshold_ratio)
    return float((value.abs() > threshold).float().mean().cpu())
