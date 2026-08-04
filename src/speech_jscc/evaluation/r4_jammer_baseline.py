"""Deterministic data-RE jammer construction for the physical R4 evaluator.

The jammer is calibrated on the *attacked data RE subset* of the transmit
resource grid.  It is subsequently OFDM-modulated and passed through the same
multipath realization as the legitimate transmission by ``R4WaveformForward``.
This module deliberately has no allocation policy: it only describes an
exogenous interference grid.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from channels.physical_ofdm import active_grid_masks


def tensor_hash(value: Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class R4Jammer:
    grid: Tensor
    mask: Tensor
    statistics: dict[str, float | str]


def _generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


def _interval(length: int, fraction: float, generator: torch.Generator, device: torch.device) -> slice:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("jammer fraction must be in (0, 1]")
    count = max(1, math.ceil(length * float(fraction)))
    start = int(torch.randint(0, length - count + 1, (), generator=generator, device=device).item())
    return slice(start, start + count)


def _mask(
    data_mask: Tensor,
    jammer_type: str,
    generator: torch.Generator,
    *,
    subband_fraction: float,
    burst_fraction: float,
    tone_count: int,
) -> Tensor:
    if data_mask.ndim != 2:
        raise ValueError("data_mask must be [active_subcarriers, ofdm_symbols]")
    if jammer_type == "no_jammer":
        return torch.zeros_like(data_mask)
    if jammer_type == "broadband_awgn":
        return data_mask.clone()
    active, symbols = data_mask.shape
    selected = torch.zeros_like(data_mask)
    if jammer_type == "subband":
        selected[_interval(active, subband_fraction, generator, data_mask.device), :] = True
    elif jammer_type == "burst":
        selected[:, _interval(symbols, burst_fraction, generator, data_mask.device)] = True
    elif jammer_type == "block":
        selected[_interval(active, subband_fraction, generator, data_mask.device), _interval(symbols, burst_fraction, generator, data_mask.device)] = True
    elif jammer_type == "tone":
        if tone_count <= 0:
            raise ValueError("tone_count must be positive")
        count = min(int(tone_count), active)
        tones = torch.randperm(active, generator=generator, device=data_mask.device)[:count]
        selected[tones, :] = True
    else:
        raise ValueError(f"unsupported R4 jammer type: {jammer_type}")
    return selected & data_mask


def build_r4_jammer(
    signal_grid: Tensor,
    data_mask: Tensor,
    *,
    jammer_type: str,
    jsr_db: float | None,
    seed: int,
    subband_fraction: float = 0.25,
    burst_fraction: float = 0.25,
    tone_count: int = 4,
    epsilon: float = 1e-12,
) -> R4Jammer:
    """Create a deterministic jammer grid matching ``signal_grid``.

    ``signal_grid`` is [batch, active_subcarriers, OFDM_symbols].  JSR is
    measured only where the generated data-RE mask is active, preventing a
    partial-band jammer from appearing weaker merely because its zeros are
    included in the denominator.
    """
    if not torch.is_complex(signal_grid) or signal_grid.ndim != 3:
        raise ValueError("signal_grid must be complex [batch, active, time]")
    if tuple(signal_grid.shape[-2:]) != tuple(data_mask.shape):
        raise ValueError("signal grid and data mask shapes differ")
    mask2d = _mask(data_mask.to(device=signal_grid.device, dtype=torch.bool), jammer_type, _generator(signal_grid.device, seed), subband_fraction=subband_fraction, burst_fraction=burst_fraction, tone_count=tone_count)
    mask = mask2d.unsqueeze(0).expand(signal_grid.shape[0], -1, -1)
    if jammer_type == "no_jammer":
        grid = torch.zeros_like(signal_grid)
        jsr = None
    else:
        if jsr_db is None:
            raise ValueError("a numeric jsr_db is required for an active jammer")
        generator = _generator(signal_grid.device, seed + 1)
        real_dtype = signal_grid.real.dtype
        raw = torch.complex(
            torch.randn(signal_grid.shape, device=signal_grid.device, dtype=real_dtype, generator=generator),
            torch.randn(signal_grid.shape, device=signal_grid.device, dtype=real_dtype, generator=generator),
        ) / math.sqrt(2.0)
        raw = raw * mask
        signal_power = signal_grid[mask].abs().square().mean().clamp_min(epsilon)
        raw_power = raw[mask].abs().square().mean().clamp_min(epsilon)
        grid = raw * torch.sqrt(signal_power * (10.0 ** (float(jsr_db) / 10.0)) / raw_power)
        jsr = float(10.0 * torch.log10((grid[mask].abs().square().mean() / signal_power).clamp_min(epsilon)))
    active = int(mask2d.sum())
    active_subcarriers = int(mask2d.any(dim=1).sum())
    active_times = int(mask2d.any(dim=0).sum())
    return R4Jammer(
        grid=grid,
        mask=mask,
        statistics={
            "jammer_type": jammer_type,
            "target_jsr_db": None if jsr_db is None else float(jsr_db),
            "measured_pre_channel_jsr_db": jsr,
            "active_re_fraction": active / float(data_mask.numel()),
            "active_subcarrier_fraction": active_subcarriers / float(data_mask.shape[0]),
            "active_time_fraction": active_times / float(data_mask.shape[1]),
            "jammer_mask_hash": tensor_hash(mask2d),
            "jammer_tensor_hash": tensor_hash(grid),
        },
    )


def repetition_overlap_diagnostics(allocation, jammer_mask: Tensor) -> dict:
    """Summarize exact 0/1/2/3-jammed-copy exposure in source-symbol order."""
    if jammer_mask.ndim != 3 or jammer_mask.shape[0] != 1:
        raise ValueError("jammer_mask must be [1, active_subcarriers, ofdm_symbols]")
    source_order = allocation.extract_source_order(jammer_mask.to(torch.complex64))[0]
    copy_count = source_order.abs().gt(0).sum(0).to(torch.long)
    histogram = {str(count): int((copy_count == count).sum()) for count in range(4)}
    layer_histograms = {
        str(layer): {str(count): int((copy_count[layer * 240:(layer + 1) * 240] == count).sum()) for count in range(4)}
        for layer in range(8)
    }
    candidates = active_grid_masks(allocation.profile, device=jammer_mask.device).candidate_data.nonzero()
    coordinates = candidates[allocation.source_to_candidate_indices.to(jammer_mask.device)]
    # Pairwise separations quantify the existing geometric diversity; no
    # jammer-specific allocation policy is introduced here.
    frequency = coordinates[..., 0].to(torch.float32)
    time = coordinates[..., 1].to(torch.float32)
    pair_frequency = torch.stack([abs(frequency[a] - frequency[b]) for a, b in ((0, 1), (0, 2), (1, 2))])
    pair_time = torch.stack([abs(time[a] - time[b]) for a, b in ((0, 1), (0, 2), (1, 2))])
    return {
        "source_symbols": 1920,
        "copy_count_histogram": histogram,
        "copy_count_fraction": {key: value / 1920.0 for key, value in histogram.items()},
        "copy_count_ge_2_fraction": (histogram["2"] + histogram["3"]) / 1920.0,
        "per_layer_copy_count_histogram": layer_histograms,
        "pairwise_frequency_separation": {"mean": float(pair_frequency.mean()), "minimum": float(pair_frequency.min()), "p5": float(torch.quantile(pair_frequency, 0.05))},
        "pairwise_time_separation": {"mean": float(pair_time.mean()), "minimum": float(pair_time.min()), "p5": float(torch.quantile(pair_time, 0.05))},
        "source_copy_count": [int(value) for value in copy_count.cpu()],
    }


__all__ = ["R4Jammer", "build_r4_jammer", "repetition_overlap_diagnostics", "tensor_hash"]
