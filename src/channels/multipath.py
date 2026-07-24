from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.complex128 else torch.float32


def exponential_pdp(
    num_taps: int,
    pdp_decay: float = 0.7,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return normalized exponential tap powers with unit expected total power."""
    if int(num_taps) < 1:
        raise ValueError("num_taps must be at least 1")
    if not 0.0 < float(pdp_decay) <= 1.0:
        raise ValueError("pdp_decay must satisfy 0 < pdp_decay <= 1")
    powers = torch.pow(
        torch.as_tensor(float(pdp_decay), device=device, dtype=dtype),
        torch.arange(int(num_taps), device=device, dtype=dtype),
    )
    if not torch.isfinite(powers).all() or torch.any(powers < 0):
        raise ValueError("PDP must be finite and nonnegative")
    total = powers.sum()
    if not torch.isfinite(total) or total <= 0:
        raise ValueError("PDP must have positive finite sum")
    return powers / total


def sample_tdl_taps(
    batch_size: int,
    pdp: Tensor,
    *,
    reference: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample complex Gaussian TDL taps with per-tap variance equal to `pdp`."""
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if pdp.ndim != 1 or pdp.numel() < 1:
        raise ValueError("pdp must have shape [num_taps]")
    if not torch.isfinite(pdp).all() or torch.any(pdp < 0) or pdp.sum() <= 0:
        raise ValueError("pdp must be finite, nonnegative, and have positive sum")
    if reference is not None:
        device = reference.device
        complex_dtype = reference.dtype if reference.is_complex() else torch.complex64
        real_dtype = _real_dtype(complex_dtype)
    else:
        device = pdp.device
        real_dtype = pdp.dtype if pdp.is_floating_point() else torch.float32
        complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
    powers = (pdp / pdp.sum()).to(device=device, dtype=real_dtype)
    shape = (int(batch_size), int(powers.numel()))
    real = torch.randn(shape, device=device, dtype=real_dtype, generator=generator)
    imag = torch.randn(shape, device=device, dtype=real_dtype, generator=generator)
    taps = torch.complex(real, imag) / math.sqrt(2.0)
    return (taps * torch.sqrt(powers)[None, :]).to(complex_dtype)


def taps_to_ofdm_response(taps: Tensor, subcarriers: int, ofdm_symbols: int) -> Tensor:
    """FFT TDL taps into a block-fading OFDM response `[B,K,N]`.

    This assumes an ideal cyclic prefix long enough to remove ISI. The channel
    is modeled directly in the frequency domain; no time-domain OFDM
    modulation, CP overhead, Doppler, or ICI is included.
    """
    if not torch.is_complex(taps) or taps.ndim != 2:
        raise ValueError("taps must be complex [B,num_taps]")
    if int(subcarriers) <= 0 or int(ofdm_symbols) <= 0:
        raise ValueError("subcarriers and ofdm_symbols must be positive")
    if taps.shape[1] > int(subcarriers):
        raise ValueError("num_taps must not exceed the number of subcarriers")
    response = torch.fft.fft(taps, n=int(subcarriers), dim=1)
    return response[:, :, None].expand(taps.shape[0], int(subcarriers), int(ofdm_symbols))


def multipath_block_fading(
    *,
    batch_size: int,
    subcarriers: int,
    ofdm_symbols: int,
    num_taps: int = 6,
    pdp_decay: float = 0.7,
    reference: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Generate independent signal/jammer TDL channels and OFDM responses."""
    if int(num_taps) > int(subcarriers):
        raise ValueError("num_taps must not exceed the number of subcarriers")
    dtype = _real_dtype(reference.dtype) if reference is not None and reference.is_complex() else torch.float32
    device = reference.device if reference is not None else None
    pdp = exponential_pdp(num_taps, pdp_decay, device=device, dtype=dtype)
    signal_taps = sample_tdl_taps(batch_size, pdp, reference=reference, generator=generator)
    jammer_taps = sample_tdl_taps(batch_size, pdp, reference=reference, generator=generator)
    return {
        "signal_taps": signal_taps,
        "jammer_taps": jammer_taps,
        "signal_fading": taps_to_ofdm_response(signal_taps, subcarriers, ofdm_symbols),
        "jammer_fading": taps_to_ofdm_response(jammer_taps, subcarriers, ofdm_symbols),
        "pdp": pdp,
        "fading_model": "multipath_block",
        "block_fading_over_time": True,
        "assume_ideal_cp": True,
    }


__all__ = [
    "exponential_pdp",
    "multipath_block_fading",
    "sample_tdl_taps",
    "taps_to_ofdm_response",
]
