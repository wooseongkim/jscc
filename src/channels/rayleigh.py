from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from channels.multipath import multipath_block_fading


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.complex128 else torch.float32


def _complex_normal(
    shape: tuple[int, ...],
    reference: Tensor,
    generator: torch.Generator | None = None,
) -> Tensor:
    real = torch.randn(
        shape,
        device=reference.device,
        dtype=_real_dtype(reference.dtype),
        generator=generator,
    )
    imag = torch.randn(
        shape,
        device=reference.device,
        dtype=_real_dtype(reference.dtype),
        generator=generator,
    )
    return torch.complex(real, imag) / math.sqrt(2.0)


def _batch_parameter(value: float | Tensor, reference: Tensor) -> Tensor:
    result = torch.as_tensor(value, device=reference.device, dtype=_real_dtype(reference.dtype))
    if result.ndim == 0:
        result = result.repeat(reference.shape[0])
    if result.shape != (reference.shape[0],):
        raise ValueError(f"expected a scalar or one value per batch item, got {tuple(result.shape)}")
    return result.reshape(reference.shape[0], *([1] * (reference.ndim - 1)))


def _fading_shape(signal: Tensor, fading: str) -> tuple[int, ...]:
    if fading == "auto":
        fading = "flat" if signal.ndim == 2 else "ofdm"
    if fading == "flat":
        return (signal.shape[0],) + (1,) * (signal.ndim - 1)
    if fading == "ofdm" and signal.ndim == 3:
        return tuple(signal.shape)
    if fading == "multipath_block" and signal.ndim == 3:
        return tuple(signal.shape)
    raise ValueError(
        "fading must be 'auto', 'flat', 'ofdm', or 'multipath_block' "
        "(OFDM modes require [B, K, N])"
    )


def compute_effective_sinr(
    faded_signal: Tensor,
    faded_jammer: Tensor,
    noise: Tensor,
    *,
    db: bool = False,
    eps: float = 1e-12,
) -> Tensor:
    """Compute per-example effective SINR from realized channel components."""
    if faded_signal.shape != faded_jammer.shape or faded_signal.shape != noise.shape:
        raise ValueError("signal, jammer, and noise must have matching shapes")
    if not all(torch.is_complex(value) for value in (faded_signal, faded_jammer, noise)):
        raise ValueError("signal, jammer, and noise must be complex")
    dimensions = tuple(range(1, faded_signal.ndim))
    signal_power = faded_signal.abs().square().mean(dimensions)
    interference_power = faded_jammer.abs().square().mean(dimensions)
    noise_power = noise.abs().square().mean(dimensions)
    ratio = signal_power / (interference_power + noise_power).clamp_min(eps)
    return 10.0 * torch.log10(ratio.clamp_min(eps)) if db else ratio


def post_channel_jsr(
    faded_signal: Tensor,
    faded_jammer: Tensor,
    *,
    db: bool = False,
    eps: float = 1e-12,
) -> Tensor:
    """Return realized post-channel JSR from faded signal and jammer powers."""
    if faded_signal.shape != faded_jammer.shape:
        raise ValueError("faded signal and jammer must have matching shapes")
    dimensions = tuple(range(1, faded_signal.ndim))
    ratio = faded_jammer.abs().square().mean(dimensions) / faded_signal.abs().square().mean(
        dimensions
    ).clamp_min(eps)
    return 10.0 * torch.log10(ratio.clamp_min(eps)) if db else ratio


def rayleigh_channel(
    transmitted: Tensor,
    jammer: Tensor | None = None,
    snr_db: float | Tensor = 10.0,
    *,
    fading: str = "auto",
    signal_fading: Tensor | None = None,
    jammer_fading: Tensor | None = None,
    noise: Tensor | None = None,
    generator: torch.Generator | None = None,
    equalizer_epsilon: float = 1e-6,
    num_taps: int = 6,
    pdp_decay: float = 0.7,
) -> dict[str, Tensor]:
    """Apply Rayleigh fading, optional jamming, and complex AWGN.

    `transmitted` may be flat symbols `[B, M]` or an OFDM grid `[B, K, N]`.
    `fading='flat'` draws one coefficient per example. For OFDM grids,
    `fading='ofdm'` (and the default `auto`) draws one coefficient per resource
    element. Explicit fading tensors may be supplied for deterministic tests.
    """
    if not torch.is_complex(transmitted) or transmitted.ndim not in (2, 3):
        raise ValueError("transmitted must be complex [B, M] or [B, K, N]")
    if jammer is None:
        jammer = torch.zeros_like(transmitted)
    if jammer.shape != transmitted.shape or not torch.is_complex(jammer):
        raise ValueError("jammer must be a matching complex tensor")
    if fading == "auto":
        fading = "flat" if transmitted.ndim == 2 else "ofdm"
    coefficient_shape = _fading_shape(transmitted, fading)
    diagnostics: dict[str, Tensor | str | bool] = {"fading_model": fading}
    if fading == "multipath_block" and (signal_fading is None or jammer_fading is None):
        multipath = multipath_block_fading(
            batch_size=transmitted.shape[0],
            subcarriers=transmitted.shape[1],
            ofdm_symbols=transmitted.shape[2],
            num_taps=num_taps,
            pdp_decay=pdp_decay,
            reference=transmitted,
            generator=generator,
        )
        if signal_fading is None:
            signal_fading = multipath["signal_fading"]
            diagnostics["signal_taps"] = multipath["signal_taps"]
        if jammer_fading is None:
            jammer_fading = multipath["jammer_fading"]
            diagnostics["jammer_taps"] = multipath["jammer_taps"]
        diagnostics["pdp"] = multipath["pdp"]
        diagnostics["block_fading_over_time"] = True
        diagnostics["assume_ideal_cp"] = True
    else:
        if signal_fading is None:
            signal_fading = _complex_normal(coefficient_shape, transmitted, generator)
        if jammer_fading is None:
            jammer_fading = _complex_normal(coefficient_shape, transmitted, generator)
    try:
        torch.broadcast_shapes(tuple(transmitted.shape), tuple(signal_fading.shape))
        torch.broadcast_shapes(tuple(transmitted.shape), tuple(jammer_fading.shape))
    except RuntimeError as error:
        raise ValueError("fading tensors are not broadcastable to transmitted") from error

    dimensions = tuple(range(1, transmitted.ndim))
    signal_power = transmitted.abs().square().mean(dimensions, keepdim=True)
    requested_noise_power = signal_power / torch.pow(
        10.0, _batch_parameter(snr_db, transmitted) / 10.0
    )
    if noise is None:
        noise = _complex_normal(tuple(transmitted.shape), transmitted, generator) * torch.sqrt(
            requested_noise_power
        )
    elif noise.shape != transmitted.shape or not torch.is_complex(noise):
        raise ValueError("explicit noise must be a matching complex tensor")
    noise_power = noise.abs().square().mean(dimensions, keepdim=True)
    faded_signal = signal_fading * transmitted
    faded_jammer = jammer_fading * jammer
    received = faded_signal + faded_jammer + noise
    denominator = signal_fading.abs().square().clamp_min(equalizer_epsilon)
    equalized = received * signal_fading.conj() / denominator
    effective_sinr = compute_effective_sinr(faded_signal, faded_jammer, noise)
    return {
        "received": received,
        "equalized": equalized,
        "signal_fading": signal_fading,
        "jammer_fading": jammer_fading,
        "faded_signal": faded_signal,
        "faded_jammer": faded_jammer,
        "noise": noise,
        "noise_power": noise_power,
        "effective_sinr": effective_sinr,
        "post_channel_jsr": post_channel_jsr(faded_signal, faded_jammer),
        **diagnostics,
    }


class RayleighChannel(nn.Module):
    """Module wrapper around :func:`rayleigh_channel`."""

    def __init__(
        self,
        fading: str = "auto",
        equalizer_epsilon: float = 1e-6,
        *,
        num_taps: int = 6,
        pdp_decay: float = 0.7,
    ):
        super().__init__()
        self.fading = fading
        self.equalizer_epsilon = equalizer_epsilon
        self.num_taps = int(num_taps)
        self.pdp_decay = float(pdp_decay)

    def forward(
        self,
        transmitted: Tensor,
        jammer: Tensor | None = None,
        snr_db: float | Tensor = 10.0,
    ) -> dict[str, Tensor]:
        return rayleigh_channel(
            transmitted,
            jammer,
            snr_db,
            fading=self.fading,
            equalizer_epsilon=self.equalizer_epsilon,
            num_taps=self.num_taps,
            pdp_decay=self.pdp_decay,
        )
