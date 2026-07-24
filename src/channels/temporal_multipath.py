from __future__ import annotations

import math

import torch
from torch import Tensor


# The simulator convention uses the standard engineering approximation.
SPEED_OF_LIGHT_MPS = 3.0e8


def doppler_frequency_hz(user_speed_mps: float, carrier_frequency_hz: float) -> float:
    if user_speed_mps < 0 or carrier_frequency_hz <= 0:
        raise ValueError("speed must be nonnegative and carrier frequency positive")
    return float(user_speed_mps) * float(carrier_frequency_hz) / SPEED_OF_LIGHT_MPS


def jakes_slot_correlation(
    user_speed_mps: float, carrier_frequency_hz: float, slot_duration_s: float
) -> float:
    if slot_duration_s <= 0:
        raise ValueError("slot_duration_s must be positive")
    argument = 2.0 * math.pi * doppler_frequency_hz(
        user_speed_mps, carrier_frequency_hz
    ) * float(slot_duration_s)
    return float(torch.special.bessel_j0(torch.tensor(argument, dtype=torch.float64)))


def _proper_complex(
    shape: tuple[int, ...], pdp: Tensor, generator: torch.Generator
) -> Tensor:
    real_dtype = pdp.dtype
    real = torch.randn(shape, generator=generator, dtype=real_dtype, device=pdp.device)
    imag = torch.randn(shape, generator=generator, dtype=real_dtype, device=pdp.device)
    return torch.complex(real, imag) * torch.sqrt(pdp)[None, :] / math.sqrt(2.0)


def iid_tap_trajectory(
    *, slots: int, batch_size: int, pdp: Tensor, seed: int
) -> Tensor:
    _validate(slots, batch_size, pdp)
    generator = torch.Generator(device=pdp.device).manual_seed(int(seed))
    return torch.stack(
        [_proper_complex((batch_size, pdp.numel()), pdp, generator) for _ in range(slots)]
    )


def correlated_tap_trajectory(
    *, slots: int, batch_size: int, pdp: Tensor, rho: float, seed: int
) -> Tensor:
    _validate(slots, batch_size, pdp)
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must be in [-1, 1]")
    generator = torch.Generator(device=pdp.device).manual_seed(int(seed))
    current = _proper_complex((batch_size, pdp.numel()), pdp, generator)
    trajectory = [current]
    innovation_scale = math.sqrt(max(0.0, 1.0 - float(rho) ** 2))
    for _ in range(1, slots):
        innovation = _proper_complex((batch_size, pdp.numel()), pdp, generator)
        current = float(rho) * current + innovation_scale * innovation
        trajectory.append(current)
    return torch.stack(trajectory)


def _validate(slots: int, batch_size: int, pdp: Tensor) -> None:
    if slots < 1 or batch_size < 1:
        raise ValueError("slots and batch_size must be positive")
    if pdp.ndim != 1 or pdp.numel() < 1 or torch.any(pdp < 0):
        raise ValueError("pdp must be a nonnegative vector")
    if not torch.isfinite(pdp).all() or not torch.isclose(
        pdp.sum(), torch.ones((), dtype=pdp.dtype, device=pdp.device)
    ):
        raise ValueError("pdp must be finite and normalized")


def measured_lag1_correlation(taps: Tensor) -> float:
    if taps.ndim != 3 or taps.shape[0] < 2:
        raise ValueError("taps must have shape [slots,batch,taps] with slots >= 2")
    previous, current = taps[:-1].reshape(-1), taps[1:].reshape(-1)
    numerator = (previous.conj() * current).mean().real
    denominator = torch.sqrt(previous.abs().square().mean() * current.abs().square().mean())
    return float(numerator / denominator)


def taps_to_slot_frequency_response(
    taps: Tensor, *, subcarriers: int, ofdm_symbols: int
) -> Tensor:
    if not taps.is_complex() or taps.ndim != 3:
        raise ValueError("taps must be complex [slots,batch,taps]")
    if subcarriers < taps.shape[-1] or ofdm_symbols < 1:
        raise ValueError("invalid OFDM dimensions")
    response = torch.fft.fft(taps, n=int(subcarriers), dim=-1)
    return response[..., None].expand(*response.shape, int(ofdm_symbols))


__all__ = [
    "correlated_tap_trajectory",
    "doppler_frequency_hz",
    "iid_tap_trajectory",
    "jakes_slot_correlation",
    "measured_lag1_correlation",
    "taps_to_slot_frequency_response",
]
