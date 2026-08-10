from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PhysicalOFDMProfile:
    name: str
    n_fft: int
    active_subcarriers: int
    subcarrier_spacing_hz: float
    n_ofdm_symbols: int
    cp_samples: int
    pilot_symbol_indices: tuple[int, int] = (3, 17)

    def __post_init__(self) -> None:
        if self.active_subcarriers % 2 or self.active_subcarriers >= self.n_fft:
            raise ValueError("active_subcarriers must be even and smaller than n_fft")
        if max(self.pilot_symbol_indices) >= self.n_ofdm_symbols:
            raise ValueError("pilot symbol is outside the TTI")

    @property
    def sample_rate_hz(self) -> float:
        return self.n_fft * self.subcarrier_spacing_hz

    @property
    def occupied_bandwidth_hz(self) -> float:
        return self.active_subcarriers * self.subcarrier_spacing_hz

    @property
    def useful_symbol_duration_s(self) -> float:
        return self.n_fft / self.sample_rate_hz

    @property
    def cp_duration_s(self) -> float:
        return self.cp_samples / self.sample_rate_hz

    @property
    def ofdm_symbol_duration_s(self) -> float:
        return (self.n_fft + self.cp_samples) / self.sample_rate_hz

    @property
    def tti_duration_s(self) -> float:
        return self.n_ofdm_symbols * self.ofdm_symbol_duration_s

    @property
    def total_active_re(self) -> int:
        return self.active_subcarriers * self.n_ofdm_symbols

    @property
    def pilot_re(self) -> int:
        return self.active_subcarriers

    @property
    def candidate_data_re(self) -> int:
        return self.total_active_re - self.pilot_re

    @property
    def active_fft_bins(self) -> tuple[int, ...]:
        half = self.active_subcarriers // 2
        return tuple(range(self.n_fft - half, self.n_fft)) + tuple(range(1, half + 1))

    @property
    def guard_fft_bins(self) -> tuple[int, ...]:
        active = set(self.active_fft_bins)
        return tuple(index for index in range(self.n_fft) if index not in active and index != 0)


@dataclass(frozen=True)
class ActiveGridMasks:
    pilot: Tensor
    candidate_data: Tensor


@dataclass(frozen=True)
class LegacyAbstractProfile:
    name: str = "legacy_64x32"
    grid_shape: tuple[int, int] = (64, 32)
    pilot_re: int = 128
    candidate_data_re: int = 1920
    is_physical: bool = False

    @property
    def total_re(self) -> int:
        return self.grid_shape[0] * self.grid_shape[1]


NR_LIKE_R2 = PhysicalOFDMProfile("nr_like_r2", 256, 144, 30_000.0, 28, 18)
NR_LIKE_R3 = PhysicalOFDMProfile("nr_like_r3", 256, 216, 30_000.0, 28, 18)
NR_LIKE_R4 = PhysicalOFDMProfile("nr_like_r4", 512, 360, 30_000.0, 28, 36)
LEGACY_64X32 = LegacyAbstractProfile()


def active_grid_masks(profile: PhysicalOFDMProfile, *, device=None) -> ActiveGridMasks:
    pilot = torch.zeros(
        profile.active_subcarriers, profile.n_ofdm_symbols, dtype=torch.bool, device=device
    )
    first, second = profile.pilot_symbol_indices
    pilot[0::2, first] = True
    pilot[1::2, second] = True
    return ActiveGridMasks(pilot=pilot, candidate_data=~pilot)


def insert_physical_pilots(
    active_grid: Tensor,
    profile: PhysicalOFDMProfile,
    pilot_value: complex = 1 + 0j,
) -> tuple[Tensor, Tensor]:
    if not active_grid.is_complex() or tuple(active_grid.shape[-2:]) != (
        profile.active_subcarriers,
        profile.n_ofdm_symbols,
    ):
        raise ValueError("active_grid shape does not match profile")
    mask = active_grid_masks(profile, device=active_grid.device).pilot
    pilots = torch.zeros_like(active_grid)
    pilots[..., mask] = torch.as_tensor(
        pilot_value, dtype=active_grid.dtype, device=active_grid.device
    )
    return torch.where(mask, pilots, active_grid), pilots


def estimate_comb_dft_ls(
    received_grid: Tensor,
    pilots: Tensor,
    profile: PhysicalOFDMProfile,
    *,
    num_taps: int,
    tap_delay_samples: tuple[int, ...] | None = None,
    ridge_lambda: float = 1e-6,
) -> Tensor:
    if received_grid.shape != pilots.shape or received_grid.ndim != 3:
        raise ValueError("received_grid and pilots must match [B,active,time]")
    if not 0 < num_taps < profile.cp_samples:
        raise ValueError("estimator taps must be positive and shorter than CP")
    mask = active_grid_masks(profile, device=received_grid.device).pilot
    active_bins = torch.tensor(
        profile.active_fft_bins, device=received_grid.device, dtype=received_grid.real.dtype
    )
    pilot_coordinates = mask.nonzero()
    pilot_active_indices = pilot_coordinates[:, 0]
    pilot_symbols = pilot_coordinates[:, 1]
    observations = (
        received_grid[:, pilot_active_indices, pilot_symbols]
        / pilots[:, pilot_active_indices, pilot_symbols]
    )
    if tap_delay_samples is None:
        tap_delay_samples = tuple(range(num_taps))
    if len(tap_delay_samples) != num_taps or max(tap_delay_samples) >= profile.cp_samples:
        raise ValueError("tap delay samples must match taps and remain inside CP")
    delays = torch.tensor(
        tap_delay_samples, device=received_grid.device, dtype=received_grid.real.dtype
    )
    phase = (
        -2j
        * torch.pi
        * active_bins[pilot_active_indices, None]
        * delays[None, :]
        / profile.n_fft
    )
    design = torch.exp(phase).to(received_grid.dtype)
    gram = design.conj().T @ design
    if ridge_lambda:
        gram = gram + float(ridge_lambda) * torch.eye(
            num_taps, device=gram.device, dtype=gram.dtype
        )
    rhs = design.conj().T @ observations.T
    taps = torch.linalg.solve(gram, rhs).T
    full_design = torch.exp(
        -2j * torch.pi * active_bins[:, None] * delays[None, :] / profile.n_fft
    ).to(received_grid.dtype)
    active_response = taps @ full_design.T
    return active_response[..., None].expand_as(received_grid)


def modulate_tti(active_grid: Tensor, profile: PhysicalOFDMProfile) -> Tensor:
    expected = (profile.active_subcarriers, profile.n_ofdm_symbols)
    if not active_grid.is_complex() or tuple(active_grid.shape[-2:]) != expected:
        raise ValueError(f"active_grid must end in {expected}")
    batch_shape = active_grid.shape[:-2]
    fft_grid = torch.zeros(
        *batch_shape,
        profile.n_ofdm_symbols,
        profile.n_fft,
        dtype=active_grid.dtype,
        device=active_grid.device,
    )
    active = active_grid.transpose(-2, -1)
    fft_grid[..., list(profile.active_fft_bins)] = active
    useful = torch.fft.ifft(fft_grid, dim=-1, norm="ortho")
    with_cp = torch.cat((useful[..., -profile.cp_samples :], useful), dim=-1)
    return with_cp.reshape(*batch_shape, -1)


def demodulate_tti(waveform: Tensor, profile: PhysicalOFDMProfile) -> Tensor:
    samples_per_symbol = profile.n_fft + profile.cp_samples
    expected = profile.n_ofdm_symbols * samples_per_symbol
    if not waveform.is_complex() or waveform.shape[-1] != expected:
        raise ValueError(f"waveform must contain exactly {expected} complex samples")
    symbols = waveform.reshape(*waveform.shape[:-1], profile.n_ofdm_symbols, samples_per_symbol)
    useful = symbols[..., profile.cp_samples :]
    fft_grid = torch.fft.fft(useful, dim=-1, norm="ortho")
    return fft_grid[..., list(profile.active_fft_bins)].transpose(-2, -1)


def apply_tti_multipath(
    waveform: Tensor, taps: Tensor, profile: PhysicalOFDMProfile
) -> Tensor:
    if not waveform.is_complex() or not taps.is_complex():
        raise TypeError("waveform and taps must be complex")
    if taps.ndim != 2 or waveform.ndim != 2 or taps.shape[0] != waveform.shape[0]:
        raise ValueError("waveform and taps must be [B,samples] and [B,taps]")
    max_delay = taps.shape[-1] - 1
    if max_delay >= profile.cp_samples:
        raise ValueError("max_channel_delay_samples must be smaller than cp_samples")
    convolution_length = waveform.shape[-1] + taps.shape[-1] - 1
    signal_fft = torch.fft.fft(waveform, n=convolution_length, dim=-1)
    taps_fft = torch.fft.fft(taps, n=convolution_length, dim=-1)
    convolved = torch.fft.ifft(signal_fft * taps_fft, n=convolution_length, dim=-1)
    return convolved[..., : waveform.shape[-1]]


__all__ = [
    "ActiveGridMasks",
    "LEGACY_64X32",
    "LegacyAbstractProfile",
    "NR_LIKE_R2",
    "NR_LIKE_R3",
    "NR_LIKE_R4",
    "PhysicalOFDMProfile",
    "active_grid_masks",
    "apply_tti_multipath",
    "demodulate_tti",
    "estimate_comb_dft_ls",
    "insert_physical_pilots",
    "modulate_tti",
]
