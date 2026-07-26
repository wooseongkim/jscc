from __future__ import annotations

import math

import pytest
import torch

from channels.physical_ofdm import (
    LEGACY_64X32,
    NR_LIKE_R2,
    NR_LIKE_R3,
    active_grid_masks,
    apply_tti_multipath,
    demodulate_tti,
    estimate_comb_dft_ls,
    insert_physical_pilots,
    modulate_tti,
)
from speech_jscc.diagnostics.physical_fdd import (
    PhysicalCSIReport,
    allocate_physical_resources,
    bounded_power_weights,
    recover_source_symbols,
)


@pytest.mark.parametrize(
    ("profile", "active", "pilots", "candidate"),
    [(NR_LIKE_R2, 144, 144, 3888), (NR_LIKE_R3, 216, 216, 5832)],
)
def test_profile_timing_bins_and_resource_counts(profile, active, pilots, candidate):
    assert profile.sample_rate_hz == 7_680_000
    assert profile.useful_symbol_duration_s == pytest.approx(1 / 30_000)
    assert profile.cp_duration_s == pytest.approx(18 / 7_680_000)
    assert profile.tti_duration_s == pytest.approx(28 * 274 / 7_680_000)
    assert profile.tti_duration_s == pytest.approx(0.001, abs=2e-6)
    assert len(profile.active_fft_bins) == active
    assert 0 not in profile.active_fft_bins
    assert len(set(profile.active_fft_bins)) == active
    assert all(0 <= index < profile.n_fft for index in profile.active_fft_bins)
    masks = active_grid_masks(profile)
    assert int(masks.pilot.sum()) == pilots
    assert int(masks.candidate_data.sum()) == candidate
    assert not torch.any(masks.pilot & masks.candidate_data)
    assert torch.all(masks.pilot | masks.candidate_data)


def test_legacy_profile_is_retained_only_as_abstract_regression_metadata():
    assert LEGACY_64X32.name == "legacy_64x32"
    assert LEGACY_64X32.total_re == 2048
    assert LEGACY_64X32.pilot_re == 128
    assert LEGACY_64X32.candidate_data_re == 1920
    assert LEGACY_64X32.is_physical is False


def test_no_channel_ofdm_round_trip_and_cp_safety():
    generator = torch.Generator().manual_seed(7)
    grid = torch.complex(
        torch.randn(2, NR_LIKE_R3.active_subcarriers, 28, generator=generator),
        torch.randn(2, NR_LIKE_R3.active_subcarriers, 28, generator=generator),
    )
    waveform = modulate_tti(grid, NR_LIKE_R3)
    recovered = demodulate_tti(waveform, NR_LIKE_R3)
    assert float((recovered - grid).abs().max()) <= 1e-6
    taps = torch.tensor([[1 + 0j, .2 + .1j, 0, 0, .05 - .1j, 0]], dtype=torch.complex64)
    assert taps.shape[-1] - 1 < NR_LIKE_R3.cp_samples
    received = apply_tti_multipath(waveform[:1], taps, NR_LIKE_R3)
    recovered_channel = demodulate_tti(received, NR_LIKE_R3)
    h = torch.fft.fft(taps, n=NR_LIKE_R3.n_fft)[
        :, torch.tensor(NR_LIKE_R3.active_fft_bins)
    ]
    expected = grid[:1] * h[..., None]
    assert float((recovered_channel - expected).abs().max()) <= 2e-5


@pytest.mark.parametrize("profile", [NR_LIKE_R2, NR_LIKE_R3])
def test_comb_pilots_cover_opposite_parities_and_dft_ls_recovers_six_taps(profile):
    grid = torch.zeros(1, profile.active_subcarriers, 28, dtype=torch.complex64)
    transmitted, pilots = insert_physical_pilots(grid, profile)
    masks = active_grid_masks(profile)
    assert torch.all(transmitted[masks.pilot[None]] == 1)
    assert torch.count_nonzero(pilots).item() == profile.pilot_re
    taps = torch.tensor([[1 + 0j, .2 + .1j, -.1j, .03, 0, .01j]], dtype=torch.complex64)
    h_fft = torch.fft.fft(taps, n=profile.n_fft)
    h_active = h_fft[:, torch.tensor(profile.active_fft_bins)]
    received = transmitted * h_active[..., None]
    estimate = estimate_comb_dft_ls(
        received, pilots, profile, num_taps=6, ridge_lambda=0.0
    )
    assert estimate.shape == received.shape
    assert float((estimate - h_active[..., None]).abs().max()) <= 2e-5


@pytest.mark.parametrize("profile", [NR_LIKE_R2, NR_LIKE_R3])
def test_physical_selection_is_bijective_zero_fills_unused_and_is_invertible(profile):
    report = PhysicalCSIReport.from_reliability(
        generated_tti=0,
        reliability=torch.ones(profile.candidate_data_re),
    )
    allocation = allocate_physical_resources(
        profile=profile,
        tx_tti=1,
        report=report,
        layer_importance_order=[1, 0, 2, 5, 3, 4, 6, 7],
    )
    assert allocation.selected_candidate_indices.numel() == 1920
    assert torch.unique(allocation.selected_candidate_indices).numel() == 1920
    assert sorted(allocation.resource_to_source.tolist()) == list(range(1920))
    source = torch.randn(2, 1920, dtype=torch.complex64)
    grid = allocation.place(source)
    masks = active_grid_masks(profile)
    assert torch.count_nonzero(grid[:, masks.candidate_data]).item() == 2 * 1920
    selected = torch.zeros(profile.candidate_data_re, dtype=torch.bool)
    selected[allocation.selected_candidate_indices] = True
    assert torch.count_nonzero(grid[:, masks.candidate_data][:, ~selected]).item() == 0
    restored = recover_source_symbols(grid, allocation)
    assert torch.equal(restored, source)


def test_tti_zero_is_deterministic_and_current_csi_cannot_enter_delayed_allocator():
    first = allocate_physical_resources(
        profile=NR_LIKE_R3,
        tx_tti=0,
        report=None,
        layer_importance_order=[1, 0, 2, 5, 3, 4, 6, 7],
    )
    second = allocate_physical_resources(
        profile=NR_LIKE_R3,
        tx_tti=0,
        report=None,
        layer_importance_order=[1, 0, 2, 5, 3, 4, 6, 7],
    )
    assert torch.equal(first.selected_candidate_indices, second.selected_candidate_indices)
    current = PhysicalCSIReport(
        generated_tti=1,
        available_tti=1,
        reliability=torch.ones(NR_LIKE_R3.candidate_data_re),
    )
    with pytest.raises(ValueError, match="past CSI"):
        allocate_physical_resources(
            profile=NR_LIKE_R3,
            tx_tti=1,
            report=current,
            layer_importance_order=[1, 0, 2, 5, 3, 4, 6, 7],
        )


def test_bounded_power_obeys_limits_and_fixed_energy():
    reliability = torch.logspace(-5, 5, 1920)
    weights = bounded_power_weights(
        reliability, alpha=.5, minimum_relative_power=.5, maximum_relative_power=2.0
    )
    assert float(weights.min()) >= .5 - 1e-6
    assert float(weights.max()) <= 2.0 + 1e-6
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-6)
    source = torch.randn(4, 1920, dtype=torch.complex64)
    source = source / source.abs().square().mean(dim=1, keepdim=True).sqrt()
    powered = source * weights.sqrt()
    assert float(powered.abs().square().mean()) == pytest.approx(1.0, abs=.03)


def test_fixed_expected_data_energy_is_profile_invariant_and_receiver_reproduces_map():
    energies = [float(torch.ones(1920).sum())]
    for profile in (NR_LIKE_R2, NR_LIKE_R3):
        report = PhysicalCSIReport.from_reliability(
            0, torch.rand(profile.candidate_data_re)
        )
        tx = allocate_physical_resources(
            profile=profile, tx_tti=1, report=report,
            layer_importance_order=[1, 0, 2, 5, 3, 4, 6, 7],
        )
        rx = allocate_physical_resources(
            profile=profile, tx_tti=1, report=report,
            layer_importance_order=[1, 0, 2, 5, 3, 4, 6, 7],
        )
        assert torch.equal(tx.selected_candidate_indices, rx.selected_candidate_indices)
        assert torch.equal(tx.resource_to_source, rx.resource_to_source)
        assert torch.equal(tx.relative_power, rx.relative_power)
        energies.append(float(tx.relative_power.sum()))
    assert energies == pytest.approx([1920.0] * 3, abs=2e-4)
