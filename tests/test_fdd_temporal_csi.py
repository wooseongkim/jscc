from __future__ import annotations

import math

import pytest
import torch

from channels.temporal_multipath import (
    correlated_tap_trajectory,
    doppler_frequency_hz,
    iid_tap_trajectory,
    jakes_slot_correlation,
    measured_lag1_correlation,
    taps_to_slot_frequency_response,
)
from speech_jscc.diagnostics.fdd_temporal_csi import (
    CSIReport,
    DelayedCSIBuffer,
    allocate_from_current_oracle,
    allocate_from_report,
    apply_resource_map,
    deterministic_interleaver,
    invert_resource_map,
    mmse_equalize,
)


PDP = torch.tensor([0.5, 0.3, 0.2])


def test_jakes_rho_is_derived_from_mobility_not_hardcoded():
    fd = doppler_frequency_hz(3.0, 3.5e9)
    rho = jakes_slot_correlation(3.0, 3.5e9, 1e-3)
    assert fd == pytest.approx(35.0)
    assert rho == pytest.approx(torch.special.bessel_j0(torch.tensor(2 * math.pi * .035)).item())


def test_correlated_taps_match_configured_lag_and_power():
    taps = correlated_tap_trajectory(
        slots=4000, batch_size=32, pdp=PDP, rho=0.92, seed=23,
    )
    assert measured_lag1_correlation(taps) == pytest.approx(0.92, abs=0.015)
    measured = taps.abs().square().mean(dim=(0, 1))
    assert torch.allclose(measured, PDP, atol=0.02)
    assert float(measured.sum()) == pytest.approx(1.0, abs=0.02)


def test_iid_taps_are_independent_across_slots():
    taps = iid_tap_trajectory(slots=4000, batch_size=16, pdp=PDP, seed=23)
    assert abs(measured_lag1_correlation(taps)) < 0.03


def test_frequency_response_is_fft_of_correlated_taps_and_block_constant():
    taps = correlated_tap_trajectory(slots=4, batch_size=2, pdp=PDP, rho=.9, seed=23)
    response = taps_to_slot_frequency_response(taps, subcarriers=64, ofdm_symbols=32)
    assert response.shape == (4, 2, 64, 32)
    assert torch.equal(response[..., 0], torch.fft.fft(taps, n=64, dim=-1))
    assert torch.equal(response[..., 0], response[..., -1])


def test_feedback_delay_and_slot_zero_bootstrap_are_exact():
    buffer = DelayedCSIBuffer(delay_slots=1)
    assert buffer.available_for_tx(0) is None
    report = CSIReport.from_reliability(0, torch.arange(10.0))
    buffer.submit(report)
    assert buffer.available_for_tx(0) is None
    available = buffer.available_for_tx(1)
    assert available is not report
    assert available.generated_slot == 0
    assert available.available_slot == 1


def test_feedback_objects_do_not_alias_source_tensor():
    reliability = torch.arange(12.0)
    report = CSIReport.from_reliability(0, reliability)
    reliability.zero_()
    assert float(report.reliability.sum()) > 0


def test_future_channel_cannot_change_current_delayed_allocation():
    mask = torch.zeros(64, 32, dtype=torch.bool)
    mask[::4, ::4] = True
    base = deterministic_interleaver(mask)
    report = CSIReport.from_reliability(0, torch.linspace(0, 1, 1920))
    first = allocate_from_report(1, report, base, [1, 0, 2, 5, 3, 4, 6, 7])
    _future_h = torch.randn(64, 32, dtype=torch.complex64) * 100
    second = allocate_from_report(1, report, base, [1, 0, 2, 5, 3, 4, 6, 7])
    assert torch.equal(first, second)


def test_same_slot_report_is_rejected_but_oracle_api_is_explicit():
    mask = torch.zeros(64, 32, dtype=torch.bool)
    mask[::4, ::4] = True
    base = deterministic_interleaver(mask)
    same_slot = CSIReport(1, 1, torch.rand(1920))
    with pytest.raises(ValueError, match="causally available"):
        allocate_from_report(1, same_slot, base, [1, 0, 2, 5, 3, 4, 6, 7])
    oracle = allocate_from_current_oracle(
        1, same_slot.reliability, base, [1, 0, 2, 5, 3, 4, 6, 7]
    )
    assert torch.unique(oracle).numel() == 1920
    with pytest.raises(ValueError, match="slot 0"):
        allocate_from_current_oracle(
            0, same_slot.reliability, base, [1, 0, 2, 5, 3, 4, 6, 7]
        )


def test_previous_report_changes_next_slot_allocation():
    mask = torch.zeros(64, 32, dtype=torch.bool)
    mask[::4, ::4] = True
    base = deterministic_interleaver(mask)
    a = CSIReport.from_reliability(0, torch.arange(1920.0))
    b = CSIReport.from_reliability(0, torch.arange(1919, -1, -1.0))
    assert not torch.equal(
        allocate_from_report(1, a, base, [1, 0, 2, 5, 3, 4, 6, 7]),
        allocate_from_report(1, b, base, [1, 0, 2, 5, 3, 4, 6, 7]),
    )


def test_interleaver_and_delayed_map_are_bijective_and_exactly_invertible():
    mask = torch.zeros(64, 32, dtype=torch.bool)
    mask[::4, ::4] = True
    base = deterministic_interleaver(mask)
    report = CSIReport.from_reliability(0, torch.rand(1920))
    mapping = allocate_from_report(1, report, base, [1, 0, 2, 5, 3, 4, 6, 7])
    assert sorted(mapping.tolist()) == list(range(1920))
    source = torch.randn(2, 1920, dtype=torch.complex64)
    allocated = apply_resource_map(source, mapping)
    restored = invert_resource_map(allocated, mapping)
    assert torch.equal(restored, source)
    layers = mapping // 240
    assert torch.bincount(layers, minlength=8).tolist() == [240] * 8
    assert int(mask.sum()) == 128
    assert int((~mask).sum()) == 1920


def test_current_rx_mmse_uses_supplied_current_estimate():
    received = torch.ones(1, 4, dtype=torch.complex64)
    current = torch.full_like(received, 2 + 0j)
    stale = torch.full_like(received, 4 + 0j)
    output = mmse_equalize(received, current, noise_power=.1, signal_power=1.0)
    stale_output = mmse_equalize(received, stale, noise_power=.1, signal_power=1.0)
    assert not torch.equal(output, stale_output)
