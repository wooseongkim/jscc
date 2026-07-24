from __future__ import annotations

import torch

from speech_jscc.diagnostics.oracle_csi_comparison import (
    empirical_symbol_metrics,
    paired_evaluation_grid,
    residual_decomposition,
)


def test_paired_grid_has_exact_snr_realization_coverage_and_unique_seeds():
    rows = paired_evaluation_grid(23, utterance_count=64, realizations=2)
    assert len(rows) == 64 * 2 * 3
    assert {row["snr_db"] for row in rows} == {5.0, 10.0, 15.0}
    assert len({row["seed"] for row in rows}) == len(rows)
    assert rows == paired_evaluation_grid(23, utterance_count=64, realizations=2)


def test_oracle_empirical_and_theoretical_sinr_match_for_constant_channel_noise():
    x = torch.ones(1, 4, dtype=torch.complex64)
    h = torch.full_like(x, 2 + 0j)
    noise = torch.tensor([[1, -1, 1j, -1j]], dtype=torch.complex64) * 0.1
    x_hat = (h * x + noise) / h
    metrics = empirical_symbol_metrics(
        x, x_hat, h=h, requested_noise_power=0.01, oracle=True
    )
    assert metrics["post_eq_sinr_empirical_db"] == pytest.approx(
        metrics["post_eq_sinr_oracle_theory_db"], abs=1e-5
    )


def test_residual_decomposition_sums_noise_csi_and_cross_energy():
    x = torch.tensor([[1 + 0j, 2 + 0j]], dtype=torch.complex64)
    h = torch.tensor([[1 + 0j, 0.5 + 0j]], dtype=torch.complex64)
    h_hat = torch.tensor([[0.8 + 0j, 0.4 + 0j]], dtype=torch.complex64)
    noise = torch.tensor([[0.1 + 0j, -0.1 + 0j]], dtype=torch.complex64)
    result = residual_decomposition(x, h, h_hat, noise)
    reconstructed = (
        result["noise_component_energy_ratio"]
        + result["csi_distortion_energy_ratio"]
        + result["cross_term_energy_ratio"]
    )
    assert reconstructed == pytest.approx(result["total_residual_energy_ratio"], abs=1e-6)


import pytest
