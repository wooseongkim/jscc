from __future__ import annotations

import inspect

import pytest
import torch

from channels.physical_ofdm import NR_LIKE_R3, active_grid_masks
from channels.repetition_mrc import (
    RepetitionCSIReport,
    allocate_repetition3,
    coherent_mrc,
    oracle_branch_sinr,
)


ORDER = [1, 0, 2, 5, 3, 4, 6, 7]


def test_r3_three_copy_mapping_counts_groups_and_inversion():
    assert NR_LIKE_R3.candidate_data_re == 5832
    allocation = allocate_repetition3(
        profile=NR_LIKE_R3, tx_tti=0, report=None, layer_importance_order=ORDER
    )
    assert allocation.selected_candidate_indices.shape == (3, 1920)
    assert torch.unique(allocation.selected_candidate_indices).numel() == 5760
    assert allocation.unused_candidate_re == 72
    masks = active_grid_masks(NR_LIKE_R3)
    coordinates = masks.candidate_data.nonzero()
    for branch in range(3):
        selected = allocation.selected_candidate_indices[branch]
        frequencies = coordinates[selected, 0]
        assert int(frequencies.min()) >= branch * 72
        assert int(frequencies.max()) < (branch + 1) * 72
        assert torch.unique(selected).numel() == 1920
        assert allocation.group_unused_counts[branch] == 24
    source = torch.randn(2, 1920, dtype=torch.complex64)
    grid = allocation.place(source)
    assert torch.count_nonzero(grid[:, masks.candidate_data]).item() == 2 * 5760
    branches = allocation.extract_source_order(grid)
    assert branches.shape == (2, 3, 1920)
    expected = source[:, None, :] * allocation.power_source_order.sqrt()[None]
    assert torch.allclose(branches, expected)


def test_every_symbol_has_three_distinct_frequency_groups_and_layer_counts():
    allocation = allocate_repetition3(
        profile=NR_LIKE_R3, tx_tti=0, report=None, layer_importance_order=ORDER
    )
    masks = active_grid_masks(NR_LIKE_R3)
    coordinates = masks.candidate_data.nonzero()
    source_resources = allocation.source_to_candidate_indices
    assert source_resources.shape == (3, 1920)
    physical_frequency = coordinates[source_resources, 0]
    assert torch.all(physical_frequency[0] < 72)
    assert torch.all((physical_frequency[1] >= 72) & (physical_frequency[1] < 144))
    assert torch.all(physical_frequency[2] >= 144)
    assert torch.all(physical_frequency[0] != physical_frequency[1])
    assert torch.all(physical_frequency[1] != physical_frequency[2])
    times = coordinates[source_resources, 1]
    assert float(
        ((times[0] != times[1]) & (times[0] != times[2]) & (times[1] != times[2]))
        .float()
        .mean()
    ) > .9
    layers = torch.arange(1920) // 240
    assert torch.bincount(layers, minlength=8).tolist() == [240] * 8
    assert (torch.bincount(layers, minlength=8) * 3).tolist() == [720] * 8


def test_delayed_report_is_causal_and_tx_rx_reproduce_all_maps():
    reliability = torch.rand(NR_LIKE_R3.candidate_data_re)
    report = RepetitionCSIReport.from_reliability(0, reliability)
    tx = allocate_repetition3(
        profile=NR_LIKE_R3, tx_tti=1, report=report, layer_importance_order=ORDER
    )
    rx = allocate_repetition3(
        profile=NR_LIKE_R3, tx_tti=1, report=report, layer_importance_order=ORDER
    )
    assert torch.equal(tx.selected_candidate_indices, rx.selected_candidate_indices)
    assert torch.equal(tx.resource_to_source, rx.resource_to_source)
    assert torch.equal(tx.power_per_resource, rx.power_per_resource)
    current = RepetitionCSIReport(1, 1, reliability)
    with pytest.raises(ValueError, match="past CSI"):
        allocate_repetition3(
            profile=NR_LIKE_R3, tx_tti=1, report=current,
            layer_importance_order=ORDER,
        )


@pytest.mark.parametrize(
    ("contract", "branch_energy", "packet_energy"),
    [("fixed_power_per_copy", 1920.0, 5760.0),
     ("fixed_total_packet_energy", 640.0, 1920.0)],
)
def test_energy_contracts_are_exact(contract, branch_energy, packet_energy):
    report = RepetitionCSIReport.from_reliability(
        0, torch.logspace(-3, 3, NR_LIKE_R3.candidate_data_re)
    )
    allocation = allocate_repetition3(
        profile=NR_LIKE_R3, tx_tti=1, report=report,
        layer_importance_order=ORDER, energy_contract=contract,
    )
    energies = allocation.power_per_resource.sum(dim=1)
    assert energies.tolist() == pytest.approx([branch_energy] * 3, abs=3e-4)
    assert float(energies.sum()) == pytest.approx(packet_energy, abs=6e-4)
    expected_mean = 1.0 if contract == "fixed_power_per_copy" else 1 / 3
    assert allocation.power_per_resource.mean(dim=1).tolist() == pytest.approx(
        [expected_mean] * 3, abs=2e-7
    )


def test_scalar_coherent_mrc_matches_hand_calculation_and_weights():
    y = torch.tensor([[[2 + 1j], [1 - 2j], [-1 + .5j]]])
    h = torch.tensor([[[1 + 0j], [.5 + .5j], [0 + 1j]]])
    power = torch.tensor([[1.0], [2.0], [.5]])
    noise = torch.tensor([1.0, 2.0, .5])
    result = coherent_mrc(y, h, power, noise)
    g = h * power.sqrt()[None]
    numerator = (g.conj() * y / noise[None, :, None]).sum(1)
    denominator = (g.abs().square() / noise[None, :, None]).sum(1)
    assert torch.allclose(result.estimate, numerator / (denominator + 1e-12))
    assert torch.all(result.weights >= 0)
    assert torch.allclose(result.weights.sum(1), torch.ones(1, 1))


def test_perfect_channel_mrc_reconstructs_and_zero_branch_gets_zero_weight():
    source = torch.randn(2, 1920, dtype=torch.complex64)
    h = torch.randn(2, 3, 1920, dtype=torch.complex64)
    h[:, 2] = 0
    power = torch.ones(3, 1920)
    y = h * source[:, None]
    result = coherent_mrc(y, h, power, 1.0)
    assert result.estimate.shape == source.shape
    assert torch.allclose(result.estimate, source, atol=2e-6)
    assert float(result.weights[:, 2].abs().max()) == 0


def test_oracle_theory_is_additive_and_matches_empirical_noise_simulation():
    generator = torch.Generator().manual_seed(91)
    samples = 50_000
    source = torch.ones(1, samples, dtype=torch.complex64)
    h = torch.tensor([1 + .2j, .5 - .3j, -.2 + .8j])[:, None].expand(3, samples)
    power = torch.tensor([1.0, .8, 1.2])[:, None].expand_as(h.real)
    variance = .2
    noise = torch.complex(
        torch.randn(1, 3, samples, generator=generator),
        torch.randn(1, 3, samples, generator=generator),
    ) * (variance / 2) ** .5
    y = h[None] * power.sqrt()[None] * source[:, None] + noise
    result = coherent_mrc(y, h[None], power, variance)
    branch = oracle_branch_sinr(h[None], power, variance, source_power=1.0)
    assert torch.all(branch > 0)
    assert torch.allclose(branch.sum(1), result.theoretical_combined_sinr)
    empirical = source.abs().square().sum() / (result.estimate - source).abs().square().sum()
    assert float(empirical / result.theoretical_combined_sinr.mean()) == pytest.approx(
        1.0, rel=.025
    )
    assert float(branch[:, :2].sum()) >= float(branch[:, 0].sum())


def test_default_mrc_uses_only_denominator_epsilon_without_clamps():
    source = inspect.getsource(coherent_mrc)
    assert "clamp" not in source
    assert "threshold" not in source
    assert "epsilon" in source
