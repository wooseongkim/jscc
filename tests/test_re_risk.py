import inspect

import pytest
import torch

from channels.global_triplet_allocator import (
    GlobalTripletCSIReport,
    allocate_global_balanced_triplets,
)
from channels.physical_ofdm import NR_LIKE_R4, active_grid_masks
from channels.re_risk import (
    RERiskReport,
    allocate_risk_aware_global_balanced_triplets,
    combine_csi_and_risk_report,
    combine_reliability_with_risk,
    estimate_rx_residual_risk_map,
    oracle_jamming_mask_to_risk_report,
)


ORDER = [1, 0, 2, 5, 3, 4, 6, 7]


def _csi_report(tti=1):
    count = NR_LIKE_R4.candidate_data_re
    return GlobalTripletCSIReport.from_reliability(tti - 1, torch.linspace(0.2, 2.0, count))


def test_zero_risk_alpha_is_exactly_the_legacy_global_triplet_allocation():
    report = _csi_report()
    risk = RERiskReport.from_risk(0, torch.linspace(0, 1, NR_LIKE_R4.candidate_data_re))
    legacy = allocate_global_balanced_triplets(
        profile=NR_LIKE_R4, tx_tti=1, report=report, layer_importance_order=ORDER,
    )
    aware = allocate_risk_aware_global_balanced_triplets(
        profile=NR_LIKE_R4, tx_tti=1, csi_report=report, risk_report=risk,
        risk_alpha=0.0, layer_importance_order=ORDER,
    )
    assert torch.equal(legacy.selected_candidate_indices, aware.selected_candidate_indices)
    assert torch.equal(legacy.resource_to_source, aware.resource_to_source)


def test_high_risk_reduces_effective_reliability():
    reliability = torch.tensor([1.0, 1.0])
    effective = combine_reliability_with_risk(reliability, torch.tensor([0.0, 1.0]), 1.0)
    assert effective[0] == pytest.approx(1.0)
    assert effective[1] < effective[0]


def test_re_risk_report_uses_cpu_like_the_delayed_csi_report():
    report = RERiskReport.from_risk(0, torch.ones(NR_LIKE_R4.candidate_data_re))
    assert report.risk.device.type == "cpu"


def test_oracle_jamming_mask_becomes_candidate_re_risk_report():
    masks = active_grid_masks(NR_LIKE_R4)
    jammer_mask = torch.zeros(1, NR_LIKE_R4.active_subcarriers, NR_LIKE_R4.n_ofdm_symbols, dtype=torch.bool)
    coordinate = masks.candidate_data.nonzero()[7]
    jammer_mask[0, coordinate[0], coordinate[1]] = True
    report = oracle_jamming_mask_to_risk_report(jammer_mask, masks.candidate_data, generated_tti=2)
    assert report.risk.shape == (NR_LIKE_R4.candidate_data_re,)
    assert report.risk[7] == 1.0
    assert report.available_tti == 3


def test_delayed_risk_report_must_be_causally_available():
    csi = _csi_report(tti=2)
    future = RERiskReport.from_risk(2, torch.zeros(NR_LIKE_R4.candidate_data_re))
    with pytest.raises(ValueError, match="causally available"):
        combine_csi_and_risk_report(2, csi, future, 0.5)
    old = RERiskReport.from_risk(1, torch.zeros(NR_LIKE_R4.candidate_data_re))
    combined = combine_csi_and_risk_report(2, csi, old, 0.5)
    assert combined is not None
    assert combined.available_tti == 2


def test_deployable_rx_residual_estimator_cannot_accept_true_jammer_labels_or_masks():
    parameters = set(inspect.signature(estimate_rx_residual_risk_map).parameters)
    assert "jammer_mask" not in parameters
    assert "jammer_type" not in parameters
    assert "jammer_tensor" not in parameters
