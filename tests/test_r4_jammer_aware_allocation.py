import inspect

import pytest
import torch
from torch import nn

from channels.global_triplet_allocator import GlobalTripletCSIReport
from channels.physical_ofdm import NR_LIKE_R4
from channels.r4_uep_allocator import UEPProfile


ORDER = [1, 0, 2, 5, 3, 4, 6, 7]
XBEST = UEPProfile(
    "xbest",
    (3, 4, 3, 1, 5, 1, 4, 3),
    power_share=(
        0.0876379188,
        0.2577466505,
        0.2155657833,
        0.1523641692,
        0.0539495679,
        0.1115058737,
        0.0659890437,
        0.0552409929,
    ),
)


def _reports():
    count = NR_LIKE_R4.candidate_data_re
    csi = GlobalTripletCSIReport.from_reliability(
        0, torch.linspace(0.2, 2.0, count)
    )
    from channels.re_risk import REInterferenceReport

    interference = REInterferenceReport.from_power(
        0, torch.linspace(0.01, 0.4, count), noise_power=0.05
    )
    return csi, interference


def test_residual_interference_estimator_has_no_true_jammer_input():
    from channels.re_risk import estimate_rx_residual_interference_power

    parameters = set(inspect.signature(estimate_rx_residual_interference_power).parameters)
    assert {"jammer_mask", "jammer_tensor", "jammer_type"}.isdisjoint(parameters)


def test_interference_report_must_be_causally_available():
    from channels.re_risk import REInterferenceReport, require_available_interference_report

    report = REInterferenceReport.from_power(
        1, torch.ones(NR_LIKE_R4.candidate_data_re), noise_power=0.1
    )
    with pytest.raises(ValueError, match="causally available"):
        require_available_interference_report(tx_tti=1, report=report)


def test_sinr_allocator_enforces_copy_counts_unique_res_and_subband_diversity():
    from channels.r4_jammer_aware_allocator import allocate_r4_jammer_aware_sinr

    csi, interference = _reports()
    allocation = allocate_r4_jammer_aware_sinr(
        profile=NR_LIKE_R4,
        tx_tti=1,
        csi_report=csi,
        interference_report=interference,
        layer_importance_order=ORDER,
        uep_profile=XBEST,
    )
    expected = torch.tensor(XBEST.repetition).repeat_interleave(240)
    used = allocation.selected_candidate_indices[allocation.selected_candidate_indices.ge(0)]
    assert allocation.selected_re_count == 5760
    assert torch.equal(allocation.copy_count_per_source, expected)
    assert torch.unique(used).numel() == 5760
    assert allocation.distinct_subband_per_source
    assert allocation.subband_count == 5
    assert torch.allclose(allocation.power_per_source_copy.sum(), torch.tensor(5760.0), atol=2e-3)


def test_higher_interference_lowers_assignment_sinr():
    from channels.r4_jammer_aware_allocator import allocation_sinr

    gain = torch.tensor([1.0, 1.0])
    interference = torch.tensor([0.0, 1.0])
    sinr = allocation_sinr(gain, interference, noise_power=0.1, per_re_power=1.0)
    assert sinr[0] > sinr[1]


def test_same_ranked_csi_and_interference_reports_give_deterministic_mapping():
    from channels.r4_jammer_aware_allocator import allocate_r4_jammer_aware_sinr

    csi, interference = _reports()
    kwargs = dict(
        profile=NR_LIKE_R4,
        tx_tti=1,
        csi_report=csi,
        interference_report=interference,
        layer_importance_order=ORDER,
        uep_profile=XBEST,
    )
    first = allocate_r4_jammer_aware_sinr(**kwargs)
    second = allocate_r4_jammer_aware_sinr(**kwargs)
    assert torch.equal(first.selected_candidate_indices, second.selected_candidate_indices)
    assert torch.equal(first.power_per_source_copy, second.power_per_source_copy)


def test_sinr_allocator_keeps_one_live_heap_entry_per_source_subband(monkeypatch):
    """A stale refresh must replace, not accumulate, a source/band proposal."""
    import channels.r4_jammer_aware_allocator as allocator

    csi, interference = _reports()
    real_push = allocator.heapq.heappush
    maximum_live_entries = 1920 * max(XBEST.repetition)

    def bounded_push(heap, item):
        # The candidate being inserted is the only temporarily extra entry.
        assert len(heap) <= maximum_live_entries
        return real_push(heap, item)

    monkeypatch.setattr(allocator.heapq, "heappush", bounded_push)
    result = allocator.allocate_r4_jammer_aware_sinr(
        profile=NR_LIKE_R4,
        tx_tti=1,
        csi_report=csi,
        interference_report=interference,
        layer_importance_order=ORDER,
        uep_profile=XBEST,
    )
    assert result.selected_re_count == 5760


def test_csi_only_allocator_consumes_no_interference_or_jammer_input():
    """CSI-only placement is a distinct, deterministic channel-only ablation."""
    from channels.r4_jammer_aware_allocator import allocate_r4_csi_only

    csi, _ = _reports()
    parameters = set(inspect.signature(allocate_r4_csi_only).parameters)
    assert {"interference_report", "jammer_mask", "jammer_tensor", "jammer_type"}.isdisjoint(parameters)
    first = allocate_r4_csi_only(
        profile=NR_LIKE_R4,
        tx_tti=1,
        csi_report=csi,
        layer_importance_order=ORDER,
        uep_profile=XBEST,
    )
    second = allocate_r4_csi_only(
        profile=NR_LIKE_R4,
        tx_tti=1,
        csi_report=csi,
        layer_importance_order=ORDER,
        uep_profile=XBEST,
    )
    assert first.allocation_mode == "csi_only"
    assert first.selected_re_count == 5760
    assert first.distinct_subband_per_source
    assert torch.equal(first.selected_candidate_indices, second.selected_candidate_indices)


class _Codec(nn.Module):
    representation_shape = (2, 3, 4)
    waveform_samples = 3


class _Encoder(nn.Module):
    channel_state_dim = 5

    def forward(self, representation, state):
        flat = representation.flatten(1)
        return torch.complex(flat.mean(1, keepdim=True), flat.std(1, keepdim=True)).repeat(1, 1920)


class _Decoder(nn.Module):
    def forward(self, symbols, state):
        return symbols.real[:, :24].reshape(-1, 2, 3, 4)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.decoder = _Encoder(), _Decoder()


def _condition(tti: int):
    from speech_jscc.training.r4_waveform_finetune import R4ForwardCondition

    return R4ForwardCondition(
        snr_db=5.0,
        tti=tti,
        tap_coefficients=torch.tensor([[1 + 0j, .2 + .1j, .1 - .1j, .05 + 0j, .02j, .01 + 0j]]),
        noise_seed=23 + tti,
    )


def test_forward_uses_only_delayed_interference_and_preserves_fixed_uep_profile():
    from speech_jscc.training.r4_waveform_finetune import R4WaveformForward

    engine = R4WaveformForward(
        _Codec(), _Model(), uep_profile=XBEST,
        jammer_aware_allocation={"enabled": True, "mode": "delayed_rx_interference", "delay_ttis": 1},
    )
    first = engine.forward(torch.randn(1, 2, 3, 4), channel_condition=_condition(0), training=False)
    second = engine.forward(
        torch.randn(1, 2, 3, 4), channel_condition=_condition(1),
        delayed_csi=first.next_delayed_csi,
        delayed_re_interference=first.next_re_interference_report,
        training=False,
    )
    assert first.allocation.uep_profile == XBEST
    assert second.allocation.uep_profile == XBEST
    assert first.next_re_interference_report is not None
    assert second.allocation.selected_re_count == 5760
    assert torch.isfinite(second.combined_symbols).all()


def test_forward_csi_only_uses_delayed_csi_without_interference_report():
    from speech_jscc.training.r4_waveform_finetune import R4WaveformForward

    engine = R4WaveformForward(
        _Codec(), _Model(), uep_profile=XBEST,
        jammer_aware_allocation={"enabled": True, "mode": "csi_only", "delay_ttis": 1},
    )
    first = engine.forward(torch.randn(1, 2, 3, 4), channel_condition=_condition(0), training=False)
    second = engine.forward(
        torch.randn(1, 2, 3, 4), channel_condition=_condition(1),
        delayed_csi=first.next_delayed_csi,
        training=False,
    )
    assert second.allocation.allocation_mode == "csi_only"
    assert second.allocation.uep_profile == XBEST
    assert second.allocation.selected_re_count == 5760
