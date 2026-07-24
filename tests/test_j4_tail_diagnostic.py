import math
from dataclasses import replace

import pytest
import torch

from channels.pilot import equalize_with_csi
from evaluation.paired import PairedEvaluationBatch
from speech_jscc.diagnostics.conv_conformer_integration import build_j4_validation_suite
from speech_jscc.diagnostics.j4_tail import (
    ROOT_CAUSE_THRESHOLDS,
    classify_root_cause,
    data_only_batch,
    distribution,
    pair_stochastic_environment,
    paired_burst_barrage_batches,
    select_strongest_condition,
    summarize_failure_rates,
    wilson_interval,
)


def _batch():
    shape = (1, 4, 4)
    pilot = torch.zeros(shape, dtype=torch.bool); pilot[:, ::2, ::2] = True
    burst = torch.zeros(shape, dtype=torch.bool); burst[:, :, :2] = True
    one = torch.ones(shape, dtype=torch.complex64)
    return PairedEvaluationBatch(7, "burst", 1.0, None, torch.zeros(1, 8, 2, 2),
        torch.tensor([5.]), torch.tensor([0.]), pilot, one, one * burst, one,
        one, one * 2, burst, {"test": True})


def test_selects_observed_worst_tail_instead_of_largest_fraction():
    distribution = {"selected_snr_range_db": [5, 15], "selected_global_jsr_range_db": [-10, 0],
        "selected_burst_fractions": [.125, .25, .5],
        "worst_tail_condition": {"snr_db": 5, "jsr_db": 0, "requested_fraction": .125}}
    assert select_strongest_condition(distribution) == {"snr_db": 5.0, "jsr_db": 0.0, "burst_fraction": .125}
    base = {"scenarios": [{"group": "unseen_speaker_unseen_utterance_unseen_channel", "utterance_id": "u"}]}
    suite = build_j4_validation_suite(base, 23, distribution)
    strongest = next(x for x in suite["scenarios"] if x["group"] == "j4_strongest_selected_condition")
    assert strongest["jammed_fraction"] == .125


def test_wilson_interval_and_distinct_content_rates():
    low, high = wilson_interval(1, 8)
    assert low == pytest.approx(.0224, abs=1e-3) and high == pytest.approx(.4709, abs=1e-3)
    rows = [{"utterance_id": "a", "layer7_improvement": -.1},
            {"utterance_id": "a", "layer7_improvement": .1},
            {"utterance_id": "b", "layer7_improvement": .2}]
    value = summarize_failure_rates(rows)
    assert value["realization"]["negative"]["count"] == 1
    assert value["realization"]["negative"]["total"] == 3
    assert value["utterance"]["negative"]["count"] == 1
    assert value["utterance"]["negative"]["total"] == 2


def test_data_only_batch_preserves_paired_tensors_and_removes_pilot_energy():
    original = _batch(); modified = data_only_batch(original)
    for name in ("representation", "noise", "signal_fading", "jammer_fading", "pilot_mask"):
        assert getattr(modified, name).data_ptr() == getattr(original, name).data_ptr()
    assert not modified.jammer_mask[modified.pilot_mask].any()
    assert modified.jammer[modified.pilot_mask].abs().max() == 0
    assert torch.isclose(modified.jammer.abs().square().mean(), original.jammer.abs().square().mean())


def test_barrage_variant_reuses_baseline_channel_noise_and_content():
    baseline = _batch(); variant = replace(_batch(), noise=torch.zeros_like(baseline.noise),
        signal_fading=torch.full_like(baseline.signal_fading, 3), jammer_fading=torch.full_like(baseline.jammer_fading, 4))
    paired = pair_stochastic_environment(baseline, variant)
    for name in ("representation", "noise", "signal_fading", "jammer_fading", "pilot_mask"):
        assert getattr(paired, name).data_ptr() == getattr(baseline, name).data_ptr()
    assert paired.jammer.data_ptr() == variant.jammer.data_ptr()


def test_burst_and_barrage_share_canonical_complex_waveform():
    burst, equal_global, equal_active, source_hash = paired_burst_barrage_batches(_batch(), seed=99)
    mask = burst.jammer_mask
    phase_ratio = burst.jammer[mask] / equal_global.jammer[mask]
    assert torch.allclose(phase_ratio.imag, torch.zeros_like(phase_ratio.imag), atol=1e-6)
    assert torch.allclose(phase_ratio.real, torch.full_like(phase_ratio.real, phase_ratio.real[0]), atol=1e-6)
    assert source_hash and equal_active.jsr_db.item() == pytest.approx(10 * math.log10(1 / .5))


def test_root_cause_rules_use_predeclared_substantial_reduction():
    baseline = {"negative_rate": .20, "p10": -.04, "mean": .04}
    oracle = {"negative_rate": .08, "p10": -.01, "mean": .06}
    unchanged = {"negative_rate": .19, "p10": -.035, "mean": .04}
    result = classify_root_cause(baseline, oracle=oracle, data_only=unchanged,
        clipped={10: unchanged}, oracle_clipped={10: unchanged}, equal_global_barrage=unchanged,
        thresholds=ROOT_CAUSE_THRESHOLDS)
    assert result["classification"] == "PILOT_CSI_DOMINANT"
    assert result["evidence"]["oracle"]["substantial"] is True


def test_single_sample_distribution_has_zero_population_std():
    assert distribution([0.25])["std"] == 0.0


def test_diagnostic_gain_cap_limits_zf_without_changing_default():
    received = torch.ones(1, 2, dtype=torch.complex64)
    channel = torch.tensor([[.01 + 0j, .1 + 0j]], dtype=torch.complex64)
    ordinary = equalize_with_csi(received, channel)
    assert torch.equal(ordinary, equalize_with_csi(received, channel, gain_cap=None))
    clipped = equalize_with_csi(received, channel, gain_cap=20.0)
    assert clipped.abs().max() <= 20.0 + 1e-6
