import torch
from evaluation.paired import generate_paired_evaluation_batch
from speech_jscc.codecs import MockContinuousCodec
from speech_jscc.diagnostics.conv_conformer_integration import (
    build_j1_validation_suite,
    j1_realization_policy,
    j1_stage_gate,
    jammer_power_diagnostics,
)

def batch(seed=1,jsr=-12.5):
    codec=MockContinuousCodec(8,2,3,64,seed=1)
    return generate_paired_evaluation_batch(codec,batch_size=1,waveform_samples=64,channel_shape=(64,32),snr_db=10,jsr_db=jsr,jammer_type="barrage",jammed_fraction=1,pilot_spacing=4,pilot_time_spacing=4,target_power=1,seed=seed,device=torch.device("cpu"),fading="multipath_block",num_taps=6,channel_estimator="dft_tap_ls",estimator_num_taps=6)

def test_j1_policy_is_weak_random_and_reproducible():
    rows=[j1_realization_policy(23,x) for x in range(1,5)]
    assert len({x["seed"] for x in rows})==4 and len({x["jsr_db"] for x in rows})>1
    assert all(5<=x["snr_db"]<=15 and -15<=x["jsr_db"]<=-10 and x["jammer_type"]=="barrage" for x in rows)

def test_barrage_covers_pilot_and_data_and_channels_are_independent():
    value=batch(); assert value.jammer_mask[value.pilot_mask].all() and value.jammer_mask[~value.pilot_mask].all()
    assert not torch.equal(value.signal_fading,value.jammer_fading)

def test_transmit_and_received_jsr_are_finite_and_recorded():
    value=batch(jsr=-12.5); metrics=jammer_power_diagnostics(value)
    assert metrics["measured_transmit_reference_jsr_db"]==pytest.approx(-12.5,abs=1e-4)
    assert all(torch.isfinite(torch.tensor(x)) for x in metrics.values())


def _passing_group():
    layer = {
        "normalized_mse": 0.8,
        "relative_improvement_over_zero": 0.2,
        "pearson_correlation": 0.3,
        "cosine_similarity": 0.3,
        "power_ratio": 0.2,
    }
    return {
        "aggregate": {
            "finite": True,
            "normalized_mse": 0.8,
            "relative_improvement_over_zero": 0.2,
            "pearson_correlation": 0.3,
            "cosine_similarity": 0.3,
            "power_ratio": 0.2,
        },
        "per_layer": [dict(layer) for _ in range(8)],
        "layer0_summary": dict(layer),
        "layers1_to_7_summary": dict(layer),
        "baselines": {"zero": 1.0, "global_mean": 1.0, "layerwise_mean": 1.0},
        "baseline_per_layer": {"zero": [{"normalized_mse": 1.0} for _ in range(8)]},
    }


def test_j1_gate_requires_strongest_slice_and_all_randomness_diversity():
    groups = {
        "same_speaker_unseen_utterance_unseen_channel": _passing_group(),
        "unseen_speaker_unseen_utterance_unseen_channel": _passing_group(),
        "j1_unseen_jsr_-10db": _passing_group(),
        "j1_joint_snr_5db_jsr_-10db": _passing_group(),
    }
    passed = j1_stage_gate(groups, channel_diversity=4, jammer_channel_diversity=4,
                           jammer_waveform_diversity=4, noise_diversity=4, required_diversity=2)
    assert passed["stage_pass"]
    groups["j1_unseen_jsr_-10db"]["layers1_to_7_summary"]["relative_improvement_over_zero"] = 0.01
    failed = j1_stage_gate(groups, channel_diversity=4, jammer_channel_diversity=4,
                           jammer_waveform_diversity=4, noise_diversity=4, required_diversity=2)
    assert not failed["stage_pass"]
    assert not failed["strongest_weak_jsr_pass"]
    nonfinite = j1_stage_gate(_passing_validation(), channel_diversity=4, jammer_channel_diversity=4,
                              jammer_waveform_diversity=4, noise_diversity=4, required_diversity=2,
                              parameters_finite=False)
    assert not nonfinite["stage_pass"]


def _passing_validation():
    return {
        "same_speaker_unseen_utterance_unseen_channel": _passing_group(),
        "unseen_speaker_unseen_utterance_unseen_channel": _passing_group(),
        "j1_unseen_jsr_-10db": _passing_group(),
        "j1_joint_snr_5db_jsr_-10db": _passing_group(),
    }


def test_j1_validation_suite_is_stable_and_records_fixed_jammer_seeds():
    base = {"scenarios": [{"group": "unseen_speaker_unseen_utterance_unseen_channel",
                            "utterance_id": "u", "channel_seed": 7}]}
    first = build_j1_validation_suite(base, 23)
    second = build_j1_validation_suite(base, 23)
    assert first == second
    assert all("jammer_seed" in row for row in first["scenarios"])

import pytest
