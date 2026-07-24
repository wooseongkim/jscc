from diagnose_j3_narrowband_boundary import select_distribution


def row(snr, jsr, fraction, aggregate, enhancement, layer7):
    return {
        "snr_db": float(snr), "jsr_db": float(jsr), "requested_fraction": float(fraction),
        "group": "unseen_speaker_unseen_utterance_unseen_channel",
        "aggregate_relative_improvement_over_zero_mean": aggregate,
        "layers1_to_7_relative_improvement_over_zero_mean": enhancement,
        "layer7_relative_improvement_over_zero_mean": layer7,
        "pilot_resource_overlap_ratio_mean": fraction,
    }


def test_selection_uses_swept_transition_evidence_and_multiple_bandwidths():
    rows = [
        row(5, -10, .125, .20, .12, .08),
        row(15, 0, .25, .12, .06, .025),
        row(5, 0, .50, -.02, -.03, -.04),
    ]
    selected = select_distribution(rows)
    assert selected["defined"]
    assert selected["selected_snr_range_db"] == [5.0, 15.0]
    assert selected["selected_global_jsr_range_db"] == [-10.0, 0.0]
    assert selected["selected_jammed_subcarrier_fractions"] == [.125, .25]


def test_selection_refuses_single_bandwidth_evidence():
    selected = select_distribution([row(5, -10, .25, .20, .12, .08)])
    assert not selected["defined"]
