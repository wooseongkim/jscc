import math

import torch


def test_no_jammer_has_zero_grid_and_no_active_resources():
    from speech_jscc.evaluation.r4_jammer_baseline import build_r4_jammer

    signal = torch.ones(1, 4, 6, dtype=torch.complex64)
    data = torch.ones(4, 6, dtype=torch.bool)
    jammer = build_r4_jammer(signal, data, jammer_type="no_jammer", jsr_db=None, seed=7)

    assert not jammer.mask.any()
    assert torch.count_nonzero(jammer.grid) == 0
    assert jammer.statistics["active_re_fraction"] == 0.0


def test_broadband_jammer_meets_subset_jsr_and_is_deterministic():
    from speech_jscc.evaluation.r4_jammer_baseline import build_r4_jammer

    signal = torch.ones(1, 4, 8, dtype=torch.complex64)
    data = torch.ones(4, 8, dtype=torch.bool)
    first = build_r4_jammer(signal, data, jammer_type="broadband_awgn", jsr_db=10.0, seed=91)
    second = build_r4_jammer(signal, data, jammer_type="broadband_awgn", jsr_db=10.0, seed=91)

    assert torch.equal(first.mask, second.mask)
    assert torch.equal(first.grid, second.grid)
    assert math.isclose(first.statistics["measured_pre_channel_jsr_db"], 10.0, abs_tol=1e-4)
    assert first.statistics["active_re_fraction"] == 1.0


def test_subband_burst_and_block_masks_are_data_only_and_deterministic():
    from speech_jscc.evaluation.r4_jammer_baseline import build_r4_jammer

    signal = torch.ones(1, 10, 12, dtype=torch.complex64)
    data = torch.ones(10, 12, dtype=torch.bool)
    data[0, 0] = False
    for kind in ("subband", "burst", "block"):
        first = build_r4_jammer(signal, data, jammer_type=kind, jsr_db=0.0, seed=31,
                                 subband_fraction=0.3, burst_fraction=0.25)
        second = build_r4_jammer(signal, data, jammer_type=kind, jsr_db=0.0, seed=31,
                                  subband_fraction=0.3, burst_fraction=0.25)
        assert not (first.mask & ~data).any()
        assert torch.equal(first.mask, second.mask)
        assert torch.equal(first.grid, second.grid)


def test_r4_physical_forward_preserves_clean_path_and_injects_jammer():
    from channels.global_triplet_allocator import allocate_global_balanced_triplets
    from channels.physical_ofdm import NR_LIKE_R4
    from speech_jscc.training.r4_waveform_finetune import r4_physical_layer_forward

    allocation = allocate_global_balanced_triplets(
        profile=NR_LIKE_R4, tx_tti=0, report=None,
        layer_importance_order=[1, 0, 2, 5, 3, 4, 6, 7],
    )
    source = torch.complex(torch.randn(1, 1920), torch.randn(1, 1920))
    taps = torch.zeros(1, 11, dtype=torch.complex64); taps[:, 0] = 1
    common = dict(snr_db=5.0, tap_delay_samples=(0, 2, 4, 6, 8, 10), estimator_num_taps=6,
                  estimator_ridge_lambda=1e-6, epsilon=1e-12)
    clean = r4_physical_layer_forward(source, allocation, taps, noise_generator=torch.Generator().manual_seed(12), **common)
    explicit_clean = r4_physical_layer_forward(source, allocation, taps, noise_generator=torch.Generator().manual_seed(12), jammer_type="no_jammer", **common)
    jammed = r4_physical_layer_forward(source, allocation, taps, noise_generator=torch.Generator().manual_seed(12), jammer_type="broadband_awgn", jammer_jsr_db=0.0, jammer_seed=17, **common)

    assert torch.equal(clean.received_time, explicit_clean.received_time)
    assert torch.count_nonzero(explicit_clean.jammer_grid) == 0
    assert torch.count_nonzero(jammed.jammer_grid) > 0
    assert not torch.equal(clean.received_time, jammed.received_time)


def test_clean_anchor_is_not_applied_to_a_smoke_subset():
    from evaluate_r4_jammer_baseline import clean_anchor_status

    result = clean_anchor_status(
        observed=1.67378, reference=1.34656, tolerance=0.005,
        device="cuda", utterances=4, realizations=1, full_utterances=64,
        full_realizations=2,
    )

    assert result["status"] == "NOT_APPLICABLE_SUBSET"
    assert result["pass"] is None


def test_repetition_overlap_reports_exact_copy_histogram_for_source_symbols():
    from channels.global_triplet_allocator import allocate_global_balanced_triplets
    from channels.physical_ofdm import NR_LIKE_R4, active_grid_masks
    from speech_jscc.evaluation.r4_jammer_baseline import repetition_overlap_diagnostics

    allocation = allocate_global_balanced_triplets(
        profile=NR_LIKE_R4, tx_tti=0, report=None,
        layer_importance_order=[1, 0, 2, 5, 3, 4, 6, 7],
    )
    data = active_grid_masks(NR_LIKE_R4).candidate_data
    empty = torch.zeros(1, *data.shape, dtype=torch.bool)
    full = data.unsqueeze(0)
    no_hit = repetition_overlap_diagnostics(allocation, empty)
    all_hit = repetition_overlap_diagnostics(allocation, full)

    assert no_hit["copy_count_histogram"] == {"0": 1920, "1": 0, "2": 0, "3": 0}
    assert all_hit["copy_count_histogram"] == {"0": 0, "1": 0, "2": 0, "3": 1920}
