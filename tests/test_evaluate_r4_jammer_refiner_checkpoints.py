import csv
import math

import pytest

from speech_jscc.evaluation.evaluate_r4_jammer_refiner_checkpoints import (
    FIXED_JAMMER_TYPES,
    build_fixed_condition_plan,
    condition_key,
    mask_classification_metrics,
    _resolve_fixed_uep_profile,
    summarize_perceptual_metrics,
    verify_paired_row_keys,
    build_three_way_allocation_comparison,
)


def test_fixed_condition_plan_is_repeatable_and_has_all_requested_jammer_conditions():
    first = build_fixed_condition_plan(
        sample_ids=["a.wav", "b.wav"], crop_offsets=[0, 3], snr_db=[5],
        jsr_db=[0, 5, 10], realizations=1, seed=19,
    )
    second = build_fixed_condition_plan(
        sample_ids=["a.wav", "b.wav"], crop_offsets=[0, 3], snr_db=[5],
        jsr_db=[0, 5, 10], realizations=1, seed=19,
    )
    assert first == second
    assert {row["jammer_type"] for row in first} == set(FIXED_JAMMER_TYPES)
    assert len(first) == 2 * (1 + 4 * 3)
    assert len({condition_key(row) for row in first}) == len(first)


def test_paired_row_key_verification_rejects_checkpoint_specific_conditions():
    plan = build_fixed_condition_plan(
        sample_ids=["a.wav"], crop_offsets=[0], snr_db=[5],
        jsr_db=[0], realizations=1, seed=1,
    )
    rows = [{**row, "checkpoint_name": "first.pt"} for row in plan]
    rows += [{**row, "checkpoint_name": "second.pt"} for row in plan]
    verify_paired_row_keys(rows, ["first.pt", "second.pt"])
    rows[-1]["noise_seed"] += 1
    with pytest.raises(ValueError, match="paired validation row keys differ"):
        verify_paired_row_keys(rows, ["first.pt", "second.pt"])


def test_no_jammer_metrics_expose_false_positive_rate_instead_of_only_empty_iou():
    metrics = mask_classification_metrics(
        predicted=[[True, False], [False, False]],
        target=[[False, False], [False, False]],
    )
    assert metrics["false_positive_rate"] == pytest.approx(0.25)
    assert metrics["false_negative_rate"] == 0.0
    assert metrics["iou"] == 0.0
    assert metrics["f1"] == 0.0


def test_mask_metrics_are_finite_for_nonempty_target():
    metrics = mask_classification_metrics(
        predicted=[[True, False], [False, True]],
        target=[[True, False], [True, False]],
    )
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["iou"] == pytest.approx(1.0 / 3.0)


def test_optimizer_artifact_freezes_x_best_profile_without_reoptimizing(tmp_path):
    artifact = tmp_path / "selected_profiles.json"
    artifact.write_text('''{"selected":{"x_best":{"status":"SELECTED","candidate":{"candidate":{"profile_id":"abc","repetition":[3,4,3,1,5,1,4,3],"power_share":[0.1,0.2,0.2,0.1,0.1,0.1,0.1,0.1]}}}}}''')
    name, profile = _resolve_fixed_uep_profile({
        "uep_profile_selection_artifact": str(artifact), "uep_profile_selection_key": "x_best",
    })
    assert name == "x_best_abc"
    assert profile is not None and profile.repetition == (3, 4, 3, 1, 5, 1, 4, 3)


def test_three_way_allocation_comparison_requires_identical_condition_keys():
    key = dict(
        checkpoint_name="last.pt", sample_id="one.flac", crop_offset=0,
        snr_db=5.0, jsr_db=5.0, jammer_type="broadband", realization_index=0,
        channel_seed=1, noise_seed=2, jammer_seed=3, condition_hash="abc",
    )
    existing = [{**key, "raw_si_sdr": -2.0, "refined_si_sdr": -3.0, "mapping_hash": "old"}]
    csi_only = [{**key, "raw_si_sdr": -1.0, "refined_si_sdr": -2.0, "mapping_hash": "csi"}]
    interference = [{**key, "raw_si_sdr": 0.5, "refined_si_sdr": -0.5, "mapping_hash": "sinr"}]
    rows = build_three_way_allocation_comparison(existing, csi_only, interference)
    assert rows[0]["delta_csi_only_minus_existing"] == pytest.approx(1.0)
    assert rows[0]["delta_interference_minus_csi_only"] == pytest.approx(1.5)
    assert rows[0]["delta_interference_minus_existing"] == pytest.approx(2.5)
    interference[0]["condition_hash"] = "mismatch"
    with pytest.raises(ValueError, match="condition key mismatch"):
        build_three_way_allocation_comparison(existing, csi_only, interference)


def test_three_way_comparison_carries_all_enabled_raw_speech_metrics():
    key = dict(
        checkpoint_name="last.pt", sample_id="one.flac", crop_offset=0,
        snr_db=5.0, jsr_db=5.0, jammer_type="broadband", realization_index=0,
        channel_seed=1, noise_seed=2, jammer_seed=3, condition_hash="abc",
        raw_si_sdr=-2.0, refined_si_sdr=-3.0, mapping_hash="map",
    )
    existing = [{**key, "raw_estoi": 0.2, "raw_wer": 0.9, "raw_visqol_mos_lqo": 1.5}]
    csi = [{**key, "raw_estoi": 0.3, "raw_wer": 0.8, "raw_visqol_mos_lqo": 1.8}]
    interference = [{**key, "raw_estoi": 0.4, "raw_wer": 0.7, "raw_visqol_mos_lqo": 2.1}]
    row = build_three_way_allocation_comparison(existing, csi, interference)[0]
    assert row["raw_estoi_interference"] == pytest.approx(0.4)
    assert row["raw_wer_csi_only"] == pytest.approx(0.8)
    assert row["raw_visqol_mos_lqo_existing"] == pytest.approx(1.5)


def test_three_way_csv_loader_reads_the_distinct_csi_only_file(tmp_path):
    from speech_jscc.evaluation.evaluate_r4_jammer_refiner_checkpoints import (
        build_three_way_allocation_comparison_from_csv,
    )

    key = dict(
        checkpoint_name="last.pt", sample_id="one.flac", crop_offset=0,
        snr_db=5.0, jsr_db=5.0, jammer_type="broadband", realization_index=0,
        channel_seed=1, noise_seed=2, jammer_seed=3, condition_hash="abc",
        refined_si_sdr=-3.0, mapping_hash="map",
    )
    paths = [tmp_path / f"{name}.csv" for name in ("existing", "csi", "interference")]
    rows = [
        {**key, "raw_si_sdr": -10.0},
        {**key, "raw_si_sdr": -5.0},
        {**key, "raw_si_sdr": 0.0},
    ]
    for path, row in zip(paths, rows):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    joined = build_three_way_allocation_comparison_from_csv(*paths)
    assert joined[0]["si_sdr_existing_raw"] == pytest.approx(-10.0)
    assert joined[0]["si_sdr_csi_only_raw"] == pytest.approx(-5.0)
    assert joined[0]["si_sdr_interference_raw"] == pytest.approx(0.0)


def test_perceptual_summary_reports_raw_quality_and_absolute_si_sdr_tail():
    rows = [
        {"jammer_type": "broadband", "jsr_db": 10.0, "snr_db": 5.0,
         "raw_si_sdr": -11.0, "raw_estoi": 0.2, "raw_wer": 1.0, "raw_visqol_mos_lqo": 1.5},
        {"jammer_type": "broadband", "jsr_db": 10.0, "snr_db": 5.0,
         "raw_si_sdr": -9.0, "raw_estoi": 0.4, "raw_wer": 0.5, "raw_visqol_mos_lqo": 2.5},
    ]
    summary = summarize_perceptual_metrics(rows)
    assert len(summary) == 2
    for row in summary:
        assert row["raw_estoi"] == pytest.approx(0.3)
        assert row["raw_wer"] == pytest.approx(0.75)
        assert row["raw_visqol_mos_lqo"] == pytest.approx(2.0)
        assert row["frac_raw_si_sdr_lt_minus10"] == pytest.approx(0.5)
    assert {row["snr_db"] for row in summary} == {"5", "all"}
