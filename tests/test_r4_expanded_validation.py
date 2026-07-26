from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from speech_jscc.evaluation.expanded_validation import (
    ManifestEntry,
    audit_protocol_overlap,
    build_selection_manifest,
    build_seed_manifest,
    checkpoint_gate,
    discover_candidates,
    explicit_metric_row,
    paired_statistics,
    prepare_final_test_directory,
    prepare_output_directory,
    rank_candidates,
    shared_report_for_candidate,
    utterance_level_rows,
    write_selected_checkpoint,
)
from speech_jscc.training.r4_waveform_finetune import R4ForwardCondition


def _manifest(path: str, speaker: str, role: str) -> ManifestEntry:
    return ManifestEntry(
        source_path=path,
        speaker_id=speaker,
        utterance_id=Path(path).stem,
        source_split="fixture",
        assigned_evaluation_role=role,
        crop_start_sample=0,
        crop_num_samples=16000,
        source_sha256="fixture",
    )


def test_selection_manifest_is_deterministic_speaker_balanced_and_role_labeled(
    tmp_path: Path,
):
    source = tmp_path / "test.jsonl"
    rows = []
    for speaker in ("10", "20", "30"):
        for utterance in range(3):
            rows.append(
                {
                    "audio_path": str(tmp_path / speaker / f"{speaker}-1-{utterance}.flac"),
                    "speaker_id": speaker,
                    "utt_id": f"{speaker}-1-{utterance}",
                    "split": "test",
                }
            )
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    first = build_selection_manifest(
        source, count=6, seed=91, crop_num_samples=16000, hash_files=False
    )
    second = build_selection_manifest(
        source, count=6, seed=91, crop_num_samples=16000, hash_files=False
    )

    assert first == second
    assert len(first) == 6
    assert {entry.speaker_id for entry in first[:3]} == {"10", "20", "30"}
    assert {entry.source_split for entry in first} == {"test-clean"}
    assert {entry.assigned_evaluation_role for entry in first} == {
        "selection_validation"
    }


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("train", "selection", "train_selection"),
        ("train", "final", "train_final"),
        ("selection", "final", "selection_final"),
    ],
)
def test_overlap_audit_rejects_every_pairwise_speaker_overlap(
    left: str, right: str, expected: str
):
    groups = {
        "train": [_manifest("/train/1/1/1-1-1.flac", "1", "train")],
        "selection": [_manifest("/selection/2/1/2-1-1.flac", "2", "selection")],
        "final": [_manifest("/final/3/1/3-1-1.flac", "3", "final")],
    }
    groups[right] = [
        _manifest(
            groups[right][0].source_path,
            groups[left][0].speaker_id,
            groups[right][0].assigned_evaluation_role,
        )
    ]
    with pytest.raises(ValueError, match=expected):
        audit_protocol_overlap(groups["train"], groups["selection"], groups["final"])


def test_overlap_audit_reports_zero_for_disjoint_protocol():
    report = audit_protocol_overlap(
        [_manifest("/train/1/1/1-1-1.flac", "1", "train")],
        [_manifest("/selection/2/1/2-1-1.flac", "2", "selection")],
        [_manifest("/final/3/1/3-1-1.flac", "3", "final")],
    )
    assert report["passed"] is True
    assert report["speaker_overlap_counts"] == {
        "train_selection": 0,
        "train_final": 0,
        "selection_final": 0,
    }
    assert report["utterance_overlap_counts"] == {
        "train_selection": 0,
        "train_final": 0,
        "selection_final": 0,
    }


def test_explicit_metric_row_never_emits_ambiguous_delta_names():
    row = explicit_metric_row(
        candidate={"si_sdr_db": 1.0, "waveform_snr_db": 2.0, "stft_l1": 0.6},
        initial={"si_sdr_db": 0.5, "waveform_snr_db": 1.5, "stft_l1": 0.5},
        clean={"si_sdr_db": 2.0, "waveform_snr_db": 3.0, "stft_l1": 0.4},
    )
    expected = {
        "si_sdr_absolute_db": 1.0,
        "delta_si_sdr_vs_clean_codec_db": -1.0,
        "delta_si_sdr_vs_initial_r4_db": 0.5,
        "waveform_snr_absolute_db": 2.0,
        "delta_waveform_snr_vs_clean_codec_db": -1.0,
        "delta_waveform_snr_vs_initial_r4_db": 0.5,
        "stft_l1_absolute": 0.6,
        "stft_ratio_vs_clean_codec": 1.5,
        "delta_stft_ratio_vs_initial_r4": 0.2,
    }
    assert row == pytest.approx(expected)
    assert "delta_si_sdr" not in row


def test_paired_bootstrap_is_deterministic_and_reports_requested_distribution():
    values = [-1.0, 0.0, 1.0, 2.0]
    first = paired_statistics(values, bootstrap_samples=2000, bootstrap_seed=17)
    second = paired_statistics(values, bootstrap_samples=2000, bootstrap_seed=17)
    assert first == second
    assert first["sample_count"] == 4
    assert first["mean"] == pytest.approx(0.5)
    assert first["median"] == pytest.approx(0.5)
    assert first["improved_fraction"] == pytest.approx(0.5)
    assert first["improved_by_at_least_0_5_fraction"] == pytest.approx(0.5)
    assert first["degraded_by_at_least_0_5_fraction"] == pytest.approx(0.25)
    assert first["bootstrap_samples"] == 2000
    assert first["bootstrap_seed"] == 17


def test_utterance_statistics_average_realizations_before_selection():
    rows = [
        {"utterance_id": "a", "snr_db": 5.0, "realization": 0, "delta": 1.0},
        {"utterance_id": "a", "snr_db": 5.0, "realization": 1, "delta": 3.0},
        {"utterance_id": "b", "snr_db": 5.0, "realization": 0, "delta": -2.0},
        {"utterance_id": "b", "snr_db": 5.0, "realization": 1, "delta": 0.0},
    ]
    result = utterance_level_rows(rows, metric_keys=("delta",))
    assert result == [
        {"utterance_id": "a", "snr_db": 5.0, "delta": 2.0},
        {"utterance_id": "b", "snr_db": 5.0, "delta": -1.0},
    ]


def test_candidate_discovery_keeps_named_files_and_reports_missing_intermediates(
    tmp_path: Path,
):
    for name, step in (
        ("best_5db_si_sdr.pt", 5750),
        ("best_clean_gate.pt", 5750),
        ("best_validation_average.pt", 5750),
        ("last.pt", 20000),
        ("checkpoint_step_010000.pt", 10000),
    ):
        torch.save({"global_step": step, "curriculum_stage": "B", "model": {}}, tmp_path / name)
    (tmp_path / "validation_step_006000.json").write_text(
        json.dumps({"5db_delta_si_sdr_vs_initial_r4": 2.0})
    )

    candidates = discover_candidates(tmp_path, top_k=10)

    assert {Path(row["checkpoint_path"]).name for row in candidates["included"]} == {
        "best_5db_si_sdr.pt",
        "best_clean_gate.pt",
        "best_validation_average.pt",
        "last.pt",
        "checkpoint_step_010000.pt",
    }
    assert 6000 in candidates["missing_historical_steps"]


def _summary(name: str, five: float, *, gate: bool, p10: float = -1.0) -> dict:
    return {
        "checkpoint_path": name,
        "global_step": 1,
        "training_stage": "A",
        "light_validation_score": 100.0 if name == "light-only.pt" else 0.0,
        "clean_gate_pass": gate,
        "gate_normalized_minimum_margin": 0.1 if gate else -0.1,
        "by_snr": {
            "5.0": {
                "utterance_level": {
                    "delta_si_sdr_vs_initial_r4_db": {
                        "mean": five,
                        "p10": p10,
                        "standard_deviation": 0.2,
                    }
                },
                "delta_si_sdr_vs_clean_codec_db": -0.9 if gate else -1.1,
            },
            "10.0": {
                "utterance_level": {
                    "delta_si_sdr_vs_initial_r4_db": {"mean": 0.0}
                }
            },
            "15.0": {
                "utterance_level": {
                    "delta_si_sdr_vs_initial_r4_db": {"mean": 0.0}
                }
            },
        },
    }


def test_ranking_uses_full_statistics_not_light_validation_score():
    ranked = rank_candidates(
        [
            _summary("light-only.pt", 0.1, gate=True),
            _summary("full-winner.pt", 0.4, gate=True),
        ]
    )
    assert ranked[0]["checkpoint_path"] == "full-winner.pt"


def test_ranking_is_invariant_to_candidate_input_order():
    candidates = [
        _summary("a.pt", 0.1, gate=True, p10=-0.2),
        _summary("b.pt", 0.4, gate=True, p10=-0.1),
    ]
    forward = rank_candidates(candidates)
    reverse = rank_candidates(list(reversed(candidates)))
    assert [row["checkpoint_path"] for row in forward] == [
        row["checkpoint_path"] for row in reverse
    ]


def test_nonpassing_selection_cannot_be_written_as_passing(tmp_path: Path):
    source = tmp_path / "source.pt"
    torch.save({"model": {"weight": torch.ones(1)}}, source)
    destination = tmp_path / "best_expanded_validation.pt"
    decision = {
        "checkpoint_path": str(source),
        "clean_gate_pass": False,
        "selection_status": "best_nonpassing_candidate",
    }
    write_selected_checkpoint(source, destination, decision)
    payload = torch.load(destination, map_location="cpu", weights_only=False)
    assert payload["expanded_validation"]["clean_gate_pass"] is False
    assert (
        payload["expanded_validation"]["selection_status"]
        == "best_nonpassing_candidate"
    )
    assert payload.get("passing_checkpoint") is not True


def test_all_candidates_consume_the_exact_same_precomputed_delayed_report():
    report = object()
    assert shared_report_for_candidate(report, candidate_generated_report=object()) is report


def test_seed_manifest_fixes_channel_noise_pilot_and_bootstrap_for_all_candidates():
    first = build_seed_manifest(
        utterance_ids=["a", "b"],
        snr_db=[5.0, 10.0],
        realization_seeds=[101, 202],
    )
    second = build_seed_manifest(
        utterance_ids=["a", "b"],
        snr_db=[5.0, 10.0],
        realization_seeds=[101, 202],
    )
    assert first == second
    assert first["candidate_invariant"] is True
    rows = first["conditions"]
    assert len(rows) == 8
    assert len({row["condition_id"] for row in rows}) == 8
    assert all(row["pilot_seed"] == 0 for row in rows)
    assert all(row["bootstrap_policy"] == "uniform_tti0" for row in rows)


def test_checkpoint_gate_requires_5db_waveform_and_10_15_regression_constraints():
    passing = {
        "5.0": {
            "delta_si_sdr_vs_clean_codec_db": -0.9,
            "delta_waveform_snr_vs_clean_codec_db": -0.8,
            "stft_ratio_vs_clean_codec": 1.1,
        },
        "10.0": {
            "delta_si_sdr_vs_initial_r4_db": -0.4,
            "delta_waveform_snr_vs_initial_r4_db": -0.4,
            "delta_stft_ratio_vs_initial_r4": 0.04,
        },
        "15.0": {
            "delta_si_sdr_vs_initial_r4_db": -0.4,
            "delta_waveform_snr_vs_initial_r4_db": -0.4,
            "delta_stft_ratio_vs_initial_r4": 0.04,
        },
    }
    assert checkpoint_gate(passing)["passed"] is True
    failing = json.loads(json.dumps(passing))
    failing["5.0"]["delta_si_sdr_vs_clean_codec_db"] = -1.01
    result = checkpoint_gate(failing)
    assert result["passed"] is False
    assert result["normalized_minimum_margin"] < 0


def test_existing_output_directory_is_rejected_without_overwrite(tmp_path: Path):
    output = tmp_path / "run"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing existing output directory"):
        prepare_output_directory(output, overwrite=False)


def test_final_test_directory_reuses_selection_without_deleting_it(tmp_path: Path):
    output = tmp_path / "selection"
    output.mkdir()
    decision = output / "selection_decision.json"
    selected = output / "best_expanded_validation.pt"
    manifest = output / "final_test_manifest_reference.jsonl"
    decision.write_text('{"checkpoint_path": "source.pt"}')
    selected.write_bytes(b"checkpoint")
    manifest.write_text("{}\n")

    final = prepare_final_test_directory(output, overwrite=False)

    assert final == output / "final_test"
    assert decision.is_file()
    assert selected.read_bytes() == b"checkpoint"
    assert final.is_dir()


def test_noise_variance_override_is_part_of_the_paired_forward_condition():
    condition = R4ForwardCondition(
        snr_db=5.0,
        tti=0,
        tap_coefficients=torch.ones(1, 6, dtype=torch.complex64),
        noise_seed=91,
        noise_variance_override=10 ** (-0.5),
    )
    assert condition.noise_variance_override == pytest.approx(10 ** (-0.5))
