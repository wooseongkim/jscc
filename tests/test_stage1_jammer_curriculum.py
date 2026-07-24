from __future__ import annotations

import pytest

from speech_jscc.diagnostics.random_distribution import (
    STAGE_DEFINITIONS,
    build_stage_validation_suite,
    catastrophic_forgetting,
    validate_curriculum_parent,
)


def test_j1_to_j5_stage_definitions_match_approved_distributions() -> None:
    assert STAGE_DEFINITIONS["j1_weak_barrage"]["jsr_db_range"] == [-15.0, -10.0]
    assert STAGE_DEFINITIONS["j2_moderate_barrage"]["jsr_db_range"] == [-10.0, -5.0]
    assert STAGE_DEFINITIONS["j3_strong_barrage"]["jsr_db_range"] == [-5.0, 0.0]
    assert STAGE_DEFINITIONS["j4_mixed_sparse"]["jammer_probabilities"] == {"barrage": .5, "narrowband": .25, "burst": .25}
    assert STAGE_DEFINITIONS["j5_full_mixture"]["jammer_probabilities"]["pilot"] == .10
    assert STAGE_DEFINITIONS["j5_full_mixture"]["snr_db_range"] == [-2.0, 15.0]


def test_curriculum_and_fresh_results_cannot_mix() -> None:
    parent = {"stage_name": "o6_random_clean", "initialization_mode": "curriculum_resume"}
    validate_curriculum_parent("j1_weak_barrage", "curriculum_resume", parent)
    with pytest.raises(ValueError, match="initialization mode"):
        validate_curriculum_parent("j1_weak_barrage", "fresh_initialization_control", parent)


def test_catastrophic_forgetting_is_reported() -> None:
    result = catastrophic_forgetting(0.2, 0.3)
    assert result["previous_stage_validation_loss_before"] == .2
    assert result["previous_stage_validation_loss_after"] == .3
    assert result["relative_degradation"] == pytest.approx(.5)


def test_validation_matrix_adds_only_applicable_jammer_types() -> None:
    j1 = build_stage_validation_suite("j1_weak_barrage", 23, ["t"], ["v"])
    assert {x.get("jammer_type") for x in j1["scenarios"] if "jammer_type" in x} == {"barrage"}
    j5 = build_stage_validation_suite("j5_full_mixture", 23, ["t"], ["v"])
    assert {x.get("jammer_type") for x in j5["scenarios"] if "jammer_type" in x} == {"barrage", "narrowband", "burst", "pilot"}
