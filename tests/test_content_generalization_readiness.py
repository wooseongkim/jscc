from __future__ import annotations

import pytest

from speech_jscc.diagnostics.stage1_readiness import (
    classify_distribution_evidence,
    validate_j_stage_parent,
)


def test_o6_failure_and_j1_failed_parent_are_preserved() -> None:
    evidence = classify_distribution_evidence(
        {"gate": {"passed": False}}, {"gate": {"passed": False}, "provenance": {"parent_stage": "o6_random_clean"}}
    )
    assert evidence["o6_random_clean"] == "FAIL"
    assert evidence["j1_weak_barrage"] == "exploratory_failed_parent"
    assert evidence["curriculum_resume_allowed"] is False


def test_exploratory_j1_cannot_parent_j2() -> None:
    with pytest.raises(ValueError, match="exploratory_failed_parent"):
        validate_j_stage_parent("j2_moderate_barrage", {"stage_name": "j1_weak_barrage",
                                                          "evidence_status": "exploratory_failed_parent"})


def test_g3_pass_is_required_before_distribution_resume() -> None:
    evidence = classify_distribution_evidence({"gate": {"passed": False}}, None, g3_summary={"gate": {"passed": False}})
    assert evidence["g3_random_clean"] == "FAIL"
    assert not evidence["curriculum_resume_allowed"]
