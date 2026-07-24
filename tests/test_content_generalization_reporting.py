from __future__ import annotations

import pytest

from speech_jscc.diagnostics.content_generalization import content_group_gate, ladder_decision


def _metrics(improvement=.06, power=.02, cosine=.1, correlation=.1, finite=True):
    return {"aggregate": {"relative_improvement_over_zero": improvement, "power_ratio": power,
                           "cosine_similarity": cosine, "pearson_correlation": correlation, "finite": finite}}


def test_group_gate_requires_all_metrics() -> None:
    assert content_group_gate(_metrics())["passed"]
    assert not content_group_gate(_metrics(improvement=.049))["passed"]
    assert not content_group_gate(_metrics(power=.009))["passed"]


def test_failed_small_subset_escalates_but_failed_full_stops_stage() -> None:
    assert ladder_decision("g0_direct", "16", passed=False) == "next_subset"
    assert ladder_decision("g0_direct", "full", passed=False) == "stop_first_failing_stage"
    assert ladder_decision("g0_direct", "64", passed=True) == "next_stage"


def test_unknown_stage_or_subset_is_rejected() -> None:
    with pytest.raises(ValueError): ladder_decision("bad", "16", False)
    with pytest.raises(ValueError): ladder_decision("g0_direct", "bad", False)
