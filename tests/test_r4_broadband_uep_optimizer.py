"""Unit tests for the simulation-based broadband UEP optimizer core.

These tests intentionally exercise optimizer mathematics and protocol boundaries
without mocking a waveform score.  CUDA waveform smoke is performed by the
user-facing CLI after these deterministic checks pass.
"""
from __future__ import annotations

import json

import pytest

from speech_jscc.evaluation.r4_broadband_uep_optimizer import (
    build_selected_profiles_artifact,
    candidate_hash,
    enumerate_feasible_repetitions,
    make_candidate,
    load_selected_profiles,
    objective_from_summary,
    pareto_front,
    per_re_power,
    propose_power_transfers,
    propose_repetition_moves,
    rank_by_objective,
    sample_random_candidates,
    select_profiles,
    total_energy,
    total_re,
    validate_repetition,
    validate_search_split,
)
from channels.physical_ofdm import NR_LIKE_R4
from channels.r4_uep_allocator import UEPProfile, allocate_r4_uep


def _summary(**overrides):
    result = {
        "weighted_mean_delta_si_sdr_db": 0.4,
        "clean_cost_db": 0.1,
        "p5_delta_si_sdr_db": -0.4,
        "severe_tail_increase_fraction": 0.0,
        "stft_l1_delta": 0.0,
        "mean_delta_si_sdr_db": 0.4,
        "absolute_catastrophic_reduction_fraction": 0.0,
    }
    result.update(overrides)
    return result


def test_feasible_repetitions_have_exact_integer_budget_and_include_u0():
    repetitions = list(enumerate_feasible_repetitions())
    assert (3,) * 8 in repetitions
    assert repetitions
    for value in repetitions:
        assert validate_repetition(value) == value
        assert all(item in {1, 2, 3, 4, 5} for item in value)
        assert sum(value) == 24
        assert total_re(value) == 5760


def test_invalid_repetition_is_rejected():
    with pytest.raises(ValueError, match="sum"):
        validate_repetition((3,) * 7 + (2,))
    with pytest.raises(ValueError, match="between 1 and 5"):
        validate_repetition((0, 4, 4, 4, 3, 3, 3, 3))


def test_uniform_candidate_has_simplex_power_per_re_power_and_exact_energy():
    candidate = make_candidate((3,) * 8, [0.0] * 8)
    assert candidate.power_share == pytest.approx((0.125,) * 8)
    assert candidate.per_re_power == pytest.approx((1.0,) * 8)
    assert total_energy(candidate.repetition, candidate.per_re_power) == pytest.approx(5760.0)
    assert per_re_power((3,) * 8, (0.125,) * 8) == pytest.approx((1.0,) * 8)


def test_candidate_hash_is_deterministic_and_changes_with_decision_variable():
    first = make_candidate((3,) * 8, [0.0] * 8)
    same = make_candidate((3,) * 8, [0.0] * 8)
    different = make_candidate((4, 2, 3, 3, 3, 3, 3, 3), [0.0] * 8)
    assert candidate_hash(first) == candidate_hash(same)
    assert candidate_hash(first) != candidate_hash(different)


def test_random_generation_is_seeded_and_has_no_layer_priority_parameter():
    first = sample_random_candidates(count=12, seed=77)
    second = sample_random_candidates(count=12, seed=77)
    assert [candidate_hash(item) for item in first] == [candidate_hash(item) for item in second]
    assert "priority" not in sample_random_candidates.__code__.co_varnames
    assert "P2" not in sample_random_candidates.__code__.co_consts
    assert "RP2" not in sample_random_candidates.__code__.co_consts


def test_local_mutations_preserve_repetition_and_power_constraints():
    candidate = make_candidate((3,) * 8, [0.0] * 8)
    repetition_moves = propose_repetition_moves(candidate.repetition)
    assert repetition_moves
    assert all(sum(item) == 24 and all(1 <= value <= 5 for value in item) for item in repetition_moves)
    power_moves = propose_power_transfers(candidate.power_share, deltas=(0.05,))
    assert power_moves
    assert all(sum(item) == pytest.approx(1.0) and all(value > 0 for value in item) for item in power_moves)


def test_objective_uses_paired_u0_delta_and_only_penalizes_excess_clean_cost():
    base = objective_from_summary(_summary())
    worse_clean = objective_from_summary(_summary(clean_cost_db=0.5))
    worse_tail = objective_from_summary(_summary(p5_delta_si_sdr_db=-1.5))
    assert base["score"] == pytest.approx(0.4)
    assert worse_clean["clean_cost_penalty"] > 0
    assert worse_clean["score"] < base["score"]
    assert worse_tail["tail_risk_penalty"] > 0


def test_pareto_and_profile_selection_follow_constraints():
    candidates = [
        {"candidate_id": "best", "summary": _summary(mean_delta_si_sdr_db=0.8, weighted_mean_delta_si_sdr_db=0.8)},
        {"candidate_id": "stable", "summary": _summary(mean_delta_si_sdr_db=0.6, weighted_mean_delta_si_sdr_db=0.6)},
        {"candidate_id": "tail", "summary": _summary(mean_delta_si_sdr_db=0.9, weighted_mean_delta_si_sdr_db=0.9, clean_cost_db=0.1, p5_delta_si_sdr_db=-2.0)},
    ]
    front = pareto_front(candidates)
    selected = select_profiles(candidates, front)
    assert selected["x_best"]["candidate_id"] == "best"
    assert selected["x_stable"]["candidate_id"] == "best"
    assert selected["x_aggressive"]["candidate_id"] == "tail"


def test_aggressive_profile_requires_positive_gain_or_absolute_tail_reduction():
    records = [
        {"candidate_id": "tail_reduction", "summary": _summary(
            mean_delta_si_sdr_db=-0.2, weighted_mean_delta_si_sdr_db=-0.2,
            clean_cost_db=0.1, absolute_catastrophic_reduction_fraction=0.1,
        )},
        {"candidate_id": "positive_gain", "summary": _summary(
            mean_delta_si_sdr_db=0.3, weighted_mean_delta_si_sdr_db=0.3,
            clean_cost_db=0.2,
        )},
    ]
    selected = select_profiles(records, pareto_front(records))
    assert selected["x_aggressive"]["candidate_id"] == "positive_gain"


def test_search_is_expanded_only_and_selection_artifact_has_no_legacy_data():
    assert validate_search_split("expanded_selection") == "expanded_selection"
    with pytest.raises(ValueError, match="expanded_selection"):
        validate_search_split("legacy_final")
    artifact = build_selected_profiles_artifact(
        checkpoint={"path": "medium.pt", "sha256": "abc"},
        selected={"x_best": None, "x_stable": None, "x_aggressive": None},
        candidate_count=0,
    )
    rendered = json.dumps(artifact).lower()
    assert artifact["selection_split"] == "expanded_selection"
    assert artifact["selection_uses_legacy_metrics"] is False
    assert "legacy_final" not in rendered
    assert "legacy_artifact" not in rendered
    for entry in artifact["selected"].values():
        assert {"status", "reason", "mean_delta_si_sdr", "scalar_objective", "clean_cost", "p5_delta", "relative_tail_fraction", "absolute_catastrophic_fraction", "selected_for_final_eval"} <= set(entry)


def test_final_evaluation_requires_immutable_optimizer_selection(tmp_path):
    path = tmp_path / "selected_profiles.json"
    artifact = build_selected_profiles_artifact(
        checkpoint={"path": "medium.pt", "sha256": "abc"},
        selected={"x_best": None, "x_stable": None, "x_aggressive": None},
        candidate_count=0,
    )
    path.write_text(json.dumps(artifact))
    before = path.read_bytes()
    loaded = load_selected_profiles(path)
    assert loaded["artifact_type"] == "r4_broadband_uep_optimization_selection"
    assert path.read_bytes() == before
    with pytest.raises(FileNotFoundError):
        load_selected_profiles(tmp_path / "missing.json")


def test_negative_best_is_not_marked_for_final_evaluation():
    candidate = {"candidate_id": "negative", "summary": _summary(mean_delta_si_sdr_db=-0.1, weighted_mean_delta_si_sdr_db=-0.1)}
    candidate["objective"] = objective_from_summary(candidate["summary"])
    artifact = build_selected_profiles_artifact(
        checkpoint={"path": "medium.pt", "sha256": "abc"},
        selected={"x_best": candidate, "x_stable": None, "x_aggressive": None},
        candidate_count=1,
    )
    best = artifact["selected"]["x_best"]
    assert best["status"] == "NEGATIVE_OBJECTIVE"
    assert best["selected_for_final_eval"] is False


def test_u0_zero_objective_is_selected_when_all_nonuniform_candidates_are_negative():
    u0 = {"candidate_id": "U0", "summary": _summary(
        mean_delta_si_sdr_db=0.0, weighted_mean_delta_si_sdr_db=0.0,
        clean_cost_db=0.0, p5_delta_si_sdr_db=0.0,
    )}
    negative = {"candidate_id": "negative", "summary": _summary(
        mean_delta_si_sdr_db=-0.1, weighted_mean_delta_si_sdr_db=-0.1,
    )}
    selected = select_profiles([u0, negative], pareto_front([u0, negative]))
    assert selected["x_best"]["candidate_id"] == "U0"
    assert selected["x_stable"]["candidate_id"] == "U0"


def test_objective_ranking_includes_records_added_after_initial_screening():
    screened = {"candidate_id": "screened", "summary": _summary(mean_delta_si_sdr_db=0.1, weighted_mean_delta_si_sdr_db=0.1)}
    refined = {"candidate_id": "refined", "summary": _summary(mean_delta_si_sdr_db=0.8, weighted_mean_delta_si_sdr_db=0.8)}
    for record in (screened, refined):
        record["objective"] = objective_from_summary(record["summary"])
    ranking = rank_by_objective([screened, refined])
    assert [record["candidate_id"] for record in ranking] == ["refined", "screened"]


def test_generic_profile_preserves_variable_copy_budget_and_formula_power():
    profile = UEPProfile(
        "optimizer_candidate",
        (5, 4, 3, 3, 3, 2, 2, 2),
        power_share=(0.20, 0.16, 0.13, 0.12, 0.11, 0.10, 0.09, 0.09),
    )
    allocation = allocate_r4_uep(
        profile=NR_LIKE_R4,
        tx_tti=0,
        report=None,
        layer_importance_order=list(range(8)),
        uep_profile=profile,
    )
    expected = []
    for repeats in profile.repetition:
        expected.extend([repeats] * 240)
    assert allocation.copy_count_per_source.tolist() == expected
    assert allocation.selected_re_count == 5760
    assert allocation.power_source_order.sum().item() == pytest.approx(5760.0)
    assert profile.per_re_layer_power().tolist() == pytest.approx([24 * p / r for r, p in zip(profile.repetition, profile.power_share)])
