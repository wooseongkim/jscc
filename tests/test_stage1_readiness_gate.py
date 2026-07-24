from __future__ import annotations

import json
from pathlib import Path

from speech_jscc.diagnostics.stage1_readiness import evaluate_readiness


def test_full_command_is_hidden_when_artifacts_are_missing(tmp_path: Path) -> None:
    result = evaluate_readiness(tmp_path, expected={})
    assert result["ready"] is False
    assert result["uniform_training_command"] is None


def test_synthetic_consistent_fixture_prints_command(tmp_path: Path) -> None:
    expected = {"manifest_hashes": {"train": "a", "validation": "b"}, "latent_cache_hash": "c",
                "validation_suite_hash": "d"}
    stages = ["o6_random_clean", "j1_weak_barrage", "j2_moderate_barrage", "j3_strong_barrage", "j4_mixed_sparse", "j5_full_mixture"]
    parent = None
    for stage in stages:
        root = tmp_path / stage; root.mkdir()
        provenance = {**expected, "stage_name": stage, "diagnostic_engine_version": "stage1_random_distribution_v1",
                      "initialization_mode": "curriculum_resume", "parent_stage": parent,
                      "resolved_stage_distribution": {}, "curriculum_history": stages[:stages.index(stage)+1]}
        (root / "summary.json").write_text(json.dumps({"provenance": provenance, "gate": {"passed": True}}))
        parent = stage
    result = evaluate_readiness(tmp_path, expected=expected, fixed_path_passed=True, tests_passed=True)
    assert result["ready"] is True
    assert "train_stage1_fixed_tx.py" in result["uniform_training_command"]


def test_stale_validation_hash_rejects_readiness(tmp_path: Path) -> None:
    root = tmp_path / "o6_random_clean"; root.mkdir()
    provenance = {"stage_name": "o6_random_clean", "validation_suite_hash": "old"}
    (root / "summary.json").write_text(json.dumps({"provenance": provenance, "gate": {"passed": True}}))
    result = evaluate_readiness(tmp_path, expected={"validation_suite_hash": "new"})
    assert not result["ready"]
    assert any("validation_suite_hash" in reason for reason in result["reasons"])
