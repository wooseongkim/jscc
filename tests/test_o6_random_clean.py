from __future__ import annotations

from pathlib import Path

import pytest

from speech_jscc.diagnostics.random_distribution import (
    ENGINE_VERSION,
    SeedDeriver,
    build_subset_manifest,
    build_validation_suite,
    validate_external_steps,
)
from diagnose_stage1_random_distribution import _latent_summary
import torch


def _manifest(path: Path, prefix: str, count: int) -> Path:
    path.write_text("\n".join(f'{{"audio_path":"{prefix}-{i}.flac"}}' for i in range(count)) + "\n")
    return path


def test_o6_subsets_are_deterministic_disjoint_and_never_test(tmp_path: Path) -> None:
    train = _manifest(tmp_path / "train.jsonl", "train", 20)
    valid = _manifest(tmp_path / "valid.jsonl", "valid", 10)
    first = build_subset_manifest(train, valid, tmp_path / "cache", train_count=16, validation_count=8, seed=23)
    second = build_subset_manifest(train, valid, tmp_path / "cache", train_count=16, validation_count=8, seed=23)
    assert first == second
    assert len(first["train_utterance_ids"]) == 16
    assert len(first["validation_utterance_ids"]) == 8
    assert set(first["train_utterance_ids"]).isdisjoint(first["validation_utterance_ids"])
    assert first["diagnostic_engine_version"] == ENGINE_VERSION
    with pytest.raises(ValueError, match="test data"):
        build_subset_manifest(tmp_path / "test.jsonl", valid, tmp_path / "cache", seed=23)


def test_train_channel_and_noise_seeds_change_by_step() -> None:
    derive = SeedDeriver(23)
    assert derive.seed("train_channel", 1) != derive.seed("train_channel", 2)
    assert derive.seed("train_noise", 1) != derive.seed("train_noise", 2)
    assert derive.seed("train_channel", 1) != derive.seed("train_noise", 1)


def test_validation_suite_is_fixed_and_contains_v1_v2_snr_slices() -> None:
    first = build_validation_suite(23, ["t0", "t1"], ["v0", "v1"])
    second = build_validation_suite(23, ["t0", "t1"], ["v0", "v1"])
    assert first == second
    assert {item["suite"] for item in first["scenarios"]} == {"V1", "V2", "V3"}
    assert {item["snr_db"] for item in first["scenarios"] if item["suite"] == "V3"} == {5.0, 10.0, 15.0}


def test_long_runs_require_explicit_external_acknowledgement() -> None:
    validate_external_steps(5, allow_long_run=False)
    with pytest.raises(ValueError, match="allow-long-run"):
        validate_external_steps(1000, allow_long_run=False)


def test_latent_summary_uses_actual_zero_metric_schema() -> None:
    target = torch.ones(1, 2, 1, 2)
    summary = _latent_summary(torch.zeros_like(target), target, loss=1.0)
    assert summary["zero_predictor_loss"] == pytest.approx(1.0)
    assert summary["relative_improvement_over_zero"] == pytest.approx(0.0)
