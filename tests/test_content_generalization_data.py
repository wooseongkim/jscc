from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from speech_jscc.diagnostics.content_generalization import (
    aggregate_dataset_statistics,
    build_content_subsets,
    build_content_validation_suite,
    parse_speaker_id,
)


def _write(path: Path, values: list[str]) -> Path:
    path.write_text("\n".join(json.dumps({"audio_path": value}) for value in values) + "\n")
    return path


def test_librispeech_speaker_id_and_unknown_fallback() -> None:
    assert parse_speaker_id("root/118/47824/118-47824-0023.flac") == "118"
    assert parse_speaker_id("arbitrary.wav") == "unknown"


def test_nested_subsets_and_speaker_validation_groups(tmp_path: Path) -> None:
    train_values = [f"train/{speaker}/1/{speaker}-1-{item:04d}.flac" for speaker in range(1, 33) for item in range(20)]
    valid_values = [f"valid/{speaker}/1/{speaker}-1-{item:04d}.flac" for speaker in range(101, 105) for item in range(10)]
    result = build_content_subsets(_write(tmp_path / "train.jsonl", train_values),
                                   _write(tmp_path / "valid.jsonl", valid_values), tmp_path / "cache", seed=23)
    s16, s64, s256, full = (result["subsets"][key] for key in ("16", "64", "256", "full"))
    assert set(s16["train_ids"]) < set(s64["train_ids"]) < set(s256["train_ids"]) < set(full["train_ids"])
    for subset in (s16, s64, s256, full):
        assert set(subset["train_ids"]).isdisjoint(subset["same_speaker_unseen_ids"])
        selected_speakers = {parse_speaker_id(path) for path in subset["train_ids"]}
        assert {parse_speaker_id(path) for path in subset["same_speaker_unseen_ids"]} <= selected_speakers
        assert {parse_speaker_id(path) for path in subset["unseen_speaker_ids"]}.isdisjoint(selected_speakers)
    hashes = {build_content_validation_suite(subset, 23)["validation_suite_hash"]
              for subset in (s16, s64, s256, full)}
    assert len(hashes) == 1


def test_test_manifest_or_cache_is_rejected(tmp_path: Path) -> None:
    train = _write(tmp_path / "test.jsonl", ["x/1/1/1-1-1.flac"])
    valid = _write(tmp_path / "valid.jsonl", ["x/2/1/2-1-1.flac"])
    with pytest.raises(ValueError, match="test data"):
        build_content_subsets(train, valid, tmp_path / "cache", seed=23)


def test_dataset_statistics_match_analytic_latents_and_preprocessing_hash() -> None:
    examples = [
        {"latent": torch.tensor([1.0, 3.0]), "duration_seconds": 1.0, "speaker_id": "a"},
        {"latent": torch.tensor([5.0, 7.0]), "duration_seconds": 2.0, "speaker_id": "b"},
    ]
    first = aggregate_dataset_statistics(examples, {"sample_rate": 16000, "waveform_samples": 16000})
    second = aggregate_dataset_statistics(examples, {"sample_rate": 8000, "waveform_samples": 16000})
    assert first["latent_mean"] == pytest.approx(4.0)
    assert first["latent_power"] == pytest.approx(21.0)
    assert first["latent_std"] == pytest.approx(torch.tensor([1., 3., 5., 7.]).std(unbiased=False).item())
    assert first["utterance_duration_mean_seconds"] == pytest.approx(1.5)
    assert first["speaker_count"] == 2
    assert first["preprocessing_hash"] != second["preprocessing_hash"]
