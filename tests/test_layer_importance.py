from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from speech_jscc.layer_importance import load_layer_importance, resolve_layer_importance_config


def _artifact(path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "codec": {
            "type": "speechtokenizer",
            "representation_shape": [3, 4, 5],
            "sample_rate": 16000,
            "waveform_samples": 32000,
            "n_q": 3,
            "config_path": "config.json",
            "checkpoint_path": "SpeechTokenizer.pt",
        },
        "importance": {
            "layer_weights_mean_one": [1.5, 1.0, 0.5],
            "layer_weights_sum_one": [0.5, 0.3333333333, 0.1666666667],
            "layer_importance_order": [0, 1, 2],
            "base_layers": [0, 1],
        },
        "provenance": {"artifact_hash": "abc123"},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_load_layer_importance_validates_and_returns_structured_values(tmp_path: Path) -> None:
    artifact_path = _artifact(tmp_path / "importance.yaml")

    importance = load_layer_importance(
        artifact_path,
        expected_codec_type="speechtokenizer",
        expected_representation_shape=(3, 4, 5),
    )

    assert importance.layer_weights_mean_one == [1.5, 1.0, 0.5]
    assert importance.layer_weights_sum_one == pytest.approx([0.5, 1 / 3, 1 / 6])
    assert importance.layer_importance_order == [0, 1, 2]
    assert importance.base_layers == [0, 1]
    assert importance.metadata["path"] == str(artifact_path)


def test_load_layer_importance_strict_shape_mismatch_raises(tmp_path: Path) -> None:
    artifact_path = _artifact(tmp_path / "importance.yaml")

    with pytest.raises(ValueError, match="representation_shape"):
        load_layer_importance(
            artifact_path,
            expected_codec_type="speechtokenizer",
            expected_representation_shape=(4, 4, 5),
            strict=True,
        )


def test_resolve_layer_importance_config_applies_flags_with_precedence(tmp_path: Path) -> None:
    artifact_path = _artifact(tmp_path / "importance.yaml")
    config = {
        "model": {"layers": 3, "frames": 4, "latent_dim": 5},
        "codec": {"type": "speechtokenizer"},
        "train": {
            "layer_weights": [9.0, 9.0, 9.0],
            "layer_importance_order": [2, 1, 0],
        },
        "layer_importance": {
            "path": str(artifact_path),
            "strict_metadata": True,
            "apply_to_loss_weights": True,
            "apply_to_resource_order": False,
            "apply_to_base_layers": True,
        },
    }

    resolved = resolve_layer_importance_config(config, section="train")

    assert resolved.layer_weights == [1.5, 1.0, 0.5]
    assert resolved.layer_importance_order == [2, 1, 0]
    assert resolved.base_layers == [0, 1]
    assert resolved.artifact_path == str(artifact_path)
