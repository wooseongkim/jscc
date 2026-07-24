from __future__ import annotations

import csv
import json
import math
import wave
from pathlib import Path

import pytest
import yaml

from eval_layer_ablation import run_layer_ablation


def _write_wav(path: Path, *, sample_rate: int = 8000, samples: int = 384, freq: float = 220.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    for index in range(samples):
        value = 0.25 * math.sin(2.0 * math.pi * freq * index / sample_rate)
        frames.extend(int(value * 32767.0).to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(frames))


def _config(path: Path) -> Path:
    payload = {
        "seed": 3,
        "device": "cpu",
        "model": {"layers": 3, "frames": 4, "latent_dim": 5},
        "codec": {
            "type": "mock",
            "waveform_samples": 384,
            "sample_rate": 8000,
            "seed": 11,
        },
        "layer_importance": {
            "estimator": "combined",
            "source_protocols": {"leave_one_out": 0.7, "prefix_marginal": 0.3},
            "metric_weights": {
                "si_sdr": 1.0,
                "waveform_snr": 0.5,
                "stft_l1": 0.5,
                "stoi": 1.0,
            },
            "normalization": "max",
            "output_weight_normalization": "mean_one",
            "minimum_weight": 0.05,
            "base_layer_cumulative_threshold": 0.70,
            "enforce_prefix_base_layers": True,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _manifest(path: Path, audio_paths: list[Path]) -> Path:
    rows = []
    for index, audio_path in enumerate(audio_paths):
        rows.append(
            {
                "utt_id": f"utt-{index}",
                "audio_path": str(audio_path),
                "speaker_id": "spk",
                "chapter_id": "chap",
                "split": "test",
                "text": "HELLO WORLD",
                "duration_sec": 0.048,
                "sample_rate": 8000,
                "num_samples": 384,
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_layer_ablation_mock_writes_reports_and_importance_artifact(tmp_path: Path) -> None:
    audio_paths = [tmp_path / f"audio_{index}.wav" for index in range(4)]
    for index, audio_path in enumerate(audio_paths):
        _write_wav(audio_path, freq=220.0 + 30.0 * index)
    config_path = _config(tmp_path / "config.yaml")
    manifest_path = _manifest(tmp_path / "test.jsonl", audio_paths)
    output_dir = tmp_path / "ablation"

    artifact = run_layer_ablation(
        config_path=config_path,
        manifest_path=manifest_path,
        split="test",
        output_dir=output_dir,
        max_items=4,
        batch_size=2,
        protocols=["leave_one_out", "prefix_keep"],
        mode="codec_only",
        device_name="cpu",
        seed=19,
    )

    assert artifact["schema_version"] == 1
    assert artifact["codec"]["representation_shape"] == [3, 4, 5]
    assert artifact["dataset"]["evaluated_items"] == 4
    assert artifact["importance"]["layer_weights_mean_one"] == pytest.approx(
        artifact["importance"]["layer_weights_mean_one"]
    )
    assert sorted(artifact["importance"]["layer_importance_order"]) == [0, 1, 2]
    assert (output_dir / "layer_importance.yaml").exists()
    assert (output_dir / "layer_importance.json").exists()
    assert (output_dir / "layer_ablation_report.md").exists()
    assert (output_dir / "recommended_config_snippet.yaml").exists()
    assert (output_dir / "layer_importance_bar.png").exists()
    assert (output_dir / "metric_delta_by_layer.png").exists()
    assert (output_dir / "prefix_metric_curves.png").exists()

    summary_rows = list(csv.DictReader((output_dir / "ablation_summary.csv").open()))
    prefix_rows = list(csv.DictReader((output_dir / "prefix_summary.csv").open()))
    per_sample_rows = list(csv.DictReader((output_dir / "ablation_per_sample.csv").open()))
    assert {row["protocol"] for row in summary_rows} >= {"all_layers", "leave_one_out"}
    leave_rows = [row for row in summary_rows if row["protocol"] == "leave_one_out"]
    assert len(leave_rows) >= 3
    assert {int(row["layer_index"]) for row in leave_rows} == {0, 1, 2}
    assert {int(row["prefix_k"]) for row in prefix_rows} == {1, 2, 3}
    assert len(per_sample_rows) >= 4 * (1 + 3 + 3)
    assert "si_sdr" in artifact["metrics"]["available"]
    assert "stoi" in artifact["metrics"]["unavailable"]


def test_eval_layer_ablation_config_matches_one_second_speechtokenizer_protocol() -> None:
    config = yaml.safe_load(Path("configs/eval_layer_ablation.yaml").read_text(encoding="utf-8"))

    assert config["codec"]["sample_rate"] == 16000
    assert config["codec"]["waveform_samples"] == 16000
    assert config["codec"]["n_q"] == 8
    assert config["model"]["layers"] == 8
    assert config["model"]["frames"] == 50
    assert config["model"]["latent_dim"] == 1024
