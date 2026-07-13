from __future__ import annotations

import json
import math
import subprocess
import sys
import wave
from pathlib import Path

import torch
import yaml


def _write_sine_wav(path: Path, sample_rate: int = 8000, samples: int = 384) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    for index in range(samples):
        value = 0.25 * math.sin(2.0 * math.pi * 220.0 * index / sample_rate)
        integer = int(max(-1.0, min(1.0, value)) * 32767.0)
        frames.extend(integer.to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(frames))


def _mock_infer_config(path: Path) -> Path:
    config = {
        "seed": 123,
        "device": "cpu",
        "model": {
            "layers": 2,
            "frames": 4,
            "latent_dim": 3,
            "channel_uses": [4, 4],
            "channel_state_dim": 8,
            "hidden_dim": 16,
            "target_power": 1.0,
        },
        "codec": {
            "type": "mock",
            "waveform_samples": 384,
            "sample_rate": 8000,
            "seed": 5,
        },
        "channel": {
            "snr_db": [8.0],
            "jsr_db": [0.0],
            "jammer_types": ["pilot"],
            "jammed_fraction": 0.25,
            "pilot_spacing": 2,
            "pilot_time_spacing": 2,
        },
        "eval": {
            "checkpoint": str(path.parent / "missing.pt"),
            "paired_seed": 456,
            "allocation_modes": ["uniform"],
            "layer_importance_order": [0, 1],
            "rule_gate_thresholds_db": [4.0],
            "learned_gate_hidden_dim": 8,
            "unreliable_fraction": 0.25,
        },
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_infer_jscc_wav_mock_smoke_exports_wav_metrics_and_tensors(tmp_path: Path) -> None:
    config_path = _mock_infer_config(tmp_path / "eval_mock.yaml")
    input_wav = tmp_path / "tx.wav"
    output_wav = tmp_path / "rx.wav"
    metrics_path = tmp_path / "metrics.json"
    tensors_path = tmp_path / "tensors.pt"
    _write_sine_wav(input_wav)

    completed = subprocess.run(
        [
            sys.executable,
            "infer_jscc_wav.py",
            "--config",
            str(config_path),
            "--input",
            str(input_wav),
            "--output",
            str(output_wav),
            "--jammer",
            "pilot",
            "--save-pt",
            str(tensors_path),
            "--metrics-json",
            str(metrics_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    assert output_wav.exists()
    with wave.open(str(output_wav), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getnframes() == 384

    stdout_metrics = json.loads(completed.stdout)
    file_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert stdout_metrics["output"] == str(output_wav)
    assert file_metrics["jammer"] == "pilot"
    assert stdout_metrics["adaptation_mode"] == "uniform"
    assert "latent_mse" in stdout_metrics
    assert "effective_sinr_db" in stdout_metrics
    assert isinstance(stdout_metrics["si_sdr_db"], float)
    assert stdout_metrics["stoi_available"] is False

    payload = torch.load(tensors_path, map_location="cpu", weights_only=True)
    assert "decoded_waveform" in payload
    assert "final_reconstruction" in payload
    assert payload["metrics"]["si_sdr_db"] == stdout_metrics["si_sdr_db"]
    assert payload["decoded_waveform"].shape == (1, 384)
