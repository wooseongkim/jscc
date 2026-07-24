from __future__ import annotations

import argparse
import csv
import json
import math
import wave
from pathlib import Path

import pytest
import torch
import yaml

from eval_codec_only import align_for_metrics, run_evaluation, run_waveform_length_sweep, stft_l1, export_outlier_artifacts, waveform_metrics


def _write_sine_wav(path: Path, sample_rate: int = 8000, samples: int = 384) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    for index in range(samples):
        value = 0.2 * math.sin(2.0 * math.pi * 220.0 * index / sample_rate)
        frames.extend(int(value * 32767.0).to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(frames))


def _mock_config(path: Path) -> Path:
    config = {
        "seed": 7,
        "device": "cpu",
        "model": {"layers": 2, "frames": 4, "latent_dim": 3},
        "codec": {
            "type": "mock",
            "waveform_samples": 384,
            "sample_rate": 8000,
            "seed": 5,
        },
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
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
                "split": "valid",
                "text": "HELLO WORLD",
                "duration_sec": 0.048,
                "sample_rate": 8000,
                "num_samples": 384,
            }
        )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_codec_only_eval_mock_writes_outputs_and_noise_sweep(tmp_path: Path) -> None:
    audio_paths = [tmp_path / f"audio_{index}.wav" for index in range(3)]
    for audio_path in audio_paths:
        _write_sine_wav(audio_path)
    config_path = _mock_config(tmp_path / "config.yaml")
    manifest_path = _manifest(tmp_path / "valid.jsonl", audio_paths)
    output_dir = tmp_path / "codec_eval"

    summary = run_evaluation(
        config_path=config_path,
        manifest_path=manifest_path,
        split="valid",
        output_dir=output_dir,
        max_items=3,
        batch_size=2,
        enable_latent_noise_sweep=True,
        latent_noise_snr_db=[30.0, 10.0, 0.0],
        num_save_examples=2,
        device_name="cpu",
        seed=13,
        decode_mode="both",
        metric_align="peak_xcorr",
        max_lag_samples=8,
        snr_scale_match=True,
        metric_zero_mean=True,
        worst_k=2,
    )

    assert summary["wireless_channel_used"] is False
    assert summary["jscc_used"] is False
    assert summary["num_items_evaluated"] == 3
    assert "official_metrics_mean" in summary
    assert "continuous_sum_metrics_mean" in summary
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "per_utterance_metrics.csv").exists()
    assert (output_dir / "decode_comparison_metrics.csv").exists()
    assert (output_dir / "latent_noise_sweep_metrics.csv").exists()
    assert (output_dir / "examples" / "utt-0_original.wav").exists()
    assert (output_dir / "worst_samples" / "utt-0_original.wav").exists()
    assert (output_dir / "plots" / "waveform_snr_hist.png").exists()

    clean_rows = list(csv.DictReader((output_dir / "per_utterance_metrics.csv").open()))
    noise_rows = list(csv.DictReader((output_dir / "latent_noise_sweep_metrics.csv").open()))
    assert len(clean_rows) == 3
    assert len(noise_rows) == 9
    assert all(math.isfinite(float(row["si_sdr_db"])) for row in clean_rows)
    assert "official_si_sdr_db" in clean_rows[0]
    assert "continuous_sum_si_sdr_db" in clean_rows[0]
    assert "best_lag_samples" in clean_rows[0]
    assert set(row["latent_noise_snr_db"] for row in noise_rows) == {"30.0", "10.0", "0.0"}
    assert summary["latent_noise_sweep"]["0"]["mean_si_sdr_db"] is not None


def test_stft_l1_handles_short_waveforms() -> None:
    reference = torch.zeros(1, 12)
    estimate = torch.ones(1, 12) * 0.1

    value = stft_l1(reference, estimate)

    assert math.isfinite(value)
    assert value >= 0.0


def test_peak_xcorr_recovers_known_artificial_delay_and_lengths_match() -> None:
    reference = torch.zeros(1, 64)
    reference[:, 12:20] = 1.0
    estimate = torch.zeros(1, 64)
    estimate[:, 17:25] = 1.0

    aligned_ref, aligned_est, lags = align_for_metrics(
        reference,
        estimate,
        metric_align="peak_xcorr",
        max_lag_samples=10,
    )

    assert aligned_ref.shape == aligned_est.shape
    assert aligned_ref.shape[-1] == 59
    assert int(lags[0]) == -5


def test_silence_like_metrics_are_finite() -> None:
    reference = torch.zeros(1, 64)
    estimate = torch.zeros(1, 64)

    metrics = waveform_metrics(reference, estimate, metric_align="peak_xcorr")

    assert all(math.isfinite(float(value)) for value in metrics.values())


def test_snr_scale_match_improves_snr_without_changing_si_sdr() -> None:
    reference = torch.sin(torch.linspace(0, 6 * math.pi, 256)).unsqueeze(0)
    estimate = reference * 0.25

    without_scale = waveform_metrics(reference, estimate, snr_scale_match=False)
    with_scale = waveform_metrics(reference, estimate, snr_scale_match=True)

    assert with_scale["waveform_snr_db"] > without_scale["waveform_snr_db"] + 20.0
    assert with_scale["si_sdr_db"] == pytest.approx(without_scale["si_sdr_db"], abs=1e-5)


def test_outlier_export_writes_waveforms_and_plots(tmp_path: Path) -> None:
    reference = torch.sin(torch.linspace(0, 4 * math.pi, 128)).unsqueeze(0) * 0.2
    estimate = torch.roll(reference, shifts=4, dims=-1)

    export_outlier_artifacts(
        output_dir=tmp_path,
        utt_id="utt-outlier",
        original=reference[0],
        official_recon=estimate[0],
        continuous_sum_recon=estimate[0],
        sample_rate=8000,
        max_lag_samples=16,
    )

    out_dir = tmp_path / "outliers" / "utt-outlier"
    for name in (
        "original.wav",
        "official_recon.wav",
        "continuous_sum_recon.wav",
        "aligned_original.wav",
        "aligned_recon.wav",
        "error_waveform.wav",
        "waveform_overlay.png",
        "spectrogram_original.png",
        "spectrogram_recon.png",
        "spectrogram_error.png",
    ):
        assert (out_dir / name).exists()


def test_protocol_comparison_outputs_tables_and_diagnostic_columns(tmp_path: Path) -> None:
    audio_paths = [tmp_path / f"audio_{index}.wav" for index in range(2)]
    for audio_path in audio_paths:
        _write_sine_wav(audio_path)
    config_path = _mock_config(tmp_path / "config.yaml")
    manifest_path = _manifest(tmp_path / "valid.jsonl", audio_paths)
    output_dir = tmp_path / "codec_eval_protocols"

    summary = run_evaluation(
        config_path=config_path,
        manifest_path=manifest_path,
        split="valid",
        output_dir=output_dir,
        max_items=2,
        batch_size=2,
        enable_latent_noise_sweep=False,
        latent_noise_snr_db=[],
        num_save_examples=1,
        device_name="cpu",
        seed=21,
        decode_mode="both",
        metric_align="peak_xcorr",
        max_lag_samples=8,
        snr_scale_match=True,
        metric_zero_mean=True,
        worst_k=1,
        protocol_comparison=True,
    )

    assert "baseline_protocol_comparison" in summary
    assert len(summary["baseline_protocol_comparison"]) == 24
    assert (output_dir / "baseline_protocol_comparison.csv").exists()
    assert (output_dir / "baseline_protocol_comparison.md").exists()
    assert (output_dir / "baseline_protocol_comparison.pdf").exists()
    assert (output_dir / "codec_only_baseline_report.md").exists()
    protocol_rows = list(csv.DictReader((output_dir / "baseline_protocol_comparison.csv").open()))
    assert {row["protocol_name"] for row in protocol_rows} == {"A", "B", "C", "D"}
    assert {row["metric_align"] for row in protocol_rows} == {"none", "peak_xcorr"}
    assert {row["snr_scale_match"] for row in protocol_rows} == {"False", "True"}
    assert {row["metric_zero_mean"] for row in protocol_rows} == {"True"}
    assert "waveform_samples" in protocol_rows[0]
    assert "duration_sec" in protocol_rows[0]
    assert "n" in protocol_rows[0]
    assert "outliers" in protocol_rows[0]
    rows = list(csv.DictReader((output_dir / "per_utterance_metrics.csv").open()))
    for key in (
        "duration_before_crop",
        "crop_start_sample",
        "crop_end_sample",
        "speech_activity_ratio",
        "leading_silence_ratio",
        "trailing_silence_ratio",
        "si_sdr_no_align",
        "si_sdr_aligned",
        "si_sdr_gain_from_alignment",
        "waveform_snr_no_align",
        "waveform_snr_aligned",
        "extreme_outlier",
    ):
        assert key in rows[0]


def test_waveform_length_sweep_writes_all_lengths_and_protocols(tmp_path: Path) -> None:
    audio_paths = [tmp_path / f"audio_{index}.wav" for index in range(2)]
    for audio_path in audio_paths:
        _write_sine_wav(audio_path, samples=480)
    config_path = _mock_config(tmp_path / "config.yaml")
    manifest_path = _manifest(tmp_path / "valid.jsonl", audio_paths)
    output_dir = tmp_path / "length_sweep"
    args = argparse.Namespace(
        config=str(config_path),
        manifest=str(manifest_path),
        split="valid",
        max_items=2,
        batch_size=2,
        enable_latent_noise_sweep=False,
        latent_noise_snr_db=[],
        output_dir=str(output_dir),
        num_save_examples=0,
        device="cpu",
        seed=31,
        decode_mode="both",
        metric_align="peak_xcorr",
        max_lag_samples=8,
        snr_scale_match=True,
        metric_zero_mean=True,
        worst_k=1,
        silence_rms_threshold=1e-4,
        lag_outlier_threshold=None,
        waveform_samples=[160, 320, 480],
    )

    summary = run_waveform_length_sweep(args)

    assert set(summary["lengths"]) == {"160", "320", "480"}
    rows = list(csv.DictReader((output_dir / "waveform_length_protocol_comparison.csv").open()))
    assert {row["waveform_samples"] for row in rows} == {"160", "320", "480"}
    assert {row["protocol_name"] for row in rows} == {"A", "B", "C", "D"}
    assert {"metric", "mean", "median", "std", "p10", "p90", "min", "max", "n", "outliers"} <= set(rows[0])


def test_codec_only_baseline_config_uses_recommended_protocol() -> None:
    config = yaml.safe_load(Path("configs/codec_only_baseline.yaml").read_text(encoding="utf-8"))

    assert config["codec"]["waveform_samples"] == 32000
    assert config["eval"]["metric_align"] == "peak_xcorr"
    assert config["eval"]["snr_scale_match"] is True
    assert config["eval"]["metric_zero_mean"] is True
    assert config["eval"]["max_items"] == 100
    assert config["eval"]["batch_size"] == 4
    assert config["eval"]["outlier_threshold_si_sdr_db"] == -10.0
