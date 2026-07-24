from __future__ import annotations

import torch

from eval_jamming import evaluate_paired_condition
from speech_jscc.metrics import (
    align_waveforms,
    compute_si_sdr,
    compute_stoi,
    summarize_audio_metrics,
)
from speech_jscc.codecs import MockContinuousCodec
from speech_jscc.models import SpeechJSCC


def test_compute_si_sdr_identical_waveform_is_high_and_finite() -> None:
    reference = torch.sin(torch.linspace(0.0, 8.0, 256))

    score = compute_si_sdr(reference, reference)

    assert score.shape == (1,)
    assert torch.isfinite(score).all()
    assert score.item() > 60.0


def test_compute_si_sdr_noisy_waveform_is_lower_than_identical() -> None:
    generator = torch.Generator().manual_seed(7)
    reference = torch.sin(torch.linspace(0.0, 8.0, 256))
    estimate = reference + 0.05 * torch.randn(reference.shape, generator=generator)

    clean_score = compute_si_sdr(reference, reference)
    noisy_score = compute_si_sdr(reference, estimate)

    assert noisy_score.item() < clean_score.item()


def test_align_waveforms_crops_to_shorter_length() -> None:
    reference = torch.arange(10, dtype=torch.float32)
    estimate = torch.arange(16, dtype=torch.float32).reshape(1, 1, 16)

    aligned_reference, aligned_estimate = align_waveforms(reference, estimate)

    assert aligned_reference.shape == (1, 10)
    assert aligned_estimate.shape == (1, 10)
    torch.testing.assert_close(aligned_reference[0], reference)
    torch.testing.assert_close(aligned_estimate[0], torch.arange(10, dtype=torch.float32))


def test_compute_stoi_is_optional_dependency() -> None:
    reference = torch.sin(torch.linspace(0.0, 80.0, 10_000)).repeat(2, 1)
    estimate = reference.clone()

    score = compute_stoi(reference, estimate, sample_rate=8000)

    if score is None:
        return
    assert score.shape == (2,)
    assert torch.isfinite(score).all()


def test_summarize_audio_metrics_returns_json_serializable_values() -> None:
    reference = torch.sin(torch.linspace(0.0, 8.0, 256))
    estimate = reference + 0.01

    metrics = summarize_audio_metrics(reference, estimate, sample_rate=8000)

    assert isinstance(metrics["si_sdr_db"], float)
    assert isinstance(metrics["si_sdr_db_per_example"], list)
    assert "stoi_available" in metrics
    assert metrics["stoi"] is None or isinstance(metrics["stoi"], float)


def test_eval_jamming_mock_row_includes_audio_quality_columns() -> None:
    device = torch.device("cpu")
    config = {
        "seed": 11,
        "model": {
            "layers": 2,
            "frames": 3,
            "latent_dim": 2,
            "channel_uses": 12,
            "channel_state_dim": 8,
            "hidden_dim": 16,
            "target_power": 1.0,
        },
        "codec": {"type": "mock", "waveform_samples": 96, "sample_rate": 8000},
        "channel": {
            "jammed_fraction": 0.25,
            "pilot_spacing": 3,
            "pilot_time_spacing": None,
        },
        "eval": {
            "batches": 1,
            "batch_size": 1,
            "layer_weights": [1.0, 1.0],
            "allocation_modes": ["uniform"],
            "refiner_modes": ["no_refiner"],
            "rule_gate_thresholds_db": [4.0],
            "transmitter_csi": True,
            "enable_stoi": False,
        },
    }
    codec = MockContinuousCodec(2, 3, 2, 96, seed=3).to(device)
    model = SpeechJSCC((2, 3, 2), 12, channel_state_dim=8, hidden_dim=16).to(device)
    model.eval()

    rows = evaluate_paired_condition(
        codec,
        model,
        learned_gate=None,
        config=config,
        device=device,
        modes=["uniform"],
        jammer_type="pilot",
        snr_value=8.0,
        jsr_value=0.0,
        equalizer="estimated",
        seed_base=123,
    )

    assert "si_sdr_db" in rows[0]
    assert isinstance(rows[0]["si_sdr_db"], float)
    assert rows[0]["stoi_available"] is False
