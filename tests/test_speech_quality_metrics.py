from pathlib import Path

import pytest


def test_word_error_rate_normalizes_case_and_punctuation():
    from speech_jscc.evaluation.speech_quality_metrics import compute_wer

    assert compute_wer("HELLO, world!", "hello world") == pytest.approx(0.0)
    assert compute_wer("hello world", "hello") == pytest.approx(0.5)


def test_estoi_rejects_a_silent_surrogate_when_backend_is_unavailable(monkeypatch):
    from speech_jscc.evaluation import speech_quality_metrics as metrics

    monkeypatch.setattr(metrics, "_import_pystoi", lambda: None)
    with pytest.raises(metrics.OptionalMetricDependencyError, match="pystoi"):
        metrics.compute_estoi([0.0, 0.1], [0.0, 0.1], 16_000)


def test_metric_options_preserve_raw_only_default():
    from speech_jscc.evaluation.speech_quality_metrics import MetricOptions

    options = MetricOptions.from_config({})
    assert not options.estoi_enabled
    assert not options.wer_enabled
    assert not options.visqol_enabled


def test_visqol_uses_speech_mode_cli_contract(monkeypatch, tmp_path):
    from speech_jscc.evaluation import speech_quality_metrics as metrics

    captured = {}

    class Result:
        stdout = "MOS-LQO:      3.25\n"
        stderr = ""

    def fake_run(command, **kwargs):
        captured["command"] = command
        return Result()

    monkeypatch.setattr(metrics.subprocess, "run", fake_run)
    assert metrics.compute_visqol(Path("reference.wav"), Path("estimate.wav"), "visqol") == pytest.approx(3.25)
    assert captured["command"] == [
        "visqol", "--reference", "reference.wav", "--degraded", "estimate.wav", "--speech_mode",
    ]
