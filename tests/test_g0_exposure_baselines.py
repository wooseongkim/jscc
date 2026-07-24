import pytest
import torch

from speech_jscc.diagnostics.g0_exposure import compute_train_baselines, evaluate_baselines


def test_global_layerwise_and_speaker_means_are_analytic() -> None:
    a = torch.tensor([[[1., 3.]], [[10., 14.]]])
    b = torch.tensor([[[3., 5.]], [[14., 18.]]])
    result = compute_train_baselines([("x/1/1/1-1-1.flac", a), ("x/1/1/1-1-2.flac", b)])
    torch.testing.assert_close(result["global_mean"], (a + b) / 2)
    assert result["layerwise_mean"][0, 0, 0] == pytest.approx(3.0)
    assert result["layerwise_mean"][1, 0, 0] == pytest.approx(14.0)
    assert "1" in result["speaker_means"]


def test_speaker_baseline_requires_two_samples_and_is_unavailable_for_unseen() -> None:
    latent = torch.ones(2, 1, 2)
    baselines = compute_train_baselines([("x/1/1/1-1-1.flac", latent), ("x/2/1/2-1-1.flac", latent * 2)])
    target = torch.stack([latent]).unsqueeze(0) if False else latent.unsqueeze(0)
    result = evaluate_baselines(target, ["x/9/1/9-1-1.flac"], baselines, group="unseen")
    assert result["speaker_conditional_mean"]["available"] is False
    assert result["zero"]["aggregate"]["normalized_mse"] == pytest.approx(1.0)
