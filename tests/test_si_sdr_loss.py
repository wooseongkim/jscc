from __future__ import annotations

import pytest
import torch

from speech_jscc.training.si_sdr_loss import negative_si_sdr_loss, si_sdr


def test_identical_waveform_has_high_si_sdr_and_finite_gradient() -> None:
    target = torch.randn(3, 1600)
    estimate = target.clone().requires_grad_()
    loss, diagnostics = negative_si_sdr_loss(estimate, target)
    assert diagnostics["raw_si_sdr_db"].min() > 70
    loss.backward()
    assert torch.isfinite(estimate.grad).all()


def test_si_sdr_is_scale_and_dc_offset_invariant() -> None:
    target = torch.randn(2, 800)
    estimate = target + .2 * torch.randn_like(target)
    assert si_sdr(estimate * 3.0 + 4.0, target).mean() == pytest.approx(si_sdr(estimate, target).mean(), abs=1e-5)


def test_noise_reduces_si_sdr_and_clipping_is_training_only() -> None:
    target = torch.randn(2, 800)
    noisy = target + .5 * torch.randn_like(target)
    raw = si_sdr(noisy, target)
    _, diagnostics = negative_si_sdr_loss(noisy, target, clip_db=1.0)
    assert raw.mean() < si_sdr(target, target).mean()
    assert diagnostics["loss_si_sdr_db"].abs().max() <= 1.0


def test_silence_is_finite_without_masking_or_alignment() -> None:
    target = torch.zeros(2, 400)
    estimate = torch.randn(2, 400, requires_grad=True)
    loss, diagnostics = negative_si_sdr_loss(estimate, target)
    assert torch.isfinite(loss) and torch.isfinite(diagnostics["raw_si_sdr_db"]).all()
    loss.backward()
    assert torch.isfinite(estimate.grad).all()
