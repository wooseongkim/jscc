import inspect

import pytest
import torch

from models import JammerEstimator as PublicJammerEstimator
from models.jammer_estimator import (
    JAMMER_TYPE_CLASSES,
    JammerEstimator,
    jammer_estimation_loss,
    load_jammer_estimator_checkpoint,
    save_jammer_estimator_checkpoint,
)


def _inputs(batch: int = 2, carriers: int = 7, symbols: int = 9):
    received = torch.randn(batch, carriers, symbols, dtype=torch.complex64)
    pilots = torch.randn(batch, carriers, symbols, dtype=torch.complex64)
    pilot_mask = torch.zeros(carriers, symbols, dtype=torch.bool)
    pilot_mask[:, 1] = True
    channel = torch.randn(batch, carriers, symbols, dtype=torch.complex64)
    noise_variance = torch.full((batch,), 0.1)
    return received, pilots, pilot_mask, channel, noise_variance


def test_estimator_returns_posterior_and_grid_mask_shapes():
    received, pilots, pilot_mask, channel, noise = _inputs()
    estimator = JammerEstimator(num_jammer_types=len(JAMMER_TYPE_CLASSES), hidden_dim=12)

    estimate = estimator(received, pilots, pilot_mask, channel, noise)

    assert estimate.posterior.shape == (2, len(JAMMER_TYPE_CLASSES))
    assert estimate.mask_logits.shape == received.shape
    assert estimate.mask_prob.shape == received.shape
    assert estimate.mask_ratio.shape == (2,)
    assert torch.allclose(estimate.posterior.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.all((estimate.mask_prob >= 0) & (estimate.mask_prob <= 1))


def test_estimator_is_exported_from_models_package():
    assert PublicJammerEstimator is JammerEstimator


def test_estimator_forward_never_accepts_true_jammer_labels():
    received, pilots, pilot_mask, channel, noise = _inputs()
    estimator = JammerEstimator(num_jammer_types=len(JAMMER_TYPE_CLASSES), hidden_dim=12)
    parameters = inspect.signature(estimator.forward).parameters

    assert "true_jammer_mask" not in parameters
    assert "true_jammer_type" not in parameters
    with pytest.raises(TypeError):
        estimator(
            received, pilots, pilot_mask, channel, noise,
            true_jammer_mask=torch.zeros_like(pilot_mask),
        )


def test_estimation_loss_uses_labels_outside_estimator_forward():
    received, pilots, pilot_mask, channel, noise = _inputs()
    estimator = JammerEstimator(num_jammer_types=len(JAMMER_TYPE_CLASSES), hidden_dim=12)
    estimate = estimator(received, pilots, pilot_mask, channel, noise)

    loss, components = jammer_estimation_loss(
        estimate,
        jammer_type=torch.tensor([0, 2]),
        jammer_mask=torch.zeros_like(estimate.mask_prob),
    )

    assert torch.isfinite(loss)
    assert set(components) == {"type_ce", "mask_bce", "mask_dice"}


def test_estimator_checkpoint_preserves_inference_contract():
    estimator = JammerEstimator(num_jammer_types=len(JAMMER_TYPE_CLASSES), hidden_dim=12)
    restored = load_jammer_estimator_checkpoint(
        save_jammer_estimator_checkpoint(estimator), torch.device("cpu")
    )
    assert restored.num_jammer_types == len(JAMMER_TYPE_CLASSES)
    assert restored.hidden_dim == 12


def test_estimator_checkpoint_preserves_explicit_five_class_vocabulary():
    vocabulary = ("no_jammer", "broadband_awgn", "subband", "burst", "tone")
    estimator = JammerEstimator(num_jammer_types=5, hidden_dim=12, jammer_type_classes=vocabulary)
    restored = load_jammer_estimator_checkpoint(
        save_jammer_estimator_checkpoint(estimator), torch.device("cpu")
    )
    assert restored.jammer_type_classes == vocabulary
