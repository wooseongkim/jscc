import random

import pytest
import torch

from speech_jscc.codecs import MockContinuousCodec
from train_latent_jscc import (
    RepresentationSource,
    layer_weighted_latent_mse,
    sample_jammer_type,
    sample_uniform_db,
)


def test_layer_weighted_training_loss_and_layer_metrics():
    target = torch.zeros(2, 3, 4, 5)
    reconstruction = target.clone()
    reconstruction[:, 0] = 1.0
    reconstruction[:, 1] = 2.0
    reconstruction[:, 2] = 3.0

    loss, layer_mse = layer_weighted_latent_mse(
        reconstruction, target, torch.tensor([3.0, 2.0, 1.0])
    )

    torch.testing.assert_close(layer_mse, torch.tensor([1.0, 4.0, 9.0]))
    torch.testing.assert_close(loss, torch.tensor(20.0 / 6.0))


def test_sampled_db_values_respect_configured_range():
    values = sample_uniform_db(256, [-7.5, 3.25], torch.device("cpu"))
    assert torch.all(values >= -7.5)
    assert torch.all(values <= 3.25)


def test_jammer_probability_sampling_and_validation():
    random.seed(4)
    assert sample_jammer_type({"pilot": 1.0, "burst": 0.0}) == "pilot"
    with pytest.raises(ValueError):
        sample_jammer_type({"unknown": 1.0})
    with pytest.raises(ValueError):
        sample_jammer_type({"burst": 0.0})


def test_representation_source_loads_precomputed_latents(tmp_path):
    codec = MockContinuousCodec(2, 3, 4, 120)
    stored = torch.randn(7, 2, 3, 4)
    path = tmp_path / "latents.pt"
    torch.save({"representations": stored}, path)
    config = {
        "codec": {"waveform_samples": 120},
        "data": {"representations_path": str(path)},
    }
    source = RepresentationSource(config, codec, torch.device("cpu"))

    batch, waveform = source.next_batch(5)

    assert source.description == "precomputed"
    assert batch.shape == (5, 2, 3, 4)
    assert waveform is None

