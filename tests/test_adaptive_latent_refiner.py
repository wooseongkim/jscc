import torch

from models.adaptive_latent_refiner import (
    MoEAdaptiveLatentRefiner,
    load_adaptive_latent_refiner_checkpoint,
    save_adaptive_latent_refiner_checkpoint,
)


def test_moe_refiner_preserves_latent_at_identity_initialization():
    raw = torch.randn(2, 3, 5, 4)
    state = torch.randn(2, 7)
    mask = torch.rand(2, 6, 8)
    posterior = torch.softmax(torch.randn(2, 6), dim=-1)
    refiner = MoEAdaptiveLatentRefiner(
        representation_shape=(3, 5, 4),
        channel_state_dim=7,
        num_experts=6,
        hidden_dim=12,
    )

    output = refiner(raw, state, mask, posterior)

    assert output.shape == raw.shape
    assert torch.allclose(output, raw, atol=1e-7, rtol=0)


def test_moe_refiner_requires_posterior_per_expert():
    refiner = MoEAdaptiveLatentRefiner(
        representation_shape=(2, 3, 4), channel_state_dim=5, num_experts=6,
    )
    with torch.no_grad():
        with __import__("pytest").raises(ValueError, match="jammer_posterior"):
            refiner(torch.randn(1, 2, 3, 4), torch.randn(1, 5), torch.rand(1, 2, 4), torch.rand(1, 5))


def test_refiner_checkpoint_restores_architecture():
    refiner = MoEAdaptiveLatentRefiner(
        representation_shape=(2, 3, 4), channel_state_dim=5, num_experts=6, hidden_dim=13,
    )
    restored = load_adaptive_latent_refiner_checkpoint(
        save_adaptive_latent_refiner_checkpoint(refiner), torch.device("cpu")
    )
    assert restored.representation_shape == (2, 3, 4)
    assert restored.num_experts == 6
    assert restored.hidden_dim == 13
