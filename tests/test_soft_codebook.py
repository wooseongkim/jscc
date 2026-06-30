import torch

from models.soft_codebook import (
    CodecRepresentationLoss,
    SoftCodebook,
    continuous_latent_loss,
    soft_codebook_projection,
    top_k_token_accuracy,
)


def test_shared_soft_codebook_projection_is_embedding_expectation():
    codebook = torch.tensor([[0.0, 0.0], [2.0, 4.0]])
    logits = torch.zeros(2, 3, 2)

    projected, probabilities = soft_codebook_projection(
        logits, codebook, return_probabilities=True
    )

    torch.testing.assert_close(probabilities, torch.full_like(probabilities, 0.5))
    torch.testing.assert_close(projected, torch.tensor([1.0, 2.0]).expand(2, 3, 2))


def test_layer_specific_projection_shape_and_gradient():
    batch, layers, frames, tokens, dimension = 2, 3, 4, 5, 6
    codebook = torch.randn(layers, tokens, dimension)
    logits = torch.randn(batch, layers, frames, tokens, requires_grad=True)
    projector = SoftCodebook(codebook, temperature=0.7)

    projected = projector(logits)
    projected.square().mean().backward()

    assert projected.shape == (batch, layers, frames, dimension)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_lower_temperature_sharpens_probabilities():
    codebook = torch.eye(3)
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    _, warm = soft_codebook_projection(logits, codebook, 2.0, return_probabilities=True)
    _, cold = soft_codebook_projection(logits, codebook, 0.25, return_probabilities=True)

    assert cold.max() > warm.max()


def test_continuous_latent_loss_supports_layer_weights():
    target = torch.zeros(2, 2, 3, 4)
    reconstruction = target.clone()
    reconstruction[:, 0] = 1.0
    reconstruction[:, 1] = 2.0

    loss = continuous_latent_loss(reconstruction, target, layer_weights=[3.0, 1.0])
    per_example = continuous_latent_loss(
        reconstruction, target, layer_weights=[3.0, 1.0], reduction="none"
    )

    torch.testing.assert_close(loss, torch.tensor(1.75))
    torch.testing.assert_close(per_example, torch.full((2,), 1.75))


def test_representation_objective_has_continuous_and_soft_modes():
    target = torch.randn(2, 2, 3, 4)
    continuous = CodecRepresentationLoss("continuous")
    torch.testing.assert_close(continuous(target, target), torch.tensor(0.0))

    codebook = torch.randn(2, 5, 4)
    logits = torch.randn(2, 2, 3, 5)
    soft = CodecRepresentationLoss("soft_codebook", codebook=codebook, temperature=0.8)
    reconstruction = soft.reconstruct(logits)
    loss = soft(logits, target)

    assert reconstruction.shape == target.shape
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_top_k_accuracy_is_analysis_metric():
    logits = torch.tensor(
        [
            [[8.0, 1.0, 0.0], [0.0, 5.0, 4.0]],
            [[0.0, 1.0, 7.0], [6.0, 5.0, 0.0]],
        ]
    )
    targets = torch.tensor([[0, 2], [1, -100]])

    top1 = top_k_token_accuracy(logits, targets, 1, ignore_index=-100)
    top2 = top_k_token_accuracy(logits, targets, 2, ignore_index=-100)

    torch.testing.assert_close(top1, torch.tensor(1.0 / 3.0))
    torch.testing.assert_close(top2, torch.tensor(1.0))
