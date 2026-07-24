import torch

from speech_jscc.models.conv_conformer import ConformerBlock


def test_conformer_block_preserves_shape_and_backpropagates():
    block = ConformerBlock(32, 4, 2, 7, 0.0)
    value = torch.randn(2, 11, 32, requires_grad=True)
    output = block(value)
    assert output.shape == value.shape
    output.square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0


def test_conformer_temporal_convolution_preserves_even_config_rejected():
    try:
        ConformerBlock(32, 4, 2, 8, 0.0)
    except ValueError as error:
        assert "odd" in str(error)
    else:
        raise AssertionError("even kernel must be rejected")
