import torch

from speech_jscc.models.conv_conformer import ConvConformerJSCC


def tiny_model(**kwargs):
    return ConvConformerJSCC((8, 50, 1024), 1920, 8, d_model=32,
        encoder_conformer_blocks=1, decoder_conformer_blocks=1,
        num_attention_heads=4, ffn_expansion=2, convolution_kernel_size=7,
        layer_mixer_blocks=kwargs.pop("layer_mixer_blocks", 1), dropout=0.0,
        symbol_frames=30, complex_channels_per_symbol_frame=8, **kwargs)


def test_conv_conformer_exact_shapes_power_and_head_gradients():
    model = tiny_model()
    latent = torch.randn(1, 8, 50, 1024)
    state = torch.zeros(1, 8)
    symbols, aux = model.encoder(latent, state, return_aux=True)
    assert symbols.shape == (1, 1920) and symbols.is_complex()
    assert model.encoder.layer_channel_uses == (240,) * 8
    assert aux["temporal_feature_shape"].tolist() == [1, 8, 30, 32]
    assert torch.allclose(symbols.abs().square().mean(), torch.tensor(1.0), atol=1e-4)
    reconstruction = model.decoder(symbols, state)
    assert reconstruction.shape == latent.shape
    reconstruction.square().mean().backward()
    assert all(any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()) for head in model.encoder.symbol_heads)
    assert all(any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()) for head in model.decoder.reconstruction_heads)


def test_symbol_frame_product_is_enforced():
    try:
        tiny_model(complex_channels_per_symbol_frame=7)
    except (ValueError, TypeError) as error:
        assert "240" in str(error) or "symbol" in str(error)
    else:
        raise AssertionError("invalid symbol budget accepted")


def test_conditioning_changes_output_and_mixer_can_be_disabled():
    model = tiny_model(layer_mixer_blocks=0).eval()
    latent = torch.randn(1, 8, 50, 1024)
    zero = model.encoder(latent, torch.zeros(1, 8))
    one = model.encoder(latent, torch.ones(1, 8))
    assert not torch.allclose(zero, one)


def test_no_giant_output_linear_and_parameter_limit():
    model = ConvConformerJSCC((8, 50, 1024), 1920, 8)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total < 30_000_000
    assert all(getattr(module, "out_features", 0) != 8 * 50 * 1024 for module in model.modules())
