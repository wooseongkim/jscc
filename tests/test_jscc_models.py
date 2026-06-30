import torch

from models.jscc_decoder import JSCCDecoder
from models.jscc_encoder import JSCCEncoder, deterministic_layer_gates


def test_flat_encoder_decoder_shapes_and_power():
    batch, shape, channel_uses = 5, (3, 7, 4), 24
    encoder = JSCCEncoder(shape, channel_uses, channel_state_dim=3, hidden_dim=32, target_power=1.25)
    decoder = JSCCDecoder(shape, channel_uses, channel_state_dim=3, hidden_dim=32)
    latent = torch.randn(batch, *shape)
    state = torch.randn(batch, 3)

    symbols = encoder(latent, state)
    reconstruction = decoder(symbols, state)

    assert symbols.shape == (batch, channel_uses)
    assert symbols.is_complex()
    assert reconstruction.shape == latent.shape
    torch.testing.assert_close(
        symbols.abs().square().mean(dim=1),
        torch.full((batch,), 1.25),
        rtol=1e-5,
        atol=1e-5,
    )


def test_ofdm_encoder_decoder_shapes_and_power():
    batch, shape, grid = 4, (4, 6, 3), (8, 5)
    encoder = JSCCEncoder(shape, grid, hidden_dim=24, target_power=0.75)
    decoder = JSCCDecoder(shape, grid, hidden_dim=24)
    latent = torch.randn(batch, *shape)
    state = torch.randn(batch, 2)

    symbols = encoder(latent, state)
    reconstruction = decoder(symbols, state)

    assert symbols.shape == (batch, *grid)
    assert reconstruction.shape == latent.shape
    torch.testing.assert_close(
        symbols.abs().square().mean(dim=(1, 2)),
        torch.full((batch,), 0.75),
        rtol=1e-5,
        atol=1e-5,
    )


def test_deterministic_prefix_layer_gating():
    state = torch.tensor([[-1.0, 0.0], [0.25, 0.0], [2.0, 0.0]])
    gates = deterministic_layer_gates(state, 4, thresholds=[0.0, 0.5, 1.0])
    expected = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
    )
    torch.testing.assert_close(gates, expected)


def test_encoder_reports_gates_and_layer_power_fractions():
    encoder = JSCCEncoder(
        (3, 5, 2),
        18,
        hidden_dim=20,
        gate_thresholds=[0.0, 0.5],
        layer_power_allocation=[1.0, 2.0, 3.0],
    )
    latent = torch.randn(2, 3, 5, 2)
    state = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])

    symbols, auxiliary = encoder(latent, state, return_aux=True)

    torch.testing.assert_close(auxiliary["layer_gates"][0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(auxiliary["layer_power_fractions"][0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(
        auxiliary["layer_power_fractions"][1], torch.tensor([1.0, 2.0, 3.0]) / 6.0
    )
    torch.testing.assert_close(symbols.abs().square().mean(1), torch.ones(2), rtol=1e-5, atol=1e-5)


def test_decoder_can_apply_explicit_output_gates():
    shape = (3, 4, 2)
    decoder = JSCCDecoder(shape, 12, hidden_dim=16)
    received = torch.complex(torch.randn(2, 12), torch.randn(2, 12))
    state = torch.zeros(2, 2)
    gates = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

    reconstruction = decoder(received, state, layer_gates=gates)

    assert torch.all(reconstruction[0, 1:] == 0)
    assert torch.all(reconstruction[1, 2] == 0)

