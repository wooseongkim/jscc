from __future__ import annotations

import torch
from torch import nn
import yaml

from speech_jscc.models.conv_conformer import (
    ConvConformerJSCC,
    balanced_ragged_valid_mask,
    masked_complex_power_normalize,
    pack_valid_symbols,
    unpack_valid_symbols,
)
from speech_jscc.training.channel_free_revalidation import (
    CHANNEL_FREE_EXPERIMENTS,
    checkpoint_filenames,
    curriculum_weights,
    framewise_summed_nmse,
    summed_decoder_input,
    summed_latent_statistics,
    waveform_connected_objective,
    apply_experiment_definition,
    feasibility_classification,
)


def tiny_model(*, symbol_frames: int, channel_uses: int, complex_channels: int,
               layout: str = "dense_interpolate") -> ConvConformerJSCC:
    return ConvConformerJSCC(
        (8, 50, 16),
        channel_uses,
        8,
        1.0,
        d_model=16,
        encoder_conformer_blocks=1,
        decoder_conformer_blocks=1,
        num_attention_heads=4,
        ffn_expansion=2,
        convolution_kernel_size=3,
        dropout=0.0,
        layer_mixer_blocks=0,
        symbol_frames=symbol_frames,
        complex_channels_per_symbol_frame=complex_channels,
        temporal_symbol_layout=layout,
    )


def test_balanced_ragged_mask_has_exact_uniform_40_by_5_plus_10_by_4_pattern():
    mask = balanced_ragged_valid_mask(frames=50, max_symbols=5, valid_symbols=240)
    counts = mask.sum(-1)
    four_symbol_frames = torch.where(counts == 4)[0].tolist()
    assert mask.shape == (50, 5)
    assert int(mask.sum()) == 240
    assert len(four_symbol_frames) == 10
    assert four_symbol_frames == [2, 7, 12, 17, 22, 27, 32, 37, 42, 47]
    assert set(counts.tolist()) == {4, 5}


def test_ragged_pack_unpack_is_identity_and_invalid_slots_are_zero():
    mask = balanced_ragged_valid_mask(frames=50, max_symbols=5, valid_symbols=240)
    packed = torch.randn(2, 8, 240, dtype=torch.complex64)
    fixed = unpack_valid_symbols(packed, mask)
    restored = pack_valid_symbols(fixed, mask)
    assert fixed.shape == (2, 8, 50, 5)
    assert torch.count_nonzero(fixed[..., ~mask]) == 0
    torch.testing.assert_close(restored, packed)


def test_masked_power_normalization_ignores_invalid_slots():
    mask = balanced_ragged_valid_mask(frames=50, max_symbols=5, valid_symbols=240)
    value = torch.randn(2, 8, 50, 5, dtype=torch.complex64)
    value[..., ~mask] = 1000 + 1000j
    normalized = masked_complex_power_normalize(value, mask, target_power=1.0)
    assert torch.count_nonzero(normalized[..., ~mask]) == 0
    valid = normalized[..., mask]
    torch.testing.assert_close(
        valid.abs().square().mean(dim=-1),
        torch.ones(2, 8),
        atol=1e-5,
        rtol=1e-5,
    )


def test_cf2_preserves_50_positions_and_exact_1920_symbols_without_interpolation():
    model = tiny_model(
        symbol_frames=50,
        channel_uses=1920,
        complex_channels=5,
        layout="balanced_ragged",
    )
    representation = torch.randn(2, 8, 50, 16)
    state = torch.zeros(2, 8)
    symbols, aux = model.encoder(representation, state, return_aux=True)
    reconstruction = model.decoder(symbols, state)
    assert symbols.shape == (2, 1920)
    assert reconstruction.shape == representation.shape
    assert aux["temporal_feature_shape"].tolist() == [2, 8, 50, 16]
    assert model.encoder.layer_channel_uses == (240,) * 8
    assert model.encoder.uses_temporal_interpolation is False
    assert model.decoder.uses_temporal_interpolation is False


def test_dense_30_and_dense_50_shapes_and_channel_use_counts():
    cf1 = tiny_model(symbol_frames=30, channel_uses=1920, complex_channels=8)
    cf3 = tiny_model(symbol_frames=50, channel_uses=3200, complex_channels=8)
    value = torch.randn(1, 8, 50, 16)
    state = torch.zeros(1, 8)
    assert cf1.encoder(value, state).shape == (1, 1920)
    assert cf3.encoder(value, state).shape == (1, 3200)
    assert cf1.encoder.uses_temporal_interpolation is True
    assert cf3.encoder.uses_temporal_interpolation is False


def test_summed_metrics_use_exact_layer_sum_passed_to_codec_decoder():
    target = torch.randn(2, 8, 50, 16)
    reconstruction = target + 0.1 * torch.randn_like(target)
    exact = reconstruction.sum(dim=1)
    torch.testing.assert_close(summed_decoder_input(reconstruction), exact)
    stats = summed_latent_statistics(reconstruction, target)
    manual = (exact - target.sum(1)).square().mean() / target.sum(1).square().mean()
    torch.testing.assert_close(stats["nmse"], manual)
    assert framewise_summed_nmse(reconstruction, target).shape == (50,)


def test_waveform_gradient_reaches_reconstruction_and_jscc_but_not_frozen_codec():
    class Codec(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Conv1d(16, 1, 1)

        def decode(self, layers):
            return self.decoder(layers.sum(1).transpose(1, 2)).squeeze(1)

    jscc = nn.Linear(16, 16)
    codec = Codec()
    codec.requires_grad_(False)
    target = torch.randn(2, 8, 50, 16)
    reconstruction = jscc(target)
    reconstruction.retain_grad()
    waveform_target = codec.decode(target).detach()
    loss, _ = waveform_connected_objective(
        reconstruction,
        target,
        waveform_target,
        codec.decode,
        weights={"layer": 1.0, "sum": 1.0, "stft": 0.01, "sisdr": 0.001},
        fft_sizes=(16, 32),
    )
    loss.backward()
    assert reconstruction.grad is not None and reconstruction.grad.abs().sum() > 0
    assert jscc.weight.grad is not None and jscc.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in codec.parameters())


def test_experiment_matrix_and_curriculum_are_controlled_and_unambiguous():
    assert CHANNEL_FREE_EXPERIMENTS["cf1"]["channel_uses"] == 1920
    assert CHANNEL_FREE_EXPERIMENTS["cf2"]["channel_uses"] == 1920
    assert CHANNEL_FREE_EXPERIMENTS["cf2"]["symbol_frames"] == 50
    assert CHANNEL_FREE_EXPERIMENTS["cf2"]["temporal_symbol_layout"] == "balanced_ragged"
    assert CHANNEL_FREE_EXPERIMENTS["cf3"]["channel_uses"] == 3200
    assert CHANNEL_FREE_EXPERIMENTS["cf4"]["d_model"] == 384
    assert curriculum_weights(1)["stft"] == curriculum_weights(4000)["stft"] == 0
    assert curriculum_weights(4001)["stft"] > 0 and curriculum_weights(4001)["sisdr"] == 0
    assert curriculum_weights(12001)["sisdr"] > 0
    assert checkpoint_filenames() == {
        "per_layer_nmse": "best_per_layer_nmse.pt",
        "summed_latent_nmse": "best_summed_latent_nmse.pt",
        "waveform_si_sdr": "best_waveform_si_sdr.pt",
    }


def test_experiment_definition_records_ragged_pattern_and_rejects_wrong_counts():
    config = {
        "model": {
            "architecture": "conv_conformer_v1",
            "channel_uses": 1920,
            "symbol_frames": 30,
            "complex_channels_per_symbol_frame": 8,
            "d_model": 256,
            "encoder_conformer_blocks": 4,
            "decoder_conformer_blocks": 4,
        }
    }
    resolved = apply_experiment_definition(config, "cf2")
    layout = resolved["model"]["temporal_symbol_pattern"]
    assert layout["four_symbol_frames"] == [2, 7, 12, 17, 22, 27, 32, 37, 42, 47]
    assert layout["valid_symbols_per_layer"] == 240
    assert layout["valid_symbols_total"] == 1920
    assert yaml.safe_load(yaml.safe_dump(resolved)) == resolved


def test_waveform_feasibility_requires_all_three_metrics():
    assert feasibility_classification(-0.9, -0.9, 1.19) == "CHANNEL_FREE_FEASIBLE"
    assert feasibility_classification(-0.9, -1.1, 1.19) == "CHANNEL_FREE_CONV_CONFORMER_NOT_YET_FEASIBLE"
    assert feasibility_classification(-0.9, -0.9, 1.21) == "CHANNEL_FREE_CONV_CONFORMER_NOT_YET_FEASIBLE"
