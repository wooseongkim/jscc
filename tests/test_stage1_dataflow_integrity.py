from __future__ import annotations

import torch

from models.resource_allocator import allocate_resources, deallocate_resources
from speech_jscc.diagnostics.dataflow import audit_resource_mapping
from channels.pilot import make_pilot_mask
from channels.pilot import extract_data_resources, insert_data_and_pilots
from evaluation.paired import generate_paired_evaluation_batch, run_mode_on_paired_batch
from speech_jscc.codecs import MockContinuousCodec
from speech_jscc.models import SpeechJSCC


def test_pilot_audit_counts_overwritten_encoder_symbols() -> None:
    symbols = torch.arange(32).reshape(1, 8, 4).to(torch.complex64)
    mask = make_pilot_mask(symbols.shape, spacing=2, time_spacing=2)
    audit = audit_resource_mapping(symbols, mask, (8, 8, 8, 8))
    assert audit["grid_resources_per_sample"] == 32
    assert audit["pilot_resources_per_sample"] == 8
    assert audit["nonpilot_data_resources_per_sample"] == 24
    assert audit["encoder_symbols_per_sample"] == 32
    assert audit["overwritten_encoder_symbols_per_sample"] == 8
    assert audit["resource_mapping_defect"] is True


def test_current_stage1_grid_overwrites_128_encoder_symbols() -> None:
    symbols = torch.ones(2, 64, 32, dtype=torch.complex64)
    mask = make_pilot_mask(symbols.shape, spacing=4, time_spacing=4)
    audit = audit_resource_mapping(symbols, mask, (256,) * 8)
    assert audit["pilot_resources_per_sample"] == 128
    assert audit["nonpilot_data_resources_per_sample"] == 1920
    assert audit["overwritten_encoder_symbols_per_sample"] == 128
    assert audit["pilot_fraction"] == 0.0625
    assert audit["resource_mapping_defect"] is True


def test_o2p_pilot_reserved_identity_recovers_all_1920_encoder_symbols() -> None:
    pilot_mask = make_pilot_mask((2, 64, 32), spacing=4, time_spacing=4)
    data = torch.randn(2, 1920, dtype=torch.complex64)

    transmitted, pilots = insert_data_and_pilots(data, pilot_mask)
    recovered = extract_data_resources(transmitted, pilot_mask)

    assert transmitted.shape == (2, 64, 32)
    assert int(pilot_mask[0].sum()) == 128
    assert data.shape[1] == 1920
    assert int(torch.count_nonzero(pilots[pilot_mask])) == 256
    torch.testing.assert_close(recovered, data)


def test_production_forward_preserves_every_encoder_symbol_with_reserved_pilots() -> None:
    torch.manual_seed(3)
    codec = MockContinuousCodec(2, 3, 4, 96, seed=1)
    model = SpeechJSCC((2, 3, 4), 24, channel_state_dim=8, hidden_dim=16)
    waveform = torch.randn(1, 96)
    target = codec.encode_waveform(waveform)
    batch = generate_paired_evaluation_batch(
        codec,
        batch_size=1,
        waveform_samples=96,
        channel_shape=(8, 4),
        snr_db=100.0,
        jsr_db=0.0,
        jammer_type="none",
        jammed_fraction=0.25,
        pilot_spacing=2,
        pilot_time_spacing=2,
        target_power=1.0,
        seed=7,
        device=torch.device("cpu"),
        fading="multipath_block",
        num_taps=2,
        channel_estimator="dft_tap_ls",
        estimator_num_taps=2,
        waveform=waveform,
        representation=target,
    )
    result = run_mode_on_paired_batch(
        codec,
        model,
        batch,
        torch.zeros(1, 8),
        torch.ones(1, 2),
        equalizer="oracle",
        fading="multipath_block",
        channel_estimator="dft_tap_ls",
        estimator_num_taps=2,
        allocation_mode="uniform",
        receiver_state_mode="neutral",
        decode_waveform=False,
    )

    assert result["data_symbols"].shape == (1, 24)
    assert result["decoder_input"].shape == (1, 24)
    assert result["pilot_overwrite_count"] == 0
    assert model.encoder.layer_channel_uses == (12, 12)
    assert abs(float(result["encoder_data_power"].detach()) - 1.0) < 1.0e-5
    assert abs(float(result["transmitted_grid_power"].detach()) - 1.0) < 1.0e-5


def test_uniform_allocate_deallocate_is_exact_without_channel() -> None:
    symbols = torch.randn(2, 8, 4, dtype=torch.complex64)
    result = allocate_resources(
        symbols, torch.ones_like(symbols.real), (8, 8, 8, 8), mode="uniform"
    )
    restored = deallocate_resources(result.symbols, result.resource_to_source)
    torch.testing.assert_close(restored, symbols)


def test_stage1_production_encoder_uses_240_symbols_per_layer() -> None:
    model = SpeechJSCC((8, 50, 1024), 1920, channel_state_dim=8, hidden_dim=2)
    assert model.encoder.channel_shape == (1920,)
    assert model.decoder.channel_shape == (1920,)
    assert model.encoder.layer_channel_uses == (240,) * 8
