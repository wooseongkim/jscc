from __future__ import annotations

import torch

from evaluation.paired import generate_paired_evaluation_batch
from speech_jscc.codecs import MockContinuousCodec
from speech_jscc.models import SpeechJSCC
from speech_jscc.training.stage1 import (
    assert_stage1_startup_invariants,
    build_stage1_optimizer,
    stage1_fixed_tx_step,
)


def _small_stage1_objects():
    device = torch.device("cpu")
    codec = MockContinuousCodec(2, 3, 4, 96, seed=1).to(device)
    for parameter in codec.parameters():
        parameter.requires_grad_(False)
    codec.eval()
    model = SpeechJSCC((2, 3, 4), (8, 4), channel_state_dim=8, hidden_dim=24).to(device)
    optimizer = build_stage1_optimizer(model, learning_rate=1e-3)
    return device, codec, model, optimizer


def test_stage1_startup_invariants_and_optimizer_scope() -> None:
    _, codec, model, optimizer = _small_stage1_objects()

    info = assert_stage1_startup_invariants(codec, model, optimizer, allocation_mode="uniform")

    assert info["transmitter_policy"]["gate_mode"] == "all_ones"
    assert info["transmitter_policy"]["power_mode"] == "uniform"
    assert info["transmitter_policy"]["allocation_mode"] == "uniform"


def test_stage1_step_uses_fixed_tx_and_observable_rx_state() -> None:
    device, codec, model, optimizer = _small_stage1_objects()
    waveform = torch.randn(2, 96, device=device)
    with torch.no_grad():
        target = codec.encode_waveform(waveform)
    batch = generate_paired_evaluation_batch(
        codec,
        batch_size=2,
        waveform_samples=96,
        channel_shape=tuple(model.encoder.channel_shape),
        snr_db=10.0,
        jsr_db=0.0,
        jammer_type="none",
        jammed_fraction=0.25,
        pilot_spacing=2,
        pilot_time_spacing=2,
        target_power=1.0,
        seed=7,
        device=device,
        fading="multipath_block",
        num_taps=2,
        pdp_decay=0.7,
        channel_estimator="dft_tap_ls",
        estimator_num_taps=2,
        waveform=waveform,
        representation=target,
    )

    before_model = [parameter.detach().clone() for parameter in model.parameters()]
    before_codec = [parameter.detach().clone() for parameter in codec.parameters()]
    result = stage1_fixed_tx_step(
        codec,
        model,
        batch,
        optimizer,
        torch.ones(2),
        latent_normalization={"mode": "per_layer_power", "epsilon": 1e-6},
        channel_estimator="dft_tap_ls",
        estimator_num_taps=2,
        gradient_clip_norm=5.0,
    )

    assert result["transmitter_state"].shape == (2, 8)
    assert torch.count_nonzero(result["transmitter_state"]) == 0
    torch.testing.assert_close(result["layer_gates"], torch.ones(2, 2))
    torch.testing.assert_close(result["layer_power_fractions"], torch.full((2, 2), 0.5))
    assert result["receiver_state"].shape == (2, 8)
    assert torch.isfinite(result["receiver_state"]).all()
    assert result["allocation_mode"] == "uniform"
    assert result["fading_model"] == "multipath_block"
    assert result["channel_estimator"] == "dft_tap_ls"
    assert any(not torch.allclose(new, old) for new, old in zip(model.parameters(), before_model))
    for new, old in zip(codec.parameters(), before_codec):
        torch.testing.assert_close(new, old)


def test_stage1_fixed_batch_overfit_reduces_loss() -> None:
    device, codec, model, optimizer = _small_stage1_objects()
    waveform = torch.randn(1, 96, device=device)
    with torch.no_grad():
        target = codec.encode_waveform(waveform)
    batch = generate_paired_evaluation_batch(
        codec,
        batch_size=1,
        waveform_samples=96,
        channel_shape=tuple(model.encoder.channel_shape),
        snr_db=15.0,
        jsr_db=0.0,
        jammer_type="none",
        jammed_fraction=0.25,
        pilot_spacing=2,
        pilot_time_spacing=2,
        target_power=1.0,
        seed=11,
        device=device,
        fading="multipath_block",
        num_taps=2,
        pdp_decay=0.7,
        channel_estimator="dft_tap_ls",
        estimator_num_taps=2,
        waveform=waveform,
        representation=target,
    )

    losses = []
    for _ in range(30):
        result = stage1_fixed_tx_step(
            codec,
            model,
            batch,
            optimizer,
            torch.ones(2),
            latent_normalization={"mode": "per_layer_power", "epsilon": 1e-6},
            channel_estimator="dft_tap_ls",
            estimator_num_taps=2,
        )
        losses.append(float(result["loss"]))

    assert losses[-1] < losses[0] * 0.9
