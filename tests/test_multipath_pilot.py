from __future__ import annotations

import torch

from channels.jammer import make_jammer
from channels.pilot import (
    csi_nmse,
    estimate_channel_ls,
    estimate_ofdm_block_ls,
    insert_pilots,
    make_pilot_mask,
    pilot_evm,
)
from channels.rayleigh import rayleigh_channel
from evaluation.paired import (
    estimate_transmitter_feedback,
    generate_paired_evaluation_batch,
    run_mode_on_paired_batch,
)
from speech_jscc.codecs import MockContinuousCodec
from speech_jscc.models import SpeechJSCC


def test_block_ls_recovers_channel_with_all_subcarriers_piloted() -> None:
    batch, subcarriers, symbols = 2, 8, 4
    frequency = torch.arange(subcarriers).float()
    channel_1d = torch.complex(1.0 + 0.05 * frequency, 0.2 - 0.03 * frequency)
    channel = channel_1d.reshape(1, subcarriers, 1).expand(batch, -1, symbols)
    mask = torch.ones(batch, subcarriers, symbols, dtype=torch.bool)
    transmitted, pilots = insert_pilots(torch.zeros_like(channel), mask)
    estimate = estimate_ofdm_block_ls(channel * transmitted, pilots, mask)

    torch.testing.assert_close(estimate, channel)
    torch.testing.assert_close(estimate[:, :, 0], estimate[:, :, -1])


def test_block_ls_sparse_pilots_are_finite_and_reasonable() -> None:
    batch, subcarriers, symbols = 2, 16, 8
    taps = torch.tensor([[1.0 + 0.0j, 0.25 + 0.15j]], dtype=torch.complex64).expand(batch, -1)
    channel = torch.fft.fft(taps, n=subcarriers)[:, :, None].expand(-1, -1, symbols)
    mask = make_pilot_mask(tuple(channel.shape), spacing=4, time_spacing=2)
    transmitted, pilots = insert_pilots(torch.zeros_like(channel), mask)
    estimate = estimate_channel_ls(
        channel * transmitted,
        pilots,
        mask,
        fading="multipath_block",
        channel_estimator="block_frequency_ls",
    )

    assert torch.isfinite(estimate).all()
    assert csi_nmse(channel, estimate).mean() < 0.25
    torch.testing.assert_close(estimate[:, :, 0], estimate[:, :, -1])


def test_block_ls_averages_repeated_pilots_over_time() -> None:
    batch, subcarriers, symbols = 1, 4, 4
    channel = torch.ones(batch, subcarriers, symbols, dtype=torch.complex64)
    mask = torch.zeros_like(channel, dtype=torch.bool)
    mask[:, 0, :] = True
    transmitted, pilots = insert_pilots(torch.zeros_like(channel), mask)
    received = channel * transmitted
    received[:, 0, 0] = 2.0 + 0.0j
    received[:, 0, 1:] = 1.0 + 0.0j

    estimate = estimate_ofdm_block_ls(received, pilots, mask)

    torch.testing.assert_close(estimate[:, 0, :], torch.full((1, symbols), 1.25 + 0.0j))


def test_pilot_jammer_contamination_increases_block_csi_error() -> None:
    batch, subcarriers, symbols = 4, 16, 8
    channel = torch.ones(batch, subcarriers, symbols, dtype=torch.complex64)
    mask = make_pilot_mask(tuple(channel.shape), spacing=4, time_spacing=2)
    transmitted, pilots = insert_pilots(torch.zeros_like(channel), mask)
    clean = channel * transmitted
    jammer, _ = make_jammer(transmitted, 0.0, "pilot", pilot_mask=mask)
    clean_estimate = estimate_ofdm_block_ls(clean, pilots, mask)
    jammed_estimate = estimate_ofdm_block_ls(clean + jammer, pilots, mask)

    assert csi_nmse(channel, jammed_estimate).mean() > csi_nmse(channel, clean_estimate).mean()
    assert pilot_evm(clean + jammer, pilots, mask, clean_estimate).mean() > 0


def test_paired_multipath_batch_is_deterministic_and_block_constant() -> None:
    codec = MockContinuousCodec(2, 3, 2, 64, seed=2)
    args = dict(
        codec=codec,
        batch_size=2,
        waveform_samples=64,
        channel_shape=(8, 4),
        snr_db=10.0,
        jsr_db=0.0,
        jammer_type="narrowband",
        jammed_fraction=0.25,
        pilot_spacing=2,
        pilot_time_spacing=2,
        target_power=1.0,
        device=torch.device("cpu"),
        fading="multipath_block",
        num_taps=3,
        pdp_decay=0.7,
    )
    first = generate_paired_evaluation_batch(**args, seed=99)
    repeated = generate_paired_evaluation_batch(**args, seed=99)

    torch.testing.assert_close(first.signal_fading, repeated.signal_fading, rtol=0, atol=0)
    torch.testing.assert_close(first.jammer_fading, repeated.jammer_fading, rtol=0, atol=0)
    torch.testing.assert_close(first.noise, repeated.noise, rtol=0, atol=0)
    torch.testing.assert_close(first.jammer, repeated.jammer, rtol=0, atol=0)
    torch.testing.assert_close(first.signal_fading[:, :, 0], first.signal_fading[:, :, -1])
    assert first.metadata["fading"] == "multipath_block"


def test_paired_multipath_results_are_finite_and_modes_reuse_channel() -> None:
    codec = MockContinuousCodec(2, 3, 2, 64, seed=3)
    model = SpeechJSCC((2, 3, 2), (8, 4), channel_state_dim=8, hidden_dim=16)
    batch = generate_paired_evaluation_batch(
        codec,
        batch_size=2,
        waveform_samples=64,
        channel_shape=(8, 4),
        snr_db=10.0,
        jsr_db=0.0,
        jammer_type="pilot",
        jammed_fraction=0.25,
        pilot_spacing=2,
        pilot_time_spacing=2,
        target_power=1.0,
        seed=101,
        device=torch.device("cpu"),
        fading="multipath_block",
        num_taps=3,
        pdp_decay=0.7,
        channel_estimator="block_frequency_ls",
    )
    feedback = estimate_transmitter_feedback(
        batch, fading="multipath_block", channel_estimator="block_frequency_ls"
    )
    gates_a = torch.ones(2, 2)
    gates_b = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    result_a = run_mode_on_paired_batch(
        codec,
        model,
        batch,
        feedback["state"],
        gates_a,
        fading="multipath_block",
        channel_estimator="block_frequency_ls",
        resource_reliability=feedback["reliability"],
    )
    result_b = run_mode_on_paired_batch(
        codec,
        model,
        batch,
        feedback["state"],
        gates_b,
        fading="multipath_block",
        channel_estimator="block_frequency_ls",
        resource_reliability=feedback["reliability"],
    )

    torch.testing.assert_close(result_a["signal_fading"], result_b["signal_fading"], rtol=0, atol=0)
    torch.testing.assert_close(result_a["jammer_fading"], result_b["jammer_fading"], rtol=0, atol=0)
    torch.testing.assert_close(result_a["noise"], result_b["noise"], rtol=0, atol=0)
    assert torch.isfinite(result_a["post_equalization_sinr"]).all()
    assert torch.isfinite(result_a["csi_nmse"]).all()
    assert torch.isfinite(result_a["pilot_evm"]).all()
