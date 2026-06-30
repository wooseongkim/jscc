import math

import torch

from channels.pilot import (
    csi_nmse,
    equalize_with_csi,
    estimate_channel_ls,
    insert_pilots,
    make_pilot_mask,
    pilot_evm,
    remove_pilot_resources,
)


def complex_normal(shape, generator):
    return torch.complex(
        torch.randn(shape, generator=generator),
        torch.randn(shape, generator=generator),
    ) / math.sqrt(2.0)


def test_csi_nmse_decreases_when_pilot_snr_increases():
    generator = torch.Generator().manual_seed(10)
    batch, uses = 512, 64
    channel = complex_normal((batch, 1), generator)
    mask = make_pilot_mask((batch, uses), spacing=4)
    transmitted, pilots = insert_pilots(torch.zeros(batch, uses, dtype=torch.complex64), mask)
    base_noise = complex_normal((batch, uses), generator)

    received_low = channel * transmitted + base_noise
    received_high = channel * transmitted + base_noise * 0.1
    low_estimate = estimate_channel_ls(received_low, pilots, mask)
    high_estimate = estimate_channel_ls(received_high, pilots, mask)

    assert csi_nmse(channel, high_estimate).mean() < csi_nmse(channel, low_estimate).mean()


def test_csi_nmse_decreases_when_ofdm_pilot_density_increases():
    batch, subcarriers, symbols = 16, 16, 16
    frequency = torch.linspace(0.0, 1.0, subcarriers)[None, :, None]
    time = torch.linspace(0.0, 1.0, symbols)[None, None, :]
    channel = torch.complex(
        1.0 + 0.25 * torch.sin(math.pi * frequency) + 0.15 * time,
        0.2 * frequency + 0.1 * torch.cos(math.pi * time),
    ).expand(batch, -1, -1)
    data = torch.zeros_like(channel)
    generator = torch.Generator().manual_seed(22)
    noise = 0.01 * complex_normal(tuple(channel.shape), generator)

    sparse_mask = make_pilot_mask(tuple(channel.shape), spacing=4, time_spacing=4)
    dense_mask = make_pilot_mask(tuple(channel.shape), spacing=2, time_spacing=2)
    sparse_tx, sparse_pilots = insert_pilots(data, sparse_mask)
    dense_tx, dense_pilots = insert_pilots(data, dense_mask)
    sparse_estimate = estimate_channel_ls(channel * sparse_tx + noise, sparse_pilots, sparse_mask)
    dense_estimate = estimate_channel_ls(channel * dense_tx + noise, dense_pilots, dense_mask)

    assert csi_nmse(channel, dense_estimate).mean() < csi_nmse(channel, sparse_estimate).mean()


def test_pilot_jamming_increases_csi_nmse():
    generator = torch.Generator().manual_seed(33)
    batch, uses = 256, 64
    channel = complex_normal((batch, 1), generator)
    mask = make_pilot_mask((batch, uses), spacing=4)
    transmitted, pilots = insert_pilots(torch.zeros(batch, uses, dtype=torch.complex64), mask)
    noise = 0.02 * complex_normal((batch, uses), generator)
    pilot_jammer = 1.5 * complex_normal((batch, uses), generator) * mask

    clean_estimate = estimate_channel_ls(channel * transmitted + noise, pilots, mask)
    jammed_estimate = estimate_channel_ls(
        channel * transmitted + noise + pilot_jammer, pilots, mask
    )

    assert csi_nmse(channel, jammed_estimate).mean() > csi_nmse(channel, clean_estimate).mean()


def test_perfect_csi_oracle_has_no_higher_symbol_reconstruction_error():
    generator = torch.Generator().manual_seed(44)
    batch, uses = 256, 48
    data = complex_normal((batch, uses), generator)
    channel = 0.75 + 0.25 * complex_normal((batch, 1), generator)
    mask = make_pilot_mask((batch, uses), spacing=6)
    transmitted, pilots = insert_pilots(data, mask)
    noise = 0.12 * complex_normal((batch, uses), generator)
    received = channel * transmitted + noise
    estimated_channel = estimate_channel_ls(received, pilots, mask)

    oracle = remove_pilot_resources(equalize_with_csi(received, channel), mask)
    estimated = remove_pilot_resources(equalize_with_csi(received, estimated_channel), mask)
    target = remove_pilot_resources(data, mask)
    oracle_error = (oracle - target).abs().square().mean()
    estimated_error = (estimated - target).abs().square().mean()

    assert oracle_error <= estimated_error


def test_pilot_evm_tracks_pilot_impairment():
    batch, uses = 8, 32
    channel = torch.full((batch, 1), 0.8 + 0.2j, dtype=torch.complex64)
    mask = make_pilot_mask((batch, uses), spacing=4)
    transmitted, pilots = insert_pilots(torch.zeros(batch, uses, dtype=torch.complex64), mask)
    clean = channel * transmitted
    impaired = clean + (0.1 + 0.05j) * mask

    clean_evm = pilot_evm(clean, pilots, mask, channel)
    impaired_evm = pilot_evm(impaired, pilots, mask, channel)

    torch.testing.assert_close(clean_evm, torch.zeros_like(clean_evm))
    assert torch.all(impaired_evm > clean_evm)
