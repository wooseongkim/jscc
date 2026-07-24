from __future__ import annotations

import torch

from channels.multipath import (
    exponential_pdp,
    multipath_block_fading,
    sample_tdl_taps,
    taps_to_ofdm_response,
)
from channels.rayleigh import post_channel_jsr, rayleigh_channel


def test_exponential_pdp_validation_and_normalization() -> None:
    pdp = exponential_pdp(6, 0.7)

    assert pdp.shape == (6,)
    assert torch.isfinite(pdp).all()
    assert torch.all(pdp >= 0)
    torch.testing.assert_close(pdp.sum(), torch.tensor(1.0))

    for num_taps, decay in [(0, 0.7), (2, 0.0), (2, 1.1)]:
        try:
            exponential_pdp(num_taps, decay)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid PDP parameters should raise")


def test_taps_to_ofdm_response_matches_fft_and_is_block_constant() -> None:
    taps = torch.tensor([[1.0 + 0.0j, 0.5 + 0.25j]], dtype=torch.complex64)
    response = taps_to_ofdm_response(taps, subcarriers=8, ofdm_symbols=5)
    expected = torch.fft.fft(taps, n=8)

    assert response.shape == (1, 8, 5)
    torch.testing.assert_close(response[:, :, 0], expected)
    torch.testing.assert_close(response[:, :, 1], expected)


def test_single_tap_is_frequency_flat_and_sampling_is_deterministic() -> None:
    pdp = exponential_pdp(1, 0.7)
    gen_a = torch.Generator().manual_seed(4)
    gen_b = torch.Generator().manual_seed(4)
    taps_a = sample_tdl_taps(3, pdp, generator=gen_a)
    taps_b = sample_tdl_taps(3, pdp, generator=gen_b)
    response = taps_to_ofdm_response(taps_a, 12, 4)

    torch.testing.assert_close(taps_a, taps_b)
    torch.testing.assert_close(response[:, 0:1, :].expand_as(response), response)
    different = sample_tdl_taps(3, pdp, generator=torch.Generator().manual_seed(5))
    assert not torch.equal(taps_a, different)


def test_tdl_statistics_match_pdp_without_per_realization_normalization() -> None:
    pdp = exponential_pdp(6, 0.7)
    taps = sample_tdl_taps(20000, pdp, generator=torch.Generator().manual_seed(8))
    response = taps_to_ofdm_response(taps, 32, 2)

    empirical = taps.abs().square().mean(dim=0)
    torch.testing.assert_close(empirical, pdp, rtol=0.08, atol=0.01)
    assert abs(float(response.abs().square().mean().item()) - 1.0) < 0.08


def test_multipath_block_fading_signal_jammer_independent_and_shapes() -> None:
    result = multipath_block_fading(
        batch_size=4,
        subcarriers=16,
        ofdm_symbols=7,
        num_taps=6,
        pdp_decay=0.7,
        reference=torch.zeros(4, 16, 7, dtype=torch.complex64),
        generator=torch.Generator().manual_seed(9),
    )

    assert result["signal_fading"].shape == (4, 16, 7)
    assert result["jammer_fading"].shape == (4, 16, 7)
    assert result["signal_taps"].shape == (4, 6)
    assert result["jammer_taps"].shape == (4, 6)
    assert not torch.equal(result["signal_taps"], result["jammer_taps"])
    torch.testing.assert_close(result["signal_fading"][:, :, 0], result["signal_fading"][:, :, -1])
    torch.testing.assert_close(result["jammer_fading"][:, :, 0], result["jammer_fading"][:, :, -1])


def test_rayleigh_channel_multipath_received_equation_and_gradient() -> None:
    transmitted = torch.ones(2, 8, 3, dtype=torch.complex64, requires_grad=True)
    jammer = torch.full_like(transmitted, 0.25 + 0.1j)
    signal_fading = torch.full_like(transmitted, 0.8 + 0.2j)
    jammer_fading = torch.full_like(transmitted, -0.1 + 0.4j)
    noise = torch.full_like(transmitted, 0.01 - 0.02j)

    result = rayleigh_channel(
        transmitted,
        jammer,
        20.0,
        fading="multipath_block",
        signal_fading=signal_fading,
        jammer_fading=jammer_fading,
        noise=noise,
    )

    expected = signal_fading * transmitted + jammer_fading * jammer + noise
    torch.testing.assert_close(result["received"], expected)
    assert result["fading_model"] == "multipath_block"
    loss = result["received"].abs().square().mean()
    loss.backward()
    assert transmitted.grad is not None
    assert torch.isfinite(transmitted.grad).all()


def test_legacy_flat_and_ofdm_shapes_are_unchanged() -> None:
    flat = rayleigh_channel(torch.ones(3, 10, dtype=torch.complex64), fading="flat")
    ofdm = rayleigh_channel(torch.ones(3, 8, 4, dtype=torch.complex64), fading="ofdm")

    assert flat["signal_fading"].shape == (3, 1)
    assert ofdm["signal_fading"].shape == (3, 8, 4)


def test_post_channel_jsr_uses_faded_components() -> None:
    faded_signal = torch.ones(2, 4, dtype=torch.complex64)
    faded_jammer = torch.full_like(faded_signal, 2.0)
    ratio = post_channel_jsr(faded_signal, faded_jammer)
    torch.testing.assert_close(ratio, torch.full((2,), 4.0))
