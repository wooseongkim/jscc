from __future__ import annotations

import torch
import pytest

from channels.multipath import taps_to_ofdm_response
from channels.pilot import estimate_channel_ls, estimate_ofdm_dft_tap_ls, insert_pilots


def _dft_matrix(indices: torch.Tensor, num_taps: int, subcarriers: int, dtype=torch.complex64):
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    k = indices.to(real_dtype)[:, None]
    l = torch.arange(num_taps, dtype=real_dtype)[None, :]
    return torch.exp(torch.complex(torch.zeros_like(k * l), -2.0 * torch.pi * k * l / subcarriers)).to(dtype)


def test_dft_matrix_convention_matches_torch_fft_full_pilots() -> None:
    taps = torch.tensor([[1.0 + 0.0j, 0.5 - 0.25j, -0.1 + 0.2j]], dtype=torch.complex64)
    indices = torch.arange(8)

    expected = torch.fft.fft(taps, n=8)[0]
    actual = _dft_matrix(indices, 3, 8) @ taps[0]

    torch.testing.assert_close(actual, expected)


def test_dft_tap_ls_full_pilot_noiseless_recovery_is_exact() -> None:
    taps = torch.tensor(
        [[1.0 + 0.0j, 0.4 - 0.2j, 0.1 + 0.3j], [0.3 + 0.5j, -0.1 + 0.2j, 0.2 - 0.4j]],
        dtype=torch.complex64,
    )
    channel = taps_to_ofdm_response(taps, 12, 4)
    mask = torch.ones_like(channel, dtype=torch.bool)
    transmitted, pilots = insert_pilots(torch.zeros_like(channel), mask)

    estimate, diagnostics = estimate_ofdm_dft_tap_ls(
        channel * transmitted,
        pilots,
        mask,
        num_taps=3,
        ridge_lambda=0.0,
        return_diagnostics=True,
    )

    torch.testing.assert_close(estimate, channel, rtol=1e-5, atol=1e-5)
    assert diagnostics["estimated_taps"].shape == taps.shape
    torch.testing.assert_close(diagnostics["estimated_taps"], taps, rtol=1e-5, atol=1e-5)
    assert diagnostics["unique_pilot_subcarriers"].numel() == 12


def test_dft_tap_ls_sparse_noiseless_recovery_is_exact_when_full_rank() -> None:
    taps = torch.tensor([[1.0 + 0.0j, 0.25 + 0.1j, -0.2 + 0.05j]], dtype=torch.complex64)
    channel = taps_to_ofdm_response(taps, 16, 5)
    mask = torch.zeros_like(channel, dtype=torch.bool)
    mask[:, [0, 3, 7, 11], [0, 1, 2, 3]] = True
    transmitted, pilots = insert_pilots(torch.zeros_like(channel), mask)

    estimate = estimate_ofdm_dft_tap_ls(
        channel * transmitted,
        pilots,
        mask,
        num_taps=3,
        ridge_lambda=0.0,
    )

    torch.testing.assert_close(estimate, channel, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(estimate[:, :, 0], estimate[:, :, -1])


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_dft_tap_ls_dtype_shape_and_dispatch(dtype: torch.dtype) -> None:
    taps = torch.tensor([[1.0 + 0.0j, 0.2 - 0.1j]], dtype=dtype)
    channel = taps_to_ofdm_response(taps, 8, 3)
    mask = torch.ones_like(channel, dtype=torch.bool)
    transmitted, pilots = insert_pilots(torch.zeros_like(channel), mask)

    estimate = estimate_channel_ls(
        channel * transmitted,
        pilots,
        mask,
        fading="multipath_block",
        channel_estimator="dft_tap_ls",
        estimator_num_taps=2,
        estimator_ridge_lambda=1e-8,
    )

    assert estimate.dtype == dtype
    assert estimate.shape == channel.shape
    torch.testing.assert_close(estimate, channel, rtol=1e-5, atol=1e-5)


def test_dft_tap_ls_repeated_pilots_average_only_actual_observations() -> None:
    channel = torch.ones(1, 4, 4, dtype=torch.complex64)
    mask = torch.zeros_like(channel, dtype=torch.bool)
    mask[:, 0, [0, 2]] = True
    mask[:, 1, [1]] = True
    transmitted, pilots = insert_pilots(torch.zeros_like(channel), mask)
    received = channel * transmitted
    received[:, 0, 0] = 1.0 + 0.0j
    received[:, 0, 2] = 3.0 + 0.0j
    received[:, 1, 1] = 2.0 + 0.0j

    _, diagnostics = estimate_ofdm_dft_tap_ls(
        received,
        pilots,
        mask,
        num_taps=2,
        ridge_lambda=1e-6,
        return_diagnostics=True,
    )

    torch.testing.assert_close(
        diagnostics["pilot_observation_count_per_subcarrier"],
        torch.tensor([[2, 1, 0, 0]]),
    )


def test_dft_tap_ls_validation_and_rank_deficiency() -> None:
    channel = torch.ones(1, 8, 2, dtype=torch.complex64)
    mask = torch.zeros_like(channel, dtype=torch.bool)
    mask[:, [0, 4], :] = True
    transmitted, pilots = insert_pilots(torch.zeros_like(channel), mask)

    with pytest.raises(ValueError, match="at least num_taps"):
        estimate_ofdm_dft_tap_ls(channel * transmitted, pilots, mask, num_taps=3)
    with pytest.raises(ValueError, match="ridge_lambda"):
        estimate_ofdm_dft_tap_ls(channel * transmitted, pilots, mask, num_taps=2, ridge_lambda=-1.0)

    estimate = estimate_ofdm_dft_tap_ls(channel * transmitted, pilots, mask, num_taps=2, ridge_lambda=1e-3)
    assert torch.isfinite(estimate).all()

    ill_channel = torch.ones(1, 64, 2, dtype=torch.complex64)
    ill_mask = torch.zeros_like(ill_channel, dtype=torch.bool)
    ill_mask[:, :6, :] = True
    ill_tx, ill_pilots = insert_pilots(torch.zeros_like(ill_channel), ill_mask)
    with pytest.raises(ValueError, match="rank deficient|ill-conditioned"):
        estimate_ofdm_dft_tap_ls(ill_channel * ill_tx, ill_pilots, ill_mask, num_taps=6, ridge_lambda=0.0)


def test_dft_tap_ls_changes_with_received_and_has_gradients() -> None:
    channel = torch.ones(1, 8, 2, dtype=torch.complex64)
    mask = torch.ones_like(channel, dtype=torch.bool)
    transmitted, pilots = insert_pilots(torch.zeros_like(channel), mask)
    received = (channel * transmitted).detach().clone().requires_grad_(True)

    estimate_a = estimate_ofdm_dft_tap_ls(received, pilots, mask, num_taps=1)
    estimate_b = estimate_ofdm_dft_tap_ls(received * (2.0 + 0.0j), pilots, mask, num_taps=1)
    assert not torch.equal(estimate_a, estimate_b)
    estimate_a.abs().square().mean().backward()
    assert received.grad is not None
    assert torch.isfinite(received.grad).all()


def test_dft_tap_ls_cuda_when_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    channel = torch.ones(1, 8, 2, dtype=torch.complex64, device="cuda")
    mask = torch.ones_like(channel, dtype=torch.bool)
    transmitted, pilots = insert_pilots(torch.zeros_like(channel), mask)
    estimate = estimate_ofdm_dft_tap_ls(channel * transmitted, pilots, mask, num_taps=1)
    assert estimate.is_cuda
