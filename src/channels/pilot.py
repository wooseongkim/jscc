from __future__ import annotations

import torch
from torch import Tensor


def make_pilot_mask(
    shape: tuple[int, ...],
    spacing: int = 4,
    *,
    time_spacing: int | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Create a regular pilot mask for `[B,M]` or `[B,K,N]` resources."""
    if len(shape) not in (2, 3) or any(size <= 0 for size in shape):
        raise ValueError("shape must be positive [B,M] or [B,K,N]")
    if spacing <= 0 or (time_spacing is not None and time_spacing <= 0):
        raise ValueError("pilot spacing must be positive")
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    if len(shape) == 2:
        mask[:, ::spacing] = True
    else:
        mask[:, ::spacing, ::(time_spacing or spacing)] = True
    return mask


def insert_pilots(
    data_symbols: Tensor,
    pilot_mask: Tensor,
    pilot_value: complex = 1.0 + 0.0j,
) -> tuple[Tensor, Tensor]:
    """Replace masked resources by known pilots and return the pilot tensor."""
    if not torch.is_complex(data_symbols) or data_symbols.ndim not in (2, 3):
        raise ValueError("data_symbols must be complex [B,M] or [B,K,N]")
    mask = torch.broadcast_to(
        pilot_mask.to(device=data_symbols.device, dtype=torch.bool), data_symbols.shape
    )
    pilots = torch.zeros_like(data_symbols)
    pilots[mask] = torch.as_tensor(pilot_value, dtype=data_symbols.dtype, device=data_symbols.device)
    transmitted = torch.where(mask, pilots, data_symbols)
    return transmitted, pilots


def insert_data_and_pilots(
    data_symbols: Tensor,
    pilot_mask: Tensor,
    pilot_value: complex = 1.0 + 0.0j,
) -> tuple[Tensor, Tensor]:
    """Pack flat data only into non-pilot grid locations and fill reserved pilots."""
    if not torch.is_complex(data_symbols) or data_symbols.ndim != 2:
        raise ValueError("data_symbols must be complex [B,M_data]")
    if pilot_mask.ndim != 3 or pilot_mask.shape[0] != data_symbols.shape[0]:
        raise ValueError("pilot_mask must have shape [B,K,N]")
    mask = pilot_mask.to(device=data_symbols.device, dtype=torch.bool)
    data_counts = (~mask).flatten(1).sum(dim=1)
    if torch.any(data_counts != data_symbols.shape[1]):
        raise ValueError("data symbol count must equal non-pilot resource count")
    grid = torch.empty(mask.shape, device=data_symbols.device, dtype=data_symbols.dtype)
    for index in range(data_symbols.shape[0]):
        grid[index][~mask[index]] = data_symbols[index]
    pilots = torch.zeros_like(grid)
    pilots[mask] = torch.as_tensor(pilot_value, dtype=grid.dtype, device=grid.device)
    grid[mask] = pilots[mask]
    return grid, pilots


def extract_data_resources(resources: Tensor, pilot_mask: Tensor) -> Tensor:
    """Recover flat data from non-pilot grid locations in stable row-major order."""
    if not torch.is_complex(resources) or resources.ndim != 3:
        raise ValueError("resources must be complex [B,K,N]")
    mask = torch.broadcast_to(pilot_mask.to(resources.device, torch.bool), resources.shape)
    counts = (~mask).flatten(1).sum(dim=1)
    if torch.any(counts != counts[0]):
        raise ValueError("every sample must have the same non-pilot resource count")
    return torch.stack([resources[index][~mask[index]] for index in range(resources.shape[0])])


def estimate_flat_ls(received: Tensor, pilots: Tensor, pilot_mask: Tensor, eps: float = 1e-12) -> Tensor:
    """Estimate one block-fading coefficient per example from known pilots."""
    if received.ndim != 2 or received.shape != pilots.shape:
        raise ValueError("flat LS requires matching [B,M] received and pilot tensors")
    mask = torch.broadcast_to(pilot_mask.to(received.device, torch.bool), received.shape)
    masked_pilots = pilots * mask
    numerator = (received * masked_pilots.conj()).sum(dim=1, keepdim=True)
    denominator = masked_pilots.abs().square().sum(dim=1, keepdim=True).clamp_min(eps)
    return numerator / denominator


def interpolate_pilot_estimates(sparse_estimate: Tensor, pilot_mask: Tensor) -> Tensor:
    """Inverse-distance interpolation of complex pilot LS estimates over an OFDM grid."""
    if sparse_estimate.ndim != 3:
        raise ValueError("OFDM interpolation requires [B,K,N]")
    mask = torch.broadcast_to(pilot_mask.to(sparse_estimate.device, torch.bool), sparse_estimate.shape)
    batch, subcarriers, symbols = sparse_estimate.shape
    coordinate_dtype = sparse_estimate.real.dtype
    frequency = torch.arange(subcarriers, device=sparse_estimate.device, dtype=coordinate_dtype)
    time = torch.arange(symbols, device=sparse_estimate.device, dtype=coordinate_dtype)
    all_coordinates = torch.cartesian_prod(frequency, time)
    scale = sparse_estimate.real.new_tensor(
        [max(subcarriers - 1, 1), max(symbols - 1, 1)]
    )
    normalized_coordinates = all_coordinates / scale
    output = torch.empty_like(sparse_estimate)

    for batch_index in range(batch):
        pilot_coordinates = mask[batch_index].nonzero(as_tuple=False)
        if pilot_coordinates.numel() == 0:
            raise ValueError("every batch item requires at least one pilot")
        normalized_pilots = pilot_coordinates.to(coordinate_dtype) / scale
        distances = torch.cdist(normalized_coordinates, normalized_pilots)
        neighbors = min(4, pilot_coordinates.shape[0])
        nearest_distance, nearest_index = distances.topk(neighbors, largest=False, dim=1)
        weights = nearest_distance.clamp_min(1e-6).reciprocal()
        weights = weights / weights.sum(dim=1, keepdim=True)
        pilot_values = sparse_estimate[batch_index][mask[batch_index]]
        interpolated = (pilot_values[nearest_index] * weights).sum(dim=1)
        output[batch_index] = interpolated.reshape(subcarriers, symbols)
        output[batch_index][mask[batch_index]] = sparse_estimate[batch_index][mask[batch_index]]
    return output


def _interp_frequency(values: Tensor, pilot_indices: Tensor, subcarriers: int) -> Tensor:
    if pilot_indices.numel() < 1:
        raise ValueError("at least one unique pilot subcarrier is required")
    if pilot_indices.numel() == 1:
        return values[0].expand(subcarriers)
    order = pilot_indices.argsort()
    xp = pilot_indices[order]
    fp = values[order]
    grid = torch.arange(subcarriers, device=values.device, dtype=xp.dtype)
    right = torch.searchsorted(xp, grid, right=False).clamp(max=xp.numel() - 1)
    left = (right - 1).clamp(min=0)
    left_edge = grid <= xp[0]
    right_edge = grid >= xp[-1]
    left = torch.where(left_edge, torch.zeros_like(left), left)
    right = torch.where(left_edge, torch.zeros_like(right), right)
    left = torch.where(right_edge, torch.full_like(left, xp.numel() - 1), left)
    right = torch.where(right_edge, torch.full_like(right, xp.numel() - 1), right)
    x0 = xp[left].to(values.real.dtype)
    x1 = xp[right].to(values.real.dtype)
    weight = ((grid.to(values.real.dtype) - x0) / (x1 - x0).clamp_min(1.0)).clamp(0.0, 1.0)
    return fp[left] * (1.0 - weight) + fp[right] * weight


def estimate_ofdm_block_ls(
    received: Tensor,
    pilots: Tensor,
    pilot_mask: Tensor,
    eps: float = 1e-12,
) -> Tensor:
    """Estimate block-fading OFDM CSI by averaging pilots over time.

    The estimator computes LS values at pilot positions, averages repeated
    observations for each pilot subcarrier over OFDM symbols, interpolates
    missing subcarriers along frequency, and expands the result across time.
    It does not access oracle channel coefficients.
    """
    if received.ndim != 3 or received.shape != pilots.shape:
        raise ValueError("block OFDM LS requires matching [B,K,N] received and pilot tensors")
    mask = torch.broadcast_to(pilot_mask.to(received.device, torch.bool), received.shape)
    if torch.any(mask.sum(dim=(1, 2)) == 0):
        raise ValueError("every batch item requires at least one pilot")
    if not torch.isfinite(received).all() or not torch.isfinite(pilots).all():
        raise ValueError("received and pilots must be finite")
    batch, subcarriers, symbols = received.shape
    response = torch.empty((batch, subcarriers), device=received.device, dtype=received.dtype)
    for batch_index in range(batch):
        pilot_subcarriers = torch.nonzero(mask[batch_index].any(dim=1), as_tuple=False).flatten()
        if pilot_subcarriers.numel() < 1:
            raise ValueError("every batch item requires at least one unique pilot subcarrier")
        averaged = []
        for subcarrier in pilot_subcarriers:
            time_mask = mask[batch_index, subcarrier]
            pilot_values = pilots[batch_index, subcarrier, time_mask]
            received_values = received[batch_index, subcarrier, time_mask]
            safe = torch.where(
                pilot_values.abs() > eps,
                pilot_values,
                torch.full_like(pilot_values, eps),
            )
            averaged.append((received_values / safe).mean())
        response[batch_index] = _interp_frequency(
            torch.stack(averaged),
            pilot_subcarriers.to(received.device),
            subcarriers,
        )
    return response[:, :, None].expand(batch, subcarriers, symbols)


def dft_tap_matrix(
    pilot_subcarriers: Tensor,
    num_taps: int,
    subcarriers: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Partial forward-FFT DFT matrix matching `torch.fft.fft(taps, n=K)`."""
    if int(num_taps) < 1:
        raise ValueError("num_taps must be at least 1")
    if int(num_taps) > int(subcarriers):
        raise ValueError("num_taps must not exceed the number of subcarriers")
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    k = pilot_subcarriers.to(device=device, dtype=real_dtype)[:, None]
    l = torch.arange(int(num_taps), device=device, dtype=real_dtype)[None, :]
    phase = -2.0 * torch.pi * k * l / float(subcarriers)
    return torch.exp(torch.complex(torch.zeros_like(phase), phase)).to(dtype=dtype)


def estimate_ofdm_dft_tap_ls(
    received: Tensor,
    pilots: Tensor,
    pilot_mask: Tensor,
    num_taps: int,
    ridge_lambda: float = 1.0e-6,
    return_diagnostics: bool = False,
    eps: float = 1e-12,
) -> Tensor | tuple[Tensor, dict[str, Tensor | float]]:
    """Estimate finite TDL taps from sparse OFDM pilots using ridge LS.

    Solves `(F_P^H F_P + lambda I) h = F_P^H H_bar_P`, where `F_P` uses the
    same forward FFT convention as `torch.fft.fft(taps, n=K)`.
    """
    if not torch.is_complex(received) or not torch.is_complex(pilots):
        raise ValueError("received and pilots must be complex tensors")
    if received.ndim != 3 or received.shape != pilots.shape:
        raise ValueError("DFT tap LS requires matching [B,K,N] received and pilot tensors")
    if int(num_taps) < 1:
        raise ValueError("num_taps must be at least 1")
    if float(ridge_lambda) < 0.0:
        raise ValueError("ridge_lambda must be nonnegative")
    mask = torch.broadcast_to(pilot_mask.to(received.device, torch.bool), received.shape)
    batch, subcarriers, symbols = received.shape
    if int(num_taps) > subcarriers:
        raise ValueError("num_taps must not exceed the number of subcarriers")
    if torch.any(mask.sum(dim=(1, 2)) == 0):
        raise ValueError("every batch item requires at least one pilot")
    if not torch.isfinite(received).all() or not torch.isfinite(pilots).all():
        raise ValueError("received and pilots must be finite")

    unique_mask = mask.any(dim=(0, 2))
    pilot_subcarriers = torch.nonzero(unique_mask, as_tuple=False).flatten()
    if pilot_subcarriers.numel() < int(num_taps):
        raise ValueError("DFT tap LS requires at least num_taps unique pilot subcarriers")
    counts = mask[:, pilot_subcarriers, :].sum(dim=2)
    if torch.any(counts == 0):
        raise ValueError("every selected pilot subcarrier requires at least one observation")
    observations = []
    for subcarrier in pilot_subcarriers:
        time_mask = mask[:, subcarrier, :]
        pilot_values = pilots[:, subcarrier, :]
        received_values = received[:, subcarrier, :]
        safe = torch.where(
            pilot_values.abs() > eps,
            pilot_values,
            torch.full_like(pilot_values, eps),
        )
        ls_values = received_values / safe
        summed = (ls_values * time_mask).sum(dim=1)
        count = time_mask.sum(dim=1).clamp_min(1)
        observations.append(summed / count.to(received.real.dtype))
    h_bar = torch.stack(observations, dim=1)

    matrix = dft_tap_matrix(
        pilot_subcarriers,
        int(num_taps),
        subcarriers,
        dtype=received.dtype,
        device=received.device,
    )
    rank = torch.linalg.matrix_rank(matrix)
    condition_number = torch.linalg.cond(matrix)
    if float(ridge_lambda) == 0.0 and (
        int(rank.item()) < int(num_taps) or float(condition_number.detach().cpu().item()) > 1e6
    ):
        raise ValueError("DFT pilot matrix is rank deficient or ill-conditioned with ridge_lambda=0")
    gram = matrix.conj().transpose(0, 1) @ matrix
    rhs = h_bar @ matrix.conj()
    identity = torch.eye(int(num_taps), device=received.device, dtype=received.dtype)
    regularized = gram + float(ridge_lambda) * identity
    taps = torch.linalg.solve(regularized[None, :, :].expand(batch, -1, -1), rhs[:, :, None]).squeeze(-1)
    full_matrix = dft_tap_matrix(
        torch.arange(subcarriers, device=received.device),
        int(num_taps),
        subcarriers,
        dtype=received.dtype,
        device=received.device,
    )
    response_1d = taps @ full_matrix.transpose(0, 1)
    estimate = response_1d[:, :, None].expand(batch, subcarriers, symbols)
    if not torch.isfinite(estimate).all():
        raise ValueError("DFT tap LS produced non-finite channel estimates")
    if not return_diagnostics:
        return estimate
    diagnostics: dict[str, Tensor | float] = {
        "estimated_taps": taps,
        "unique_pilot_subcarriers": pilot_subcarriers,
        "pilot_observation_count_per_subcarrier": torch.zeros(
            batch, subcarriers, device=received.device, dtype=torch.long
        ),
        "dft_matrix_condition_number": condition_number.detach(),
        "ridge_lambda": float(ridge_lambda),
    }
    diagnostics["pilot_observation_count_per_subcarrier"][:, pilot_subcarriers] = counts
    return estimate, diagnostics


def estimate_ofdm_ls(received: Tensor, pilots: Tensor, pilot_mask: Tensor, eps: float = 1e-12) -> Tensor:
    """Estimate pilot positions by LS and interpolate all time-frequency resources."""
    if received.ndim != 3 or received.shape != pilots.shape:
        raise ValueError("OFDM LS requires matching [B,K,N] received and pilot tensors")
    mask = torch.broadcast_to(pilot_mask.to(received.device, torch.bool), received.shape)
    if torch.any(mask.sum(dim=(1, 2)) == 0):
        raise ValueError("every batch item requires at least one pilot")
    sparse = torch.zeros_like(received)
    denominator = pilots[mask]
    safe_denominator = torch.where(
        denominator.abs() > eps,
        denominator,
        torch.full_like(denominator, eps),
    )
    sparse[mask] = received[mask] / safe_denominator
    return interpolate_pilot_estimates(sparse, mask)


def estimate_channel_ls(
    received: Tensor,
    pilots: Tensor,
    pilot_mask: Tensor,
    *,
    fading: str = "auto",
    channel_estimator: str = "auto",
    estimator_num_taps: int | None = None,
    estimator_ridge_lambda: float = 1.0e-6,
) -> Tensor:
    """Dispatch block-flat or OFDM pilot LS estimation."""
    if channel_estimator not in {"auto", "inverse_distance_2d", "block_frequency_ls", "dft_tap_ls"}:
        raise ValueError(
            "channel_estimator must be auto, inverse_distance_2d, block_frequency_ls, or dft_tap_ls"
        )
    if received.ndim == 2:
        if channel_estimator != "auto":
            raise ValueError("flat channels only support the auto/flat LS estimator")
        return estimate_flat_ls(received, pilots, pilot_mask)
    if received.ndim == 3:
        if channel_estimator == "dft_tap_ls":
            if estimator_num_taps is None:
                raise ValueError("dft_tap_ls requires estimator_num_taps")
            return estimate_ofdm_dft_tap_ls(
                received,
                pilots,
                pilot_mask,
                int(estimator_num_taps),
                float(estimator_ridge_lambda),
            )
        if channel_estimator == "block_frequency_ls" or (
            channel_estimator == "auto" and fading == "multipath_block"
        ):
            return estimate_ofdm_block_ls(received, pilots, pilot_mask)
        return estimate_ofdm_ls(received, pilots, pilot_mask)
    raise ValueError("received must be [B,M] or [B,K,N]")


def equalize_with_csi(
    received: Tensor,
    channel_estimate: Tensor,
    eps: float = 1e-6,
    *,
    gain_cap: float | None = None,
) -> Tensor:
    """Complex zero-forcing equalization using estimated or oracle CSI."""
    try:
        torch.broadcast_shapes(received.shape, channel_estimate.shape)
    except RuntimeError as error:
        raise ValueError("channel estimate is not broadcastable to received") from error
    coefficient = channel_estimate.conj() / channel_estimate.abs().square().clamp_min(eps)
    if gain_cap is not None:
        if gain_cap <= 0:
            raise ValueError("gain_cap must be positive")
        magnitude = coefficient.abs()
        coefficient = coefficient * (float(gain_cap) / magnitude.clamp_min(1e-12)).clamp_max(1.0)
    return received * coefficient


def remove_pilot_resources(equalized: Tensor, pilot_mask: Tensor) -> Tensor:
    """Zero known pilot positions before passing the grid to the JSCC decoder."""
    mask = torch.broadcast_to(pilot_mask.to(equalized.device, torch.bool), equalized.shape)
    return equalized.masked_fill(mask, 0.0)


def csi_nmse(true_channel: Tensor, estimated_channel: Tensor, eps: float = 1e-12) -> Tensor:
    """Return per-example `||H-H_hat||^2 / ||H||^2`."""
    try:
        true_channel, estimated_channel = torch.broadcast_tensors(true_channel, estimated_channel)
    except RuntimeError as error:
        raise ValueError("true and estimated channels are not broadcastable") from error
    dimensions = tuple(range(1, true_channel.ndim))
    numerator = (true_channel - estimated_channel).abs().square().sum(dimensions)
    denominator = true_channel.abs().square().sum(dimensions).clamp_min(eps)
    return numerator / denominator


def pilot_evm(
    received: Tensor,
    pilots: Tensor,
    pilot_mask: Tensor,
    channel_estimate: Tensor,
    eps: float = 1e-12,
) -> Tensor:
    """Return RMS pilot residual EVM per example for a supplied channel estimate."""
    mask = torch.broadcast_to(pilot_mask.to(received.device, torch.bool), received.shape)
    expected = channel_estimate * pilots
    error_power = ((received - expected).abs().square() * mask).sum(
        tuple(range(1, received.ndim))
    )
    reference_power = (expected.abs().square() * mask).sum(
        tuple(range(1, received.ndim))
    ).clamp_min(eps)
    return torch.sqrt(error_power / reference_power)


__all__ = [
    "csi_nmse",
    "equalize_with_csi",
    "estimate_channel_ls",
    "estimate_ofdm_dft_tap_ls",
    "estimate_flat_ls",
    "estimate_ofdm_block_ls",
    "estimate_ofdm_ls",
    "insert_pilots",
    "insert_data_and_pilots",
    "extract_data_resources",
    "interpolate_pilot_estimates",
    "make_pilot_mask",
    "pilot_evm",
    "remove_pilot_resources",
]
