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


def estimate_channel_ls(received: Tensor, pilots: Tensor, pilot_mask: Tensor) -> Tensor:
    """Dispatch block-flat or OFDM pilot LS estimation."""
    if received.ndim == 2:
        return estimate_flat_ls(received, pilots, pilot_mask)
    if received.ndim == 3:
        return estimate_ofdm_ls(received, pilots, pilot_mask)
    raise ValueError("received must be [B,M] or [B,K,N]")


def equalize_with_csi(received: Tensor, channel_estimate: Tensor, eps: float = 1e-6) -> Tensor:
    """Complex zero-forcing equalization using estimated or oracle CSI."""
    try:
        torch.broadcast_shapes(received.shape, channel_estimate.shape)
    except RuntimeError as error:
        raise ValueError("channel estimate is not broadcastable to received") from error
    return received * channel_estimate.conj() / channel_estimate.abs().square().clamp_min(eps)


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
    "estimate_flat_ls",
    "estimate_ofdm_ls",
    "insert_pilots",
    "interpolate_pilot_estimates",
    "make_pilot_mask",
    "pilot_evm",
    "remove_pilot_resources",
]

