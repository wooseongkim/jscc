from __future__ import annotations

import torch
from torch import Tensor


OBSERVABLE_RECEIVER_STATE_FEATURES = (
    "mean_estimated_channel_power",
    "std_estimated_channel_power_db",
    "q05_estimated_channel_power_db",
    "mean_equalizer_gain_power",
    "q95_equalizer_gain_power",
    "log1p_pilot_evm",
    "pilot_residual_ratio",
    "estimated_deep_fade_fraction",
)


def _broadcast_mask(mask: Tensor, reference: Tensor, name: str) -> Tensor:
    try:
        return torch.broadcast_to(mask.to(device=reference.device, dtype=torch.bool), reference.shape)
    except RuntimeError as error:
        raise ValueError(f"{name} is not broadcastable to {tuple(reference.shape)}") from error


def _masked_values(values: Tensor, mask: Tensor, name: str) -> list[Tensor]:
    if values.shape != mask.shape:
        raise ValueError(f"{name} values and mask shapes must match")
    result: list[Tensor] = []
    flat_values = values.reshape(values.shape[0], -1)
    flat_mask = mask.reshape(mask.shape[0], -1)
    for batch_index in range(values.shape[0]):
        selected = flat_values[batch_index][flat_mask[batch_index]]
        if selected.numel() == 0:
            raise ValueError(f"{name} mask selects no resources for batch item {batch_index}")
        result.append(selected)
    return result


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    selected = _masked_values(values, mask, "mean")
    return torch.stack([item.mean() for item in selected])


def _masked_std(values: Tensor, mask: Tensor) -> Tensor:
    selected = _masked_values(values, mask, "std")
    return torch.stack(
        [item.std(unbiased=False) if item.numel() > 1 else item.new_zeros(()) for item in selected]
    )


def _masked_quantile(values: Tensor, mask: Tensor, q: float) -> Tensor:
    selected = _masked_values(values, mask, "quantile")
    return torch.stack([torch.quantile(item, q) for item in selected])


def _masked_sum(values: Tensor, mask: Tensor) -> Tensor:
    selected = _masked_values(values, mask, "sum")
    return torch.stack([item.sum() for item in selected])


def build_observable_receiver_state_v1(
    received: Tensor,
    pilots: Tensor,
    pilot_mask: Tensor,
    estimated_channel: Tensor,
    *,
    eps: float = 1.0e-8,
) -> Tensor:
    """Build the Stage-1 receiver state from observable pilot/CSI tensors only.

    This state intentionally does not accept true channel, jammer labels, jammer
    masks, requested SNR/JSR, true CSI NMSE, or simulator-separated SINR. Pilot
    EVM here is an in-sample conditioning feature for the decoder, not a fair
    standalone estimator-comparison metric.
    """
    if not (torch.is_complex(received) and torch.is_complex(pilots) and torch.is_complex(estimated_channel)):
        raise TypeError("received, pilots, and estimated_channel must be complex tensors")
    if received.shape != pilots.shape or received.shape != estimated_channel.shape:
        raise ValueError("received, pilots, and estimated_channel must share shape [B,...]")
    if received.ndim not in (2, 3):
        raise ValueError("observable receiver state expects [B,M] or [B,K,N] tensors")

    pilots_mask = _broadcast_mask(pilot_mask, received, "pilot_mask")
    data_mask = ~pilots_mask
    h_power = estimated_channel.abs().square()
    h_power_db = 10.0 * torch.log10(h_power.clamp_min(eps))
    equalizer = estimated_channel.conj() / h_power.clamp_min(eps)
    gain_power = equalizer.abs().square()

    pilot_residual = received - estimated_channel * pilots
    pilot_reference = estimated_channel * pilots
    pilot_evm = torch.sqrt(
        _masked_sum(pilot_residual.abs().square(), pilots_mask)
        / _masked_sum(pilot_reference.abs().square(), pilots_mask).clamp_min(eps)
    )
    residual_ratio = (
        _masked_mean(pilot_residual.abs().square(), pilots_mask)
        / _masked_mean(received.abs().square(), pilots_mask).clamp_min(eps)
    )
    mean_h_power = _masked_mean(h_power, data_mask)
    threshold = 0.1 * mean_h_power
    deep_fade = torch.stack(
        [
            (values < threshold[index]).to(h_power.dtype).mean()
            for index, values in enumerate(_masked_values(h_power, data_mask, "deep_fade"))
        ]
    )

    features = (
        (10.0 * torch.log10(mean_h_power.clamp_min(eps)) / 20.0).clamp(-3.0, 3.0),
        (_masked_std(h_power_db, data_mask) / 20.0).clamp(0.0, 3.0),
        (_masked_quantile(h_power_db, data_mask, 0.05) / 20.0).clamp(-3.0, 3.0),
        (10.0 * torch.log10(_masked_mean(gain_power, data_mask).clamp_min(eps)) / 20.0).clamp(-3.0, 3.0),
        (10.0 * torch.log10(_masked_quantile(gain_power, data_mask, 0.95).clamp_min(eps)) / 20.0).clamp(-3.0, 3.0),
        torch.log1p(pilot_evm).clamp(0.0, 3.0),
        (10.0 * torch.log10(residual_ratio.clamp_min(eps)) / 20.0).clamp(-3.0, 3.0),
        deep_fade.clamp(0.0, 1.0),
    )
    state = torch.stack(features, dim=1)
    if not torch.isfinite(state).all():
        raise RuntimeError("observable receiver state contains nonfinite values")
    return state


__all__ = ["OBSERVABLE_RECEIVER_STATE_FEATURES", "build_observable_receiver_state_v1"]
