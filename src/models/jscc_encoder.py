from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn


ChannelShape = int | tuple[int, int]


def _channel_shape_and_uses(channel_shape: ChannelShape) -> tuple[tuple[int, ...], int]:
    shape = (channel_shape,) if isinstance(channel_shape, int) else tuple(channel_shape)
    if len(shape) not in (1, 2) or any(size <= 0 for size in shape):
        raise ValueError("channel shape must be M or (K, N), with positive dimensions")
    return shape, math.prod(shape)


def normalize_complex_power(
    symbols: Tensor,
    target_power: float = 1.0,
    eps: float = 1e-12,
) -> Tensor:
    """Set mean `|x|^2` to `target_power` independently per example."""
    if not torch.is_complex(symbols) or symbols.ndim < 2:
        raise TypeError("symbols must be a batched complex tensor")
    if target_power <= 0:
        raise ValueError("target_power must be positive")
    dimensions = tuple(range(1, symbols.ndim))
    power = symbols.abs().square().mean(dimensions, keepdim=True)
    scale = torch.sqrt(
        torch.as_tensor(target_power, device=symbols.device, dtype=power.dtype)
        / power.clamp_min(eps)
    )
    return symbols * scale


def deterministic_layer_gates(
    channel_state: Tensor,
    num_layers: int,
    thresholds: Sequence[float] | Tensor | None = None,
    quality_index: int = 0,
) -> Tensor:
    """Return deterministic prefix gates from one channel-quality feature.

    Layer zero is always active. Each increasing threshold activates one more
    codec layer. With no thresholds, all layers are active.
    """
    if channel_state.ndim != 2:
        raise ValueError("channel_state must have shape [B, C]")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if not 0 <= quality_index < channel_state.shape[1]:
        raise ValueError("quality_index is outside the channel-state vector")
    if thresholds is None:
        return channel_state.new_ones((channel_state.shape[0], num_layers))
    threshold_tensor = torch.as_tensor(
        thresholds,
        device=channel_state.device,
        dtype=channel_state.dtype,
    ).flatten()
    if threshold_tensor.numel() != num_layers - 1:
        raise ValueError("gate thresholds must contain exactly L-1 values")
    if threshold_tensor.numel() > 1 and torch.any(threshold_tensor[1:] < threshold_tensor[:-1]):
        raise ValueError("gate thresholds must be nondecreasing")
    additional = channel_state[:, quality_index, None] >= threshold_tensor[None, :]
    return torch.cat(
        (channel_state.new_ones((channel_state.shape[0], 1)), additional.to(channel_state.dtype)),
        dim=1,
    )


class JSCCEncoder(nn.Module):
    """Map continuous codec layers `[B,L,T,D]` to complex channel symbols.

    Channel uses are partitioned between codec layers. Optional deterministic
    gates disable enhancement layers based on the channel-state vector. Power
    weights control the relative energy assigned to active layer partitions;
    final normalization enforces the total average-power constraint exactly.
    """

    def __init__(
        self,
        representation_shape: tuple[int, int, int],
        channel_uses: ChannelShape,
        channel_state_dim: int = 2,
        hidden_dim: int = 128,
        target_power: float = 1.0,
        *,
        gate_thresholds: Sequence[float] | None = None,
        quality_index: int = 0,
        layer_power_allocation: Sequence[float] | None = None,
    ):
        super().__init__()
        if len(representation_shape) != 3 or any(size <= 0 for size in representation_shape):
            raise ValueError("representation_shape must be positive (L, T, D)")
        self.representation_shape = tuple(representation_shape)
        self.num_layers, self.frames, self.latent_dim = self.representation_shape
        self.channel_shape, self.total_channel_uses = _channel_shape_and_uses(channel_uses)
        if self.total_channel_uses < self.num_layers:
            raise ValueError("at least one channel use is required per codec layer")
        if channel_state_dim <= 0 or hidden_dim <= 0 or target_power <= 0:
            raise ValueError("state dimension, hidden dimension, and target power must be positive")
        self.channel_state_dim = channel_state_dim
        self.target_power = float(target_power)
        self.quality_index = quality_index

        base, remainder = divmod(self.total_channel_uses, self.num_layers)
        self.layer_channel_uses = tuple(
            base + (1 if layer < remainder else 0) for layer in range(self.num_layers)
        )
        branch_input_dim = self.frames * self.latent_dim + channel_state_dim
        self.layer_encoders = nn.ModuleList(
            nn.Sequential(
                nn.Linear(branch_input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2 * uses),
            )
            for uses in self.layer_channel_uses
        )

        thresholds = [] if gate_thresholds is None else list(gate_thresholds)
        if thresholds and len(thresholds) != self.num_layers - 1:
            raise ValueError("gate_thresholds must contain L-1 values")
        self.register_buffer("gate_thresholds", torch.tensor(thresholds, dtype=torch.float32))
        allocation = (
            torch.ones(self.num_layers, dtype=torch.float32)
            if layer_power_allocation is None
            else torch.tensor(layer_power_allocation, dtype=torch.float32)
        )
        if allocation.shape != (self.num_layers,) or torch.any(allocation < 0) or allocation.sum() <= 0:
            raise ValueError("layer_power_allocation must contain L nonnegative values with positive sum")
        self.register_buffer("default_layer_power", allocation)

    def _gates(self, channel_state: Tensor, layer_gates: Tensor | None) -> Tensor:
        if layer_gates is None:
            thresholds = self.gate_thresholds if self.gate_thresholds.numel() else None
            return deterministic_layer_gates(
                channel_state,
                self.num_layers,
                thresholds,
                self.quality_index,
            )
        gates = layer_gates.to(device=channel_state.device, dtype=channel_state.dtype)
        if gates.shape != (channel_state.shape[0], self.num_layers):
            raise ValueError("layer_gates must have shape [B, L]")
        return gates.clamp(0.0, 1.0)

    def _power_fractions(
        self,
        gates: Tensor,
        allocation: Tensor | None,
    ) -> Tensor:
        if allocation is None:
            weights = self.default_layer_power[None, :].expand_as(gates)
        else:
            weights = allocation.to(device=gates.device, dtype=gates.dtype)
            if weights.ndim == 1:
                weights = weights[None, :].expand_as(gates)
            if weights.shape != gates.shape or torch.any(weights < 0):
                raise ValueError("layer_power_allocation must have shape [L] or [B, L] and be nonnegative")
        weights = weights * gates.square()
        total = weights.sum(dim=1, keepdim=True)
        fallback = torch.zeros_like(weights)
        fallback[:, 0] = 1.0
        return torch.where(total > 1e-12, weights / total.clamp_min(1e-12), fallback)

    def forward(
        self,
        representation: Tensor,
        channel_state: Tensor,
        *,
        layer_gates: Tensor | None = None,
        layer_power_allocation: Tensor | None = None,
        return_aux: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if representation.ndim != 4 or tuple(representation.shape[1:]) != self.representation_shape:
            raise ValueError(f"representation must have shape [B, {self.representation_shape}]")
        if channel_state.shape != (representation.shape[0], self.channel_state_dim):
            raise ValueError(f"channel_state must have shape [B, {self.channel_state_dim}]")
        gates = self._gates(channel_state, layer_gates)
        power_fractions = self._power_fractions(gates, layer_power_allocation)

        partitions = []
        for layer, (network, uses) in enumerate(zip(self.layer_encoders, self.layer_channel_uses)):
            gated_latent = representation[:, layer].flatten(1) * gates[:, layer, None]
            features = torch.cat((gated_latent, channel_state), dim=1)
            real_imag = network(features).reshape(representation.shape[0], uses, 2)
            branch = normalize_complex_power(torch.view_as_complex(real_imag.contiguous()), 1.0)
            branch_target_power = (
                self.target_power
                * self.total_channel_uses
                * power_fractions[:, layer]
                / uses
            )
            partitions.append(branch * torch.sqrt(branch_target_power[:, None]))

        symbols = torch.cat(partitions, dim=1).reshape(representation.shape[0], *self.channel_shape)
        symbols = normalize_complex_power(symbols, self.target_power)
        if return_aux:
            return symbols, {"layer_gates": gates, "layer_power_fractions": power_fractions}
        return symbols

