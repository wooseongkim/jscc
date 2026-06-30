from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from models.jscc_encoder import ChannelShape, _channel_shape_and_uses, deterministic_layer_gates


class JSCCDecoder(nn.Module):
    """Reconstruct `[B,L,T,D]` continuous codec tensors from complex symbols."""

    def __init__(
        self,
        representation_shape: tuple[int, int, int],
        channel_uses: ChannelShape,
        channel_state_dim: int = 2,
        hidden_dim: int = 128,
        *,
        gate_thresholds: Sequence[float] | None = None,
        quality_index: int = 0,
        apply_output_gates: bool = False,
    ):
        super().__init__()
        if len(representation_shape) != 3 or any(size <= 0 for size in representation_shape):
            raise ValueError("representation_shape must be positive (L, T, D)")
        self.representation_shape = tuple(representation_shape)
        self.channel_shape, total_channel_uses = _channel_shape_and_uses(channel_uses)
        self.channel_state_dim = channel_state_dim
        self.quality_index = quality_index
        self.apply_output_gates = apply_output_gates
        output_dim = math.prod(self.representation_shape)
        self.network = nn.Sequential(
            nn.Linear(2 * total_channel_uses + channel_state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        thresholds = [] if gate_thresholds is None else list(gate_thresholds)
        if thresholds and len(thresholds) != self.representation_shape[0] - 1:
            raise ValueError("gate_thresholds must contain L-1 values")
        self.register_buffer("gate_thresholds", torch.tensor(thresholds, dtype=torch.float32))

    def forward(
        self,
        received: Tensor,
        channel_state: Tensor,
        *,
        layer_gates: Tensor | None = None,
    ) -> Tensor:
        if not torch.is_complex(received) or tuple(received.shape[1:]) != self.channel_shape:
            raise ValueError(f"received must be complex [B, {self.channel_shape}]")
        if channel_state.shape != (received.shape[0], self.channel_state_dim):
            raise ValueError(f"channel_state must have shape [B, {self.channel_state_dim}]")
        features = torch.cat((torch.view_as_real(received).flatten(1), channel_state), dim=1)
        reconstruction = self.network(features).reshape(received.shape[0], *self.representation_shape)
        if not self.apply_output_gates and layer_gates is None:
            return reconstruction
        if layer_gates is None:
            thresholds = self.gate_thresholds if self.gate_thresholds.numel() else None
            layer_gates = deterministic_layer_gates(
                channel_state,
                self.representation_shape[0],
                thresholds,
                self.quality_index,
            )
        if layer_gates.shape != (received.shape[0], self.representation_shape[0]):
            raise ValueError("layer_gates must have shape [B, L]")
        return reconstruction * layer_gates[:, :, None, None].to(reconstruction)

