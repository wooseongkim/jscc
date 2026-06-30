from __future__ import annotations

import torch
from torch import Tensor, nn


class LearnedLayerGate(nn.Module):
    """Channel-conditioned codec-layer gate `alpha = sigmoid(MLP(c))`."""

    def __init__(self, channel_state_dim: int, num_codec_layers: int, hidden_dim: int = 32):
        super().__init__()
        if min(channel_state_dim, num_codec_layers, hidden_dim) <= 0:
            raise ValueError("gate dimensions must be positive")
        self.channel_state_dim = channel_state_dim
        self.num_codec_layers = num_codec_layers
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(channel_state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_codec_layers),
        )

    def forward(self, channel_state: Tensor) -> Tensor:
        if channel_state.ndim != 2 or channel_state.shape[1] != self.channel_state_dim:
            raise ValueError(f"channel_state must have shape [B,{self.channel_state_dim}]")
        return torch.sigmoid(self.network(channel_state))


def gate_budget_loss(alpha: Tensor) -> Tensor:
    """Mean active-layer budget `mean_b sum_l alpha[b,l]`."""
    if alpha.ndim != 2:
        raise ValueError("alpha must have shape [B,L]")
    return alpha.sum(dim=1).mean()


def gate_smoothness_loss(alpha: Tensor) -> Tensor:
    """Mean adjacent-layer total variation."""
    if alpha.ndim != 2:
        raise ValueError("alpha must have shape [B,L]")
    if alpha.shape[1] < 2:
        return alpha.new_zeros(())
    return (alpha[:, 1:] - alpha[:, :-1]).abs().sum(dim=1).mean()


def save_learned_gate_checkpoint(gate: LearnedLayerGate) -> dict[str, object]:
    return {
        "state_dict": gate.state_dict(),
        "channel_state_dim": gate.channel_state_dim,
        "num_codec_layers": gate.num_codec_layers,
        "hidden_dim": gate.hidden_dim,
    }


def load_learned_gate_checkpoint(payload: dict[str, object], device: torch.device) -> LearnedLayerGate:
    required = {"state_dict", "channel_state_dim", "num_codec_layers", "hidden_dim"}
    if not required.issubset(payload):
        raise ValueError(f"learned gate checkpoint requires keys {sorted(required)}")
    gate = LearnedLayerGate(
        int(payload["channel_state_dim"]),
        int(payload["num_codec_layers"]),
        int(payload["hidden_dim"]),
    ).to(device)
    gate.load_state_dict(payload["state_dict"])
    return gate


__all__ = [
    "LearnedLayerGate",
    "gate_budget_loss",
    "gate_smoothness_loss",
    "load_learned_gate_checkpoint",
    "save_learned_gate_checkpoint",
]

