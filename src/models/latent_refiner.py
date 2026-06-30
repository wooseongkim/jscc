from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LatentRefiner(nn.Module):
    """Lightweight residual temporal refiner conditioned on channel state and mask."""

    def __init__(
        self,
        representation_shape: tuple[int, int, int],
        channel_state_dim: int,
        hidden_dim: int = 64,
        state_features: int = 16,
    ):
        super().__init__()
        if len(representation_shape) != 3 or min(*representation_shape) <= 0:
            raise ValueError("representation_shape must be positive (L,T,D)")
        self.representation_shape = tuple(representation_shape)
        self.channel_state_dim = channel_state_dim
        self.hidden_dim = hidden_dim
        layers, _, latent_dim = representation_shape
        latent_channels = layers * latent_dim
        self.state_projection = nn.Sequential(
            nn.Linear(channel_state_dim, state_features), nn.GELU()
        )
        self.input_conv = nn.Conv1d(latent_channels + state_features + 1, hidden_dim, 3, padding=1)
        self.hidden_conv = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.output_conv = nn.Conv1d(hidden_dim, latent_channels, 3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def temporal_mask(self, resource_mask: Tensor, frames: int, dtype: torch.dtype) -> Tensor:
        if resource_mask.ndim not in (2, 3):
            raise ValueError("resource_mask must be [B,M] or [B,K,N]")
        flattened = resource_mask.to(dtype).flatten(1).unsqueeze(1)
        return F.adaptive_avg_pool1d(flattened, frames)

    def forward(
        self,
        noisy_latent: Tensor,
        channel_state: Tensor,
        resource_mask: Tensor,
    ) -> Tensor:
        if noisy_latent.ndim != 4 or tuple(noisy_latent.shape[1:]) != self.representation_shape:
            raise ValueError(f"noisy_latent must have shape [B,{self.representation_shape}]")
        if channel_state.shape != (noisy_latent.shape[0], self.channel_state_dim):
            raise ValueError(f"channel_state must have shape [B,{self.channel_state_dim}]")
        batch, layers, frames, latent_dim = noisy_latent.shape
        latent_features = noisy_latent.permute(0, 1, 3, 2).reshape(
            batch, layers * latent_dim, frames
        )
        state_features = self.state_projection(channel_state).unsqueeze(-1).expand(-1, -1, frames)
        mask_features = self.temporal_mask(resource_mask, frames, noisy_latent.dtype)
        features = torch.cat((latent_features, state_features, mask_features), dim=1)
        hidden = F.gelu(self.input_conv(features))
        hidden = F.gelu(self.hidden_conv(hidden))
        delta = self.output_conv(hidden).reshape(batch, layers, latent_dim, frames).permute(0, 1, 3, 2)
        return noisy_latent + delta


def save_latent_refiner_checkpoint(refiner: LatentRefiner) -> dict[str, object]:
    return {
        "state_dict": refiner.state_dict(),
        "representation_shape": refiner.representation_shape,
        "channel_state_dim": refiner.channel_state_dim,
        "hidden_dim": refiner.hidden_dim,
    }


def load_latent_refiner_checkpoint(payload: dict[str, object], device: torch.device) -> LatentRefiner:
    required = {"state_dict", "representation_shape", "channel_state_dim", "hidden_dim"}
    if not required.issubset(payload):
        raise ValueError(f"latent refiner checkpoint requires keys {sorted(required)}")
    refiner = LatentRefiner(
        tuple(payload["representation_shape"]),
        int(payload["channel_state_dim"]),
        int(payload["hidden_dim"]),
    ).to(device)
    refiner.load_state_dict(payload["state_dict"])
    return refiner


__all__ = [
    "LatentRefiner",
    "load_latent_refiner_checkpoint",
    "save_latent_refiner_checkpoint",
]
