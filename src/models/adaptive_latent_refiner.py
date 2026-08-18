"""Posterior-gated mixture-of-experts latent residual denoiser."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _LatentResidualExpert(nn.Module):
    def __init__(self, latent_channels: int, state_features: int, hidden_dim: int):
        super().__init__()
        self.input_conv = nn.Conv1d(latent_channels + state_features + 1, hidden_dim, 3, padding=1)
        self.hidden_conv = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.output_conv = nn.Conv1d(hidden_dim, latent_channels, 3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, features: Tensor) -> Tensor:
        hidden = F.gelu(self.input_conv(features))
        hidden = F.gelu(self.hidden_conv(hidden))
        return self.output_conv(hidden)


class MoEAdaptiveLatentRefiner(nn.Module):
    """MoE residual refiner gated by a receiver-estimated jammer posterior."""

    def __init__(
        self,
        *,
        representation_shape: tuple[int, int, int],
        channel_state_dim: int,
        num_experts: int,
        hidden_dim: int = 64,
        state_features: int = 16,
    ):
        super().__init__()
        if len(representation_shape) != 3 or min(representation_shape) <= 0:
            raise ValueError("representation_shape must be positive (L,T,D)")
        if channel_state_dim <= 0 or num_experts <= 1:
            raise ValueError("channel_state_dim must be positive and num_experts must exceed one")
        self.representation_shape = tuple(int(value) for value in representation_shape)
        self.channel_state_dim = int(channel_state_dim)
        self.num_experts = int(num_experts)
        self.hidden_dim = int(hidden_dim)
        self.state_features = int(state_features)
        layers, _, latent_dim = self.representation_shape
        self.latent_channels = layers * latent_dim
        self.state_projection = nn.Sequential(nn.Linear(channel_state_dim, state_features), nn.GELU())
        self.experts = nn.ModuleList(
            _LatentResidualExpert(self.latent_channels, state_features, hidden_dim)
            for _ in range(num_experts)
        )

    @staticmethod
    def _temporal_mask(resource_mask: Tensor, frames: int, dtype: torch.dtype) -> Tensor:
        if resource_mask.ndim not in (2, 3):
            raise ValueError("mask_prob must be [B,M] or [B,K,N]")
        return F.adaptive_avg_pool1d(resource_mask.to(dtype).flatten(1).unsqueeze(1), frames)

    def forward(
        self,
        raw_latent: Tensor,
        decoder_state: Tensor,
        mask_prob: Tensor,
        jammer_posterior: Tensor,
    ) -> Tensor:
        if raw_latent.ndim != 4 or tuple(raw_latent.shape[1:]) != self.representation_shape:
            raise ValueError(f"raw_latent must be [B,{self.representation_shape}]")
        batch, layers, frames, latent_dim = raw_latent.shape
        if decoder_state.shape != (batch, self.channel_state_dim):
            raise ValueError(f"decoder_state must be [B,{self.channel_state_dim}]")
        if jammer_posterior.shape != (batch, self.num_experts):
            raise ValueError(f"jammer_posterior must be [B,{self.num_experts}]")
        if not torch.isfinite(jammer_posterior).all() or (jammer_posterior < 0).any():
            raise ValueError("jammer_posterior must be finite and nonnegative")
        normalized_posterior = jammer_posterior / jammer_posterior.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(raw_latent.dtype).eps)
        latent_features = raw_latent.permute(0, 1, 3, 2).reshape(batch, layers * latent_dim, frames)
        state_features = self.state_projection(decoder_state).unsqueeze(-1).expand(-1, -1, frames)
        mask_features = self._temporal_mask(mask_prob, frames, raw_latent.dtype)
        features = torch.cat((latent_features, state_features, mask_features), dim=1)
        deltas = torch.stack(
            [expert(features) for expert in self.experts], dim=1
        ).reshape(batch, self.num_experts, layers, latent_dim, frames).permute(0, 1, 2, 4, 3)
        delta = (normalized_posterior[:, :, None, None, None] * deltas).sum(dim=1)
        return raw_latent + delta


def no_jammer_identity_regularization(
    refined_latent: Tensor,
    raw_latent: Tensor,
    jammer_posterior: Tensor,
    *,
    no_jammer_index: int = 0,
) -> Tensor:
    """Penalize changes when the estimator assigns mass to no-jammer."""
    if refined_latent.shape != raw_latent.shape:
        raise ValueError("refined_latent and raw_latent must match")
    if jammer_posterior.ndim != 2 or not 0 <= no_jammer_index < jammer_posterior.shape[1]:
        raise ValueError("invalid no_jammer_index")
    residual_energy = (refined_latent - raw_latent).square().mean(dim=(1, 2, 3))
    return (jammer_posterior[:, no_jammer_index] * residual_energy).mean()


def save_adaptive_latent_refiner_checkpoint(
    refiner: MoEAdaptiveLatentRefiner,
) -> dict[str, object]:
    return {
        "state_dict": refiner.state_dict(),
        "representation_shape": refiner.representation_shape,
        "channel_state_dim": refiner.channel_state_dim,
        "num_experts": refiner.num_experts,
        "hidden_dim": refiner.hidden_dim,
        "state_features": refiner.state_features,
    }


def load_adaptive_latent_refiner_checkpoint(
    payload: dict[str, object], device: torch.device,
) -> MoEAdaptiveLatentRefiner:
    required = {
        "state_dict", "representation_shape", "channel_state_dim", "num_experts",
        "hidden_dim", "state_features",
    }
    if not required.issubset(payload):
        raise ValueError(f"adaptive refiner checkpoint requires keys {sorted(required)}")
    refiner = MoEAdaptiveLatentRefiner(
        representation_shape=tuple(payload["representation_shape"]),
        channel_state_dim=int(payload["channel_state_dim"]),
        num_experts=int(payload["num_experts"]),
        hidden_dim=int(payload["hidden_dim"]),
        state_features=int(payload["state_features"]),
    ).to(device)
    refiner.load_state_dict(payload["state_dict"])
    return refiner


__all__ = [
    "MoEAdaptiveLatentRefiner",
    "load_adaptive_latent_refiner_checkpoint",
    "no_jammer_identity_regularization",
    "save_adaptive_latent_refiner_checkpoint",
]
