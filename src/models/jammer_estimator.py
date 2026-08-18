"""Receiver-observable jammer type and resource-mask estimation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


JAMMER_TYPE_CLASSES: tuple[str, ...] = (
    "no_jammer",
    "broadband_awgn",
    "subband",
    "burst",
    "block",
    "tone",
)


@dataclass(frozen=True)
class JammerEstimate:
    """Inference-only jammer estimate; no ground-truth fields are retained."""

    posterior: Tensor
    mask_logits: Tensor
    mask_prob: Tensor
    mask_ratio: Tensor


class JammerEstimator(nn.Module):
    """Estimate jammer type and a soft active-grid mask from receiver observations.

    The public forward signature intentionally excludes true jammer type and
    true jammer mask. Those values are valid only in ``jammer_estimation_loss``.
    """

    def __init__(
        self,
        *,
        num_jammer_types: int = len(JAMMER_TYPE_CLASSES),
        hidden_dim: int = 32,
        jammer_type_classes: tuple[str, ...] | None = None,
    ):
        super().__init__()
        if num_jammer_types <= 1:
            raise ValueError("num_jammer_types must exceed one")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.num_jammer_types = int(num_jammer_types)
        self.hidden_dim = int(hidden_dim)
        self.jammer_type_classes = tuple(jammer_type_classes or JAMMER_TYPE_CLASSES)
        if len(self.jammer_type_classes) != self.num_jammer_types:
            raise ValueError("jammer_type_classes length must match num_jammer_types")
        if len(set(self.jammer_type_classes)) != len(self.jammer_type_classes):
            raise ValueError("jammer_type_classes must be unique")
        # received, pilot residual, estimated channel (real/imag), pilot mask,
        # and observable noise variance: 8 receiver-side feature planes.
        self.features = nn.Sequential(
            nn.Conv2d(8, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.mask_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.posterior_head = nn.Linear(hidden_dim, self.num_jammer_types)

    @staticmethod
    def _broadcast_pilot_mask(pilot_mask: Tensor, *, batch: int, carriers: int, symbols: int, device: torch.device) -> Tensor:
        if pilot_mask.ndim == 2:
            pilot_mask = pilot_mask.unsqueeze(0).expand(batch, -1, -1)
        if pilot_mask.shape != (batch, carriers, symbols):
            raise ValueError("pilot_mask must be [K,N] or [B,K,N] matching received_grid")
        return pilot_mask.to(device=device, dtype=torch.float32)

    def forward(
        self,
        received_grid: Tensor,
        pilots: Tensor,
        pilot_mask: Tensor,
        estimated_channel: Tensor,
        noise_variance: Tensor | float,
    ) -> JammerEstimate:
        if received_grid.ndim != 3 or not torch.is_complex(received_grid):
            raise ValueError("received_grid must be complex [B,K,N]")
        if pilots.shape != received_grid.shape or estimated_channel.shape != received_grid.shape:
            raise ValueError("pilots and estimated_channel must match received_grid")
        batch, carriers, symbols = received_grid.shape
        mask = self._broadcast_pilot_mask(
            pilot_mask, batch=batch, carriers=carriers, symbols=symbols,
            device=received_grid.device,
        ).to(dtype=received_grid.real.dtype)
        residual = received_grid - estimated_channel * pilots
        noise = torch.as_tensor(noise_variance, device=received_grid.device, dtype=received_grid.real.dtype)
        if noise.ndim == 0:
            noise = noise.expand(batch)
        if noise.shape != (batch,):
            raise ValueError("noise_variance must be scalar or [B]")
        noise_plane = noise.clamp_min(torch.finfo(noise.dtype).eps).sqrt()[:, None, None].expand(-1, carriers, symbols)
        features = torch.stack(
            (
                received_grid.real, received_grid.imag,
                residual.real, residual.imag,
                estimated_channel.real, estimated_channel.imag,
                mask, noise_plane,
            ),
            dim=1,
        )
        hidden = self.features(features)
        mask_logits = self.mask_head(hidden).squeeze(1)
        mask_prob = torch.sigmoid(mask_logits)
        posterior = torch.softmax(self.posterior_head(hidden.mean(dim=(-2, -1))), dim=-1)
        return JammerEstimate(
            posterior=posterior,
            mask_logits=mask_logits,
            mask_prob=mask_prob,
            mask_ratio=mask_prob.mean(dim=(-2, -1)),
        )


def jammer_estimation_loss(
    estimate: JammerEstimate,
    *,
    jammer_type: Tensor,
    jammer_mask: Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    pos_weight: float | None = None,
    epsilon: float = 1e-8,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Supervised loss; labels remain outside estimator inference inputs."""
    if jammer_type.shape != (estimate.posterior.shape[0],):
        raise ValueError("jammer_type must be [B]")
    if jammer_mask.shape != estimate.mask_logits.shape:
        raise ValueError("jammer_mask must match mask_logits")
    if jammer_type.dtype != torch.long:
        jammer_type = jammer_type.long()
    type_ce = F.nll_loss(estimate.posterior.clamp_min(epsilon).log(), jammer_type)
    target = jammer_mask.to(dtype=estimate.mask_logits.dtype)
    bce_pos_weight = None if pos_weight is None else estimate.mask_logits.new_tensor(float(pos_weight))
    mask_bce = F.binary_cross_entropy_with_logits(
        estimate.mask_logits, target, pos_weight=bce_pos_weight,
    )
    intersection = (estimate.mask_prob * target).sum(dim=(-2, -1))
    denominator = estimate.mask_prob.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    mask_dice = (1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)).mean()
    return type_ce + float(bce_weight) * mask_bce + float(dice_weight) * mask_dice, {
        "type_ce": type_ce,
        "mask_bce": mask_bce,
        "mask_dice": mask_dice,
    }


def save_jammer_estimator_checkpoint(estimator: JammerEstimator) -> dict[str, object]:
    return {
        "state_dict": estimator.state_dict(),
        "num_jammer_types": estimator.num_jammer_types,
        "hidden_dim": estimator.hidden_dim,
        "jammer_type_classes": list(estimator.jammer_type_classes),
    }


def load_jammer_estimator_checkpoint(
    payload: dict[str, object], device: torch.device,
) -> JammerEstimator:
    required = {"state_dict", "num_jammer_types", "hidden_dim", "jammer_type_classes"}
    if not required.issubset(payload):
        raise ValueError(f"jammer estimator checkpoint requires keys {sorted(required)}")
    estimator = JammerEstimator(
        num_jammer_types=int(payload["num_jammer_types"]),
        hidden_dim=int(payload["hidden_dim"]),
        jammer_type_classes=tuple(payload["jammer_type_classes"]),
    ).to(device)
    estimator.load_state_dict(payload["state_dict"])
    return estimator


__all__ = [
    "JAMMER_TYPE_CLASSES",
    "JammerEstimate",
    "JammerEstimator",
    "jammer_estimation_loss",
    "load_jammer_estimator_checkpoint",
    "save_jammer_estimator_checkpoint",
]
