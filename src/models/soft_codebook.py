from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _validate_temperature(temperature: float) -> float:
    value = float(temperature)
    if value <= 0.0:
        raise ValueError("temperature must be positive")
    return value


def soft_codebook_projection(
    logits: Tensor,
    codebook: Tensor,
    temperature: float = 1.0,
    *,
    return_probabilities: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Project token logits to the expected continuous codebook embedding.

    A shared codebook has shape `[K,D]` and accepts logits `[...,K]`. A
    layer-specific codebook has shape `[L,K,D]` and accepts `[B,L,T,K]`.
    No hard token decision is made in this communication path.
    """
    temperature = _validate_temperature(temperature)
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")
    if not codebook.is_floating_point() or codebook.ndim not in (2, 3):
        raise ValueError("codebook must be floating point [K,D] or [L,K,D]")
    if logits.shape[-1] != codebook.shape[-2]:
        raise ValueError("logit vocabulary size must match codebook size K")

    probabilities = F.softmax(logits / temperature, dim=-1)
    if codebook.ndim == 2:
        projected = torch.einsum("...k,kd->...d", probabilities, codebook)
    else:
        if logits.ndim != 4 or logits.shape[1] != codebook.shape[0]:
            raise ValueError("layer-specific projection requires logits [B,L,T,K] and codebook [L,K,D]")
        projected = torch.einsum("bltk,lkd->bltd", probabilities, codebook)
    if return_probabilities:
        return projected, probabilities
    return projected


def continuous_latent_loss(
    reconstruction: Tensor,
    target: Tensor,
    layer_weights: Sequence[float] | Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Layer-weighted MSE for continuous `[B,L,T,D]` codec tensors.

    `reduction='none'` returns one weighted loss per batch example.
    """
    if reconstruction.shape != target.shape or reconstruction.ndim != 4:
        raise ValueError("reconstruction and target must have matching [B,L,T,D] shapes")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    per_layer = (reconstruction - target).square().mean(dim=(2, 3))
    if layer_weights is None:
        weights = reconstruction.new_ones(reconstruction.shape[1])
    else:
        weights = torch.as_tensor(
            layer_weights,
            device=reconstruction.device,
            dtype=reconstruction.dtype,
        )
    if weights.shape != (reconstruction.shape[1],) or torch.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("layer_weights must contain L nonnegative values with positive sum")
    per_example = (per_layer * (weights / weights.sum())[None, :]).sum(dim=1)
    if reduction == "none":
        return per_example
    return per_example.mean() if reduction == "mean" else per_example.sum()


@torch.no_grad()
def top_k_token_accuracy(
    logits: Tensor,
    targets: Tensor,
    k: int = 1,
    *,
    ignore_index: int | None = None,
) -> Tensor:
    """Compute top-k token accuracy for offline analysis only."""
    if logits.shape[:-1] != targets.shape:
        raise ValueError("targets must match every non-vocabulary logit dimension")
    if targets.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError("targets must contain integer token labels")
    if not 1 <= k <= logits.shape[-1]:
        raise ValueError("k must be between 1 and the vocabulary size")
    valid = torch.ones_like(targets, dtype=torch.bool)
    if ignore_index is not None:
        valid = targets != ignore_index
    if not torch.any(valid):
        return logits.new_zeros(())
    candidates = logits.topk(k, dim=-1).indices
    correct = (candidates == targets.unsqueeze(-1)).any(dim=-1)
    return correct[valid].to(logits.dtype).mean()


class SoftCodebook(nn.Module):
    """Differentiable codebook expectation layer."""

    def __init__(
        self,
        codebook: Tensor,
        temperature: float = 1.0,
        *,
        trainable: bool = False,
    ):
        super().__init__()
        if not codebook.is_floating_point() or codebook.ndim not in (2, 3):
            raise ValueError("codebook must be floating point [K,D] or [L,K,D]")
        self.temperature = _validate_temperature(temperature)
        value = codebook.detach().clone()
        if trainable:
            self.codebook = nn.Parameter(value)
        else:
            self.register_buffer("codebook", value)

    def forward(
        self,
        logits: Tensor,
        temperature: float | None = None,
        *,
        return_probabilities: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        return soft_codebook_projection(
            logits,
            self.codebook,
            self.temperature if temperature is None else temperature,
            return_probabilities=return_probabilities,
        )


class CodecRepresentationLoss(nn.Module):
    """Two-mode objective for continuous or soft-codebook reconstruction.

    In `continuous` mode, predictions are already `[B,L,T,D]`. In
    `soft_codebook` mode, predictions are `[B,L,T,K]` logits and are converted
    to continuous embeddings before the same latent MSE is evaluated.
    """

    MODES = {"continuous", "soft_codebook"}

    def __init__(
        self,
        mode: str = "continuous",
        *,
        codebook: Tensor | None = None,
        temperature: float = 1.0,
        layer_weights: Sequence[float] | Tensor | None = None,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")
        if mode == "soft_codebook" and codebook is None:
            raise ValueError("soft_codebook mode requires a codebook")
        self.mode = mode
        self.projector = SoftCodebook(codebook, temperature) if codebook is not None else None
        weights = torch.as_tensor([] if layer_weights is None else layer_weights, dtype=torch.float32)
        self.register_buffer("layer_weights", weights)

    def reconstruct(self, prediction: Tensor) -> Tensor:
        if self.mode == "continuous":
            return prediction
        if self.projector is None:  # Defensive; constructor prevents this state.
            raise RuntimeError("soft codebook projector is unavailable")
        return self.projector(prediction)

    def forward(self, prediction: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
        reconstruction = self.reconstruct(prediction)
        weights = self.layer_weights if self.layer_weights.numel() else None
        return continuous_latent_loss(reconstruction, target, weights, reduction)


SoftCodebookProjection = SoftCodebook

__all__ = [
    "CodecRepresentationLoss",
    "SoftCodebook",
    "SoftCodebookProjection",
    "continuous_latent_loss",
    "soft_codebook_projection",
    "top_k_token_accuracy",
]

