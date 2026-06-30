"""Compatibility imports for :mod:`models.soft_codebook`."""

from models.soft_codebook import (
    CodecRepresentationLoss,
    SoftCodebook,
    SoftCodebookProjection,
    continuous_latent_loss,
    soft_codebook_projection,
    top_k_token_accuracy,
)

__all__ = [
    "CodecRepresentationLoss",
    "SoftCodebook",
    "SoftCodebookProjection",
    "continuous_latent_loss",
    "soft_codebook_projection",
    "top_k_token_accuracy",
]

