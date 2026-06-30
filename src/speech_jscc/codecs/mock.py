from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from speech_jscc.codecs.base import BaseCodec


class MockContinuousCodec(BaseCodec):
    """Seeded continuous mock codec used by the installed JSCC package."""

    def __init__(
        self,
        layers: int = 4,
        frames: int = 12,
        latent_dim: int = 8,
        waveform_samples: int = 768,
        *,
        codebook_size: int = 32,
        seed: int = 0,
    ):
        super().__init__()
        if min(layers, frames, latent_dim, waveform_samples, codebook_size) <= 0:
            raise ValueError("all mock codec dimensions must be positive")
        self.layers = layers
        self.frames = frames
        self.latent_dim = latent_dim
        self.waveform_samples = waveform_samples
        generator = torch.Generator(device="cpu").manual_seed(seed)
        basis = torch.randn(layers, latent_dim, generator=generator)
        basis = basis / basis.square().mean(dim=1, keepdim=True).sqrt()
        self.register_buffer("analysis_basis", basis)
        self.register_buffer("latent_bias", 0.05 * torch.randn(layers, latent_dim, generator=generator))
        self.register_buffer(
            "codebook", torch.randn(layers, codebook_size, latent_dim, generator=generator)
        )

    @property
    def representation_shape(self) -> tuple[int, int, int]:
        return self.layers, self.frames, self.latent_dim

    def get_codebook(self) -> Tensor:
        return self.codebook

    def encode_waveform(self, waveform: Tensor) -> Tensor:
        if waveform.ndim != 2 or not waveform.is_floating_point():
            raise ValueError("waveform must be a real floating tensor [B,S]")
        frames = F.adaptive_avg_pool1d(waveform.unsqueeze(1), self.frames).squeeze(1)
        return (
            frames[:, None, :, None] * self.analysis_basis[None, :, None, :]
            + self.latent_bias[None, :, None, :]
        )

    def decode_representation(self, representation: Tensor) -> Tensor:
        if representation.ndim != 4 or tuple(representation.shape[1:]) != self.representation_shape:
            raise ValueError(f"representation must have shape [B,{self.representation_shape}]")
        centered = representation - self.latent_bias[None, :, None, :]
        basis = self.analysis_basis[None, :, None, :]
        frames = (centered * basis).sum(dim=(1, 3)) / basis.square().sum(
            dim=(1, 3)
        ).clamp_min(1e-12)
        return F.interpolate(
            frames.unsqueeze(1),
            size=self.waveform_samples,
            mode="linear",
            align_corners=False,
        ).squeeze(1)


MockCodec = MockContinuousCodec

__all__ = ["MockCodec", "MockContinuousCodec"]
