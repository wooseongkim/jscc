from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class BaseCodec(nn.Module, ABC):
    """Install-safe continuous codec interface."""

    @abstractmethod
    def encode_waveform(self, waveform: Tensor) -> Tensor:
        """Encode `[B,S]` waveforms to continuous `[B,L,T,D]` tensors."""

    @abstractmethod
    def decode_representation(self, representation: Tensor) -> Tensor:
        """Decode continuous `[B,L,T,D]` tensors to `[B,S]` waveforms."""

    @abstractmethod
    def get_codebook(self) -> Tensor | None:
        """Return optional continuous embedding weights `[L,K,D]`."""

    @property
    @abstractmethod
    def representation_shape(self) -> tuple[int, int, int]:
        """Return `(layers, frames, latent_dim)`."""

__all__ = ["BaseCodec"]
