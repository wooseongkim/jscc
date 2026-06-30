from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class BaseCodec(nn.Module, ABC):
    """Interface exposing continuous codec representations to the JSCC path."""

    @abstractmethod
    def encode_waveform(self, waveform: Tensor) -> Tensor:
        """Encode real waveforms `[B,S]` as continuous tensors `[B,L,T,D]`."""

    @abstractmethod
    def decode_representation(self, representation: Tensor) -> Tensor:
        """Decode continuous representations `[B,L,T,D]` to waveforms `[B,S]`."""

    @abstractmethod
    def get_codebook(self) -> Tensor | None:
        """Return optional embedding weights `[L,K,D]`; never serialized indices."""

    @property
    @abstractmethod
    def representation_shape(self) -> tuple[int, int, int]:
        """Return the fixed `(layers, frames, latent_dim)` representation shape."""

