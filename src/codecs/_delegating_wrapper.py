from __future__ import annotations

from typing import Any

from torch import Tensor

from .base_codec import BaseCodec
from .mock_codec import MockContinuousCodec


class DelegatingCodecWrapper(BaseCodec):
    """Common placeholder behavior for optional external codec integrations."""

    external_name = "external codec"

    def __init__(
        self,
        codec: BaseCodec | None,
        fallback_to_mock: bool,
        mock_config: dict[str, Any] | None,
    ):
        super().__init__()
        if codec is None:
            if not fallback_to_mock:
                raise ImportError(
                    f"{self.external_name} adapter was not supplied; install/configure it "
                    "or enable fallback_to_mock"
                )
            codec = MockContinuousCodec(**(mock_config or {}))
            self.using_mock = True
        elif not isinstance(codec, BaseCodec):
            raise TypeError("codec must implement BaseCodec")
        else:
            self.using_mock = False
        self.codec = codec

    @property
    def representation_shape(self) -> tuple[int, int, int]:
        return self.codec.representation_shape

    def get_codebook(self) -> Tensor | None:
        return self.codec.get_codebook()

    def encode_waveform(self, waveform: Tensor) -> Tensor:
        return self.codec.encode_waveform(waveform)

    def decode_representation(self, representation: Tensor) -> Tensor:
        return self.codec.decode_representation(representation)
