from __future__ import annotations

from typing import Any

from ._delegating_wrapper import DelegatingCodecWrapper
from .base_codec import BaseCodec


class SpeechTokenizerWrapper(DelegatingCodecWrapper):
    """Placeholder for a SpeechTokenizer adapter with deterministic fallback.

    Pass a project-specific `BaseCodec` adapter in `codec` when pretrained
    SpeechTokenizer dependencies and weights are available.
    """

    external_name = "SpeechTokenizer"

    def __init__(
        self,
        codec: BaseCodec | None = None,
        *,
        fallback_to_mock: bool = True,
        mock_config: dict[str, Any] | None = None,
    ):
        super().__init__(codec, fallback_to_mock, mock_config)


__all__ = ["SpeechTokenizerWrapper"]
