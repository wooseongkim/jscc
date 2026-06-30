from __future__ import annotations

from typing import Any

from ._delegating_wrapper import DelegatingCodecWrapper
from .base_codec import BaseCodec


class EnCodecWrapper(DelegatingCodecWrapper):
    """Placeholder for an EnCodec adapter with deterministic mock fallback.

    Pass a project-specific `BaseCodec` adapter in `codec` when pretrained
    EnCodec dependencies and weights are available.
    """

    external_name = "EnCodec"

    def __init__(
        self,
        codec: BaseCodec | None = None,
        *,
        fallback_to_mock: bool = True,
        mock_config: dict[str, Any] | None = None,
    ):
        super().__init__(codec, fallback_to_mock, mock_config)


__all__ = ["EnCodecWrapper"]
