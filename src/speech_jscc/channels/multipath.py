"""Compatibility imports for :mod:`channels.multipath`."""

from channels.multipath import (
    exponential_pdp,
    multipath_block_fading,
    sample_tdl_taps,
    taps_to_ofdm_response,
)

__all__ = [
    "exponential_pdp",
    "multipath_block_fading",
    "sample_tdl_taps",
    "taps_to_ofdm_response",
]
