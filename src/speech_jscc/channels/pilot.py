"""Compatibility imports for :mod:`channels.pilot`."""

from channels.pilot import (
    csi_nmse,
    equalize_with_csi,
    estimate_channel_ls,
    estimate_flat_ls,
    estimate_ofdm_ls,
    insert_pilots,
    interpolate_pilot_estimates,
    make_pilot_mask,
    pilot_evm,
    remove_pilot_resources,
)

__all__ = [
    "csi_nmse",
    "equalize_with_csi",
    "estimate_channel_ls",
    "estimate_flat_ls",
    "estimate_ofdm_ls",
    "insert_pilots",
    "interpolate_pilot_estimates",
    "make_pilot_mask",
    "pilot_evm",
    "remove_pilot_resources",
]
