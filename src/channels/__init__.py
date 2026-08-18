"""Complex wireless channel and jamming simulation primitives."""

from channels.jammer import (
    compute_jsr,
    jammer_mask_statistics,
    make_jammer,
    make_jammer_mask,
)
from channels.rayleigh import RayleighChannel, compute_effective_sinr, rayleigh_channel
from channels.pilot import (
    csi_nmse,
    equalize_with_csi,
    estimate_channel_ls,
    estimate_flat_ls,
    estimate_ofdm_ls,
    insert_pilots,
    make_pilot_mask,
    pilot_evm,
    remove_pilot_resources,
)
from channels.reliability import compute_resource_reliability, estimate_unreliable_mask
from channels.re_risk import (
    REInterferenceReport,
    RERiskReport,
    allocate_risk_aware_global_balanced_triplets,
    combine_csi_and_risk_report,
    combine_reliability_with_risk,
    estimate_rx_residual_risk_map,
    estimate_rx_residual_interference_power,
    oracle_jammer_grid_to_interference_report,
    oracle_jamming_mask_to_risk_report,
    require_available_interference_report,
)
from channels.r4_jammer_aware_allocator import allocate_r4_jammer_aware_sinr

__all__ = [
    "RayleighChannel",
    "RERiskReport",
    "REInterferenceReport",
    "allocate_r4_jammer_aware_sinr",
    "allocate_risk_aware_global_balanced_triplets",
    "combine_csi_and_risk_report",
    "combine_reliability_with_risk",
    "csi_nmse",
    "equalize_with_csi",
    "estimate_channel_ls",
    "estimate_rx_residual_risk_map",
    "estimate_rx_residual_interference_power",
    "estimate_flat_ls",
    "estimate_ofdm_ls",
    "compute_effective_sinr",
    "compute_jsr",
    "compute_resource_reliability",
    "jammer_mask_statistics",
    "insert_pilots",
    "make_jammer",
    "make_jammer_mask",
    "make_pilot_mask",
    "pilot_evm",
    "oracle_jamming_mask_to_risk_report",
    "oracle_jammer_grid_to_interference_report",
    "require_available_interference_report",
    "rayleigh_channel",
    "remove_pilot_resources",
    "estimate_unreliable_mask",
]
