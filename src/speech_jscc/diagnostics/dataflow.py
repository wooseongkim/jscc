from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor


def audit_resource_mapping(
    symbols: Tensor,
    pilot_mask: Tensor,
    layer_channel_uses: Sequence[int],
) -> dict[str, Any]:
    """Count usable resources and encoder symbols destroyed by pilot replacement."""
    if not torch.is_complex(symbols) or symbols.ndim not in (2, 3):
        raise ValueError("symbols must be complex [B,M] or [B,K,N]")
    mask = torch.broadcast_to(pilot_mask.to(symbols.device, torch.bool), symbols.shape)
    grid = symbols[0].numel()
    encoder_symbols = int(sum(layer_channel_uses))
    if encoder_symbols != grid:
        raise ValueError("layer channel-use counts must equal the emitted symbol count")
    pilot_counts = mask.flatten(1).sum(dim=1)
    if not torch.all(pilot_counts == pilot_counts[0]):
        raise ValueError("diagnostic requires a consistent pilot count per sample")
    pilots = int(pilot_counts[0])
    nonpilots = grid - pilots
    overwritten = min(encoder_symbols, grid) - min(encoder_symbols, nonpilots)
    return {
        "batch_size": symbols.shape[0],
        "grid_resources_per_sample": grid,
        "pilot_resources_per_sample": pilots,
        "nonpilot_data_resources_per_sample": nonpilots,
        "encoder_symbols_per_sample": encoder_symbols,
        "overwritten_encoder_symbols_per_sample": overwritten,
        "pilot_fraction": pilots / grid,
        "resource_mapping_defect": overwritten > 0,
        "finite": bool(torch.isfinite(symbols).all()),
    }
