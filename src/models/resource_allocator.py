from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


ALLOCATION_MODES = ("uniform", "random", "reliability_greedy")


@dataclass(frozen=True)
class AllocationResult:
    symbols: Tensor
    resource_to_source: Tensor
    layer_assignment: Tensor


def _source_layers(layer_channel_uses: Sequence[int], device: torch.device) -> Tensor:
    return torch.cat(
        [torch.full((uses,), layer, device=device, dtype=torch.long)
         for layer, uses in enumerate(layer_channel_uses)]
    )


def allocate_resources(
    symbols: Tensor,
    reliability: Tensor,
    layer_channel_uses: Sequence[int],
    *,
    mode: str = "uniform",
    importance_order: Sequence[int] | None = None,
    pilot_mask: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> AllocationResult:
    """Permute analog layer partitions onto resources and retain an inverse map."""
    if mode not in ALLOCATION_MODES:
        raise ValueError(f"mode must be one of {ALLOCATION_MODES}")
    if symbols.shape != reliability.shape or symbols.ndim not in (2, 3):
        raise ValueError("symbols and reliability must match [B,M] or [B,K,N]")
    batch, total = symbols.shape[0], symbols[0].numel()
    if sum(layer_channel_uses) != total:
        raise ValueError("layer channel-use counts must sum to the resource count")
    layers = len(layer_channel_uses)
    order = list(range(layers)) if importance_order is None else list(importance_order)
    if sorted(order) != list(range(layers)):
        raise ValueError("importance_order must be a permutation of codec layer indices")

    flat_symbols = symbols.flatten(1)
    flat_reliability = reliability.flatten(1)
    flat_pilots = (
        torch.zeros_like(flat_reliability, dtype=torch.bool)
        if pilot_mask is None
        else torch.broadcast_to(pilot_mask.to(symbols.device, torch.bool), symbols.shape).flatten(1)
    )
    source_layers = _source_layers(layer_channel_uses, symbols.device)
    source_groups = [torch.where(source_layers == layer)[0] for layer in order]
    important_sources = torch.cat(source_groups)
    mapping = torch.empty((batch, total), device=symbols.device, dtype=torch.long)

    for batch_index in range(batch):
        if mode == "uniform":
            destination_order = torch.arange(total, device=symbols.device)
            source_order = destination_order
        elif mode == "random":
            destination_order = torch.randperm(total, device=symbols.device, generator=generator)
            source_order = torch.arange(total, device=symbols.device)
        else:
            ranking = flat_reliability[batch_index].clone()
            ranking[flat_pilots[batch_index]] = -torch.inf
            destination_order = ranking.argsort(descending=True)
            source_order = important_sources
        mapping[batch_index, destination_order] = source_order

    allocated = torch.gather(flat_symbols, 1, mapping)
    assignment = source_layers[mapping]
    return AllocationResult(
        allocated.reshape_as(symbols),
        mapping.reshape_as(symbols),
        assignment.reshape_as(reliability),
    )


def deallocate_resources(received: Tensor, resource_to_source: Tensor) -> Tensor:
    """Restore equalized resources to the encoder's original layer partition order."""
    if received.shape != resource_to_source.shape:
        raise ValueError("received and allocation map shapes must match")
    flat_received = received.flatten(1)
    mapping = resource_to_source.flatten(1)
    restored = torch.empty_like(flat_received)
    restored.scatter_(1, mapping, flat_received)
    return restored.reshape_as(received)


__all__ = [
    "ALLOCATION_MODES",
    "AllocationResult",
    "allocate_resources",
    "deallocate_resources",
]

