"""Fixed-profile layer-aware variable-repetition allocation for R4.

This is deliberately an *open-loop* oracle attack-class experiment helper.  It
does not inspect the jammer or adapt per packet.  U0 delegates to the existing
global balanced-triplet allocator; non-uniform profiles only add/remove copies
in source-layer order while preserving the 5,760-data-RE packet budget.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from channels.global_triplet_allocator import (
    GlobalTripletCSIReport,
    SOURCE_SYMBOLS,
    allocate_global_balanced_triplets,
)
from channels.physical_ofdm import NR_LIKE_R4, PhysicalOFDMProfile, active_grid_masks


LAYERS = 8
SYMBOLS_PER_LAYER = 240


@dataclass(frozen=True)
class UEPProfile:
    name: str
    repetition: tuple[int, ...]
    power_raw: tuple[float, ...] | None = None
    # Optimizer candidates express a layer's *total* packet energy share.
    # Fixed historical profiles keep ``power_raw`` for backwards compatibility.
    power_share: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.repetition) != LAYERS:
            raise ValueError("UEP profiles require eight layer values")
        if any(value not in (1, 2, 3, 4, 5) for value in self.repetition):
            raise ValueError("R4 UEP repetition must be between 1 and 5")
        if sum(self.repetition) != 24:
            raise ValueError("UEP repetition must sum to 24 to preserve 5,760 RE")
        if self.power_raw is None and self.power_share is None:
            raise ValueError("UEP profile requires raw power or total power share")
        if self.power_raw is not None and len(self.power_raw) != LAYERS:
            raise ValueError("UEP raw power requires eight values")
        if self.power_raw is not None and any(value <= 0 for value in self.power_raw):
            raise ValueError("UEP raw powers must be positive")
        if self.power_share is not None:
            if len(self.power_share) != LAYERS or any(value <= 0 for value in self.power_share):
                raise ValueError("UEP total power shares must be positive eight-vector")
            if abs(sum(self.power_share) - 1.0) > 1e-8:
                raise ValueError("UEP total power shares must sum to one")

    @property
    def is_uniform(self) -> bool:
        if self.repetition != (3,) * LAYERS:
            return False
        if self.power_share is not None:
            return all(abs(value - 1.0 / LAYERS) < 1e-8 for value in self.power_share)
        return self.power_raw == (1.0,) * LAYERS

    def normalized_power(self) -> Tensor:
        """Per-RE layer power multiplier for historical/profile logging."""
        if self.power_share is not None:
            return self.per_re_layer_power()
        assert self.power_raw is not None
        raw = torch.tensor(self.power_raw, dtype=torch.float32)
        copies = torch.tensor(self.repetition, dtype=torch.float32)
        # Every layer contains exactly 240 complex source symbols.
        return raw / ((copies * raw).sum() / 24.0)

    def total_power_share(self) -> Tensor:
        if self.power_share is not None:
            return torch.tensor(self.power_share, dtype=torch.float32)
        copies = torch.tensor(self.repetition, dtype=torch.float32)
        per_re = self.normalized_power()
        return copies * per_re / 24.0

    def per_re_layer_power(self) -> Tensor:
        shares = self.total_power_share()
        copies = torch.tensor(self.repetition, dtype=torch.float32)
        return 24.0 * shares / copies


UEP_PROFILES = {
    "U0": UEPProfile("U0", (3, 3, 3, 3, 3, 3, 3, 3), (1, 1, 1, 1, 1, 1, 1, 1)),
    "R1": UEPProfile("R1", (4, 3, 3, 3, 3, 3, 3, 2), (1, 1, 1, 1, 1, 1, 1, 1)),
    "R2": UEPProfile("R2", (4, 4, 3, 3, 3, 3, 2, 2), (1, 1, 1, 1, 1, 1, 1, 1)),
    "P1": UEPProfile("P1", (3, 3, 3, 3, 3, 3, 3, 3), (1.25, 1.25, 1, 1, 1, 1, .75, .75)),
    "P2": UEPProfile("P2", (3, 3, 3, 3, 3, 3, 3, 3), (1.4, 1.4, 1.1, 1, .9, .9, .7, .7)),
    "RP1": UEPProfile("RP1", (4, 3, 3, 3, 3, 3, 3, 2), (1.25, 1.25, 1, 1, 1, 1, .75, .75)),
    "RP2": UEPProfile("RP2", (4, 4, 3, 3, 3, 3, 2, 2), (1.4, 1.4, 1.1, 1, .9, .9, .7, .7)),
}


@dataclass(frozen=True)
class VariableCopyAllocation:
    profile: PhysicalOFDMProfile
    selected_candidate_indices: Tensor  # [max_copies, 1920], -1 marks absent copy
    power_per_source_copy: Tensor       # [max_copies, 1920], zero for absent copy
    uep_profile: UEPProfile
    normalized_layer_power: Tensor

    @property
    def copy_count_per_source(self) -> Tensor:
        return self.selected_candidate_indices.ge(0).sum(0)

    @property
    def selected_re_count(self) -> int:
        return int(self.selected_candidate_indices.ge(0).sum())

    @property
    def unused_candidate_re(self) -> int:
        return self.profile.candidate_data_re - self.selected_re_count

    @property
    def source_to_candidate_indices(self) -> Tensor:
        return self.selected_candidate_indices

    @property
    def power_source_order(self) -> Tensor:
        return self.power_per_source_copy

    @property
    def resource_to_source(self) -> Tensor:
        return torch.arange(SOURCE_SYMBOLS, dtype=torch.long)[None].expand(
            self.selected_candidate_indices.shape[0], -1
        )

    def place(self, source: Tensor) -> Tensor:
        if source.shape[-1] != SOURCE_SYMBOLS:
            raise ValueError("source must contain 1920 symbols")
        masks = active_grid_masks(self.profile, device=source.device)
        candidates = torch.zeros(*source.shape[:-1], self.profile.candidate_data_re,
                                 dtype=source.dtype, device=source.device)
        for copy in range(self.selected_candidate_indices.shape[0]):
            valid = self.selected_candidate_indices[copy].ge(0)
            candidate = self.selected_candidate_indices[copy, valid].to(source.device)
            source_index = valid.nonzero().flatten().to(source.device)
            candidates[..., candidate] = source.index_select(-1, source_index) * self.power_per_source_copy[copy, valid].to(source.device).sqrt()
        grid = torch.zeros(*source.shape[:-1], self.profile.active_subcarriers,
                           self.profile.n_ofdm_symbols, dtype=source.dtype, device=source.device)
        grid[..., masks.candidate_data] = candidates
        return grid

    def extract_source_order(self, active_grid: Tensor) -> Tensor:
        masks = active_grid_masks(self.profile, device=active_grid.device)
        candidates = active_grid[..., masks.candidate_data]
        output = torch.zeros(*active_grid.shape[:-2], self.selected_candidate_indices.shape[0], SOURCE_SYMBOLS,
                             dtype=active_grid.dtype, device=active_grid.device)
        for copy in range(self.selected_candidate_indices.shape[0]):
            valid = self.selected_candidate_indices[copy].ge(0)
            output[..., copy, valid.to(active_grid.device)] = candidates.index_select(
                -1, self.selected_candidate_indices[copy, valid].to(active_grid.device)
            )
        return output


def _evenly_spaced(items: Tensor, count: int) -> Tensor:
    if count > int(items.numel()):
        raise ValueError("insufficient free candidate RE for additional copies")
    positions = torch.floor((torch.arange(count, dtype=torch.float64) + .5) * items.numel() / count).long()
    return items[positions]


def allocate_r4_uep(
    *, profile: PhysicalOFDMProfile, tx_tti: int, report: GlobalTripletCSIReport | None,
    layer_importance_order: list[int], uep_profile: UEPProfile,
):
    """Return the historical allocator for U0, otherwise a fixed variable-copy map."""
    base = allocate_global_balanced_triplets(
        profile=profile, tx_tti=tx_tti, report=report,
        layer_importance_order=layer_importance_order,
    )
    if uep_profile.is_uniform:
        return base
    if profile != NR_LIKE_R4:
        raise ValueError("variable UEP is defined for nr_like_r4 only")
    base_indices = base.source_to_candidate_indices
    base_power = base.power_source_order
    max_copies = max(uep_profile.repetition)
    selected = torch.full((max_copies, SOURCE_SYMBOLS), -1, dtype=torch.long)
    power = torch.zeros((max_copies, SOURCE_SYMBOLS), dtype=torch.float32)
    selected[:3] = base_indices
    layer = torch.arange(SOURCE_SYMBOLS) // SYMBOLS_PER_LAYER
    desired = torch.tensor(uep_profile.repetition, dtype=torch.long)[layer]
    # Retain only the desired subset of the three historical copies.  This
    # supports both r=1 and r=2 without changing U0's dedicated path.
    for copy in range(3):
        removed = desired.le(copy)
        selected[copy, removed] = -1
    # Additional copies (r=4/5) occupy deterministic evenly spread free REs.
    extra_pairs: list[tuple[int, int]] = []
    for copy in range(3, max_copies):
        for source_index in desired.gt(copy).nonzero().flatten().tolist():
            extra_pairs.append((copy, source_index))
    used = selected[selected.ge(0)]
    free = torch.ones(profile.candidate_data_re, dtype=torch.bool)
    free[used] = False
    coordinates = active_grid_masks(profile).candidate_data.nonzero()
    candidates = free.nonzero().flatten()
    spread = ((coordinates[candidates, 1] * 11 + coordinates[candidates, 0] * 7) % profile.n_ofdm_symbols)
    candidates = candidates[torch.argsort(spread, stable=True)]
    if extra_pairs:
        selected_candidates = _evenly_spaced(candidates, len(extra_pairs))
        for candidate_index, (copy, source_index) in zip(selected_candidates.tolist(), extra_pairs, strict=True):
            selected[copy, source_index] = candidate_index
    # Optimizer candidates use the exact total-share definition a_i=24p_i/r_i.
    # Historical named profiles retain their normalized per-RE multipliers.
    normalized = uep_profile.per_re_layer_power() if uep_profile.power_share is not None else uep_profile.normalized_power()
    for copy in range(max_copies):
        valid = selected[copy].ge(0)
        power[copy, valid] = normalized[layer[valid]]
    if int(selected.ge(0).sum()) != 5760 or not torch.equal(selected.ge(0).sum(0), desired):
        raise AssertionError("variable repetition profile violates copy/RE budget")
    if abs(float(power.sum()) - 5760.0) > 2e-3:
        raise AssertionError("UEP packet energy is not preserved")
    return VariableCopyAllocation(profile, selected, power, uep_profile, normalized)


__all__ = ["UEPProfile", "UEP_PROFILES", "VariableCopyAllocation", "allocate_r4_uep"]
