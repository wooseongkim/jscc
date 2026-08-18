"""SINR-utility RE placement for fixed R4 UEP profiles.

This is intentionally separate from the historical global balanced-triplet
allocator.  It preserves a supplied UEPProfile's repetition and power values,
and optimizes only which candidate data RE carries each source-symbol copy.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import torch
from torch import Tensor

from channels.global_triplet_allocator import GlobalTripletCSIReport, SOURCE_SYMBOLS
from channels.physical_ofdm import NR_LIKE_R4, PhysicalOFDMProfile, active_grid_masks
from channels.r4_uep_allocator import LAYERS, SYMBOLS_PER_LAYER, UEPProfile
from channels.re_risk import REInterferenceReport, require_available_interference_report


def allocation_sinr(
    channel_gain: Tensor,
    interference_power: Tensor,
    *,
    noise_power: float | Tensor,
    per_re_power: float | Tensor,
    eps: float = 1e-12,
) -> Tensor:
    """Return the allocation-time SINR proxy a|Hhat|^2/(Nhat+Ihat)."""
    gain = torch.as_tensor(channel_gain)
    interference = torch.as_tensor(interference_power, dtype=gain.dtype, device=gain.device)
    if gain.shape != interference.shape:
        raise ValueError("channel gain and interference power must have equal shape")
    if (gain < 0).any() or (interference < 0).any():
        raise ValueError("gain and interference power must be nonnegative")
    return torch.as_tensor(per_re_power, dtype=gain.dtype, device=gain.device) * gain / (
        torch.as_tensor(noise_power, dtype=gain.dtype, device=gain.device) + interference + eps
    )


def importance_weights(layer_importance_order: list[int]) -> Tensor:
    """Map the fixed importance ordering to deterministic descending weights."""
    if sorted(layer_importance_order) != list(range(LAYERS)):
        raise ValueError("layer importance must be a permutation of 0..7")
    weights = torch.empty(LAYERS, dtype=torch.float64)
    for rank, layer in enumerate(layer_importance_order):
        weights[layer] = float(LAYERS - rank)
    return weights


@dataclass(frozen=True)
class JammerAwareVariableCopyAllocation:
    profile: PhysicalOFDMProfile
    selected_candidate_indices: Tensor  # [max_copies, 1920], -1 means absent
    power_per_source_copy: Tensor       # [max_copies, 1920]
    uep_profile: UEPProfile
    normalized_layer_power: Tensor
    assignment_sinr: Tensor             # [max_copies, 1920], zero for absent
    source_subband_ids: Tensor          # [max_copies, 1920], -1 for absent
    layer_weights: Tensor
    subband_count: int
    allocation_mode: str = "jammer_aware_sinr"

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

    @property
    def distinct_subband_per_source(self) -> bool:
        for source in range(SOURCE_SYMBOLS):
            bands = self.source_subband_ids[:, source]
            bands = bands[bands.ge(0)]
            if bands.numel() != torch.unique(bands).numel():
                return False
        return True

    def place(self, source: Tensor) -> Tensor:
        if source.shape[-1] != SOURCE_SYMBOLS:
            raise ValueError("source must contain 1920 symbols")
        masks = active_grid_masks(self.profile, device=source.device)
        candidates = torch.zeros(
            *source.shape[:-1], self.profile.candidate_data_re,
            dtype=source.dtype, device=source.device,
        )
        for copy in range(self.selected_candidate_indices.shape[0]):
            valid = self.selected_candidate_indices[copy].ge(0)
            candidates[..., self.selected_candidate_indices[copy, valid].to(source.device)] = (
                source.index_select(-1, valid.nonzero().flatten().to(source.device))
                * self.power_per_source_copy[copy, valid].to(source.device).sqrt()
            )
        grid = torch.zeros(
            *source.shape[:-1], self.profile.active_subcarriers,
            self.profile.n_ofdm_symbols, dtype=source.dtype, device=source.device,
        )
        grid[..., masks.candidate_data] = candidates
        return grid

    def extract_source_order(self, active_grid: Tensor) -> Tensor:
        masks = active_grid_masks(self.profile, device=active_grid.device)
        candidates = active_grid[..., masks.candidate_data]
        output = torch.zeros(
            *active_grid.shape[:-2], self.selected_candidate_indices.shape[0], SOURCE_SYMBOLS,
            dtype=active_grid.dtype, device=active_grid.device,
        )
        for copy in range(self.selected_candidate_indices.shape[0]):
            valid = self.selected_candidate_indices[copy].ge(0)
            output[..., copy, valid.to(active_grid.device)] = candidates.index_select(
                -1, self.selected_candidate_indices[copy, valid].to(active_grid.device)
            )
        return output


def _reports_for_tti(
    tx_tti: int,
    csi_report: GlobalTripletCSIReport | None,
    interference_report: REInterferenceReport | None,
    candidate_count: int,
    *,
    use_interference: bool,
) -> tuple[Tensor, Tensor, float]:
    if tx_tti == 0:
        if csi_report is not None or (use_interference and interference_report is not None):
            raise ValueError("TTI 0 jammer-aware allocation requires no delayed reports")
        return torch.ones(candidate_count), torch.zeros(candidate_count), 1.0
    if csi_report is None or csi_report.available_tti != tx_tti or csi_report.generated_tti >= tx_tti:
        raise ValueError("CSI report is not causally available")
    if csi_report.reliability.shape != (candidate_count,):
        raise ValueError("delayed report does not match R4 candidate RE count")
    if not use_interference:
        # This is the CSI-only ablation: only delayed LS channel reliability
        # participates in the score.  Unit noise is a fixed scale factor and
        # no residual/interference/jammer observation is consumed here.
        return csi_report.reliability.double(), torch.zeros_like(csi_report.reliability, dtype=torch.float64), 1.0
    interference = require_available_interference_report(tx_tti=tx_tti, report=interference_report)
    if interference.interference_power.shape != (candidate_count,):
        raise ValueError("delayed report does not match R4 candidate RE count")
    return csi_report.reliability.double(), interference.interference_power.double(), interference.noise_power


def _allocate_r4_fixed_uep_utility(
    *,
    profile: PhysicalOFDMProfile,
    tx_tti: int,
    csi_report: GlobalTripletCSIReport | None,
    interference_report: REInterferenceReport | None,
    layer_importance_order: list[int],
    uep_profile: UEPProfile,
    use_interference: bool,
    allocation_mode: str,
    eps: float = 1e-12,
) -> JammerAwareVariableCopyAllocation:
    """Allocate fixed UEP copies by exact subband diversity and MRC utility.

    Each allocation step chooses the feasible (source symbol, candidate RE)
    pair with maximum importance-weighted log-utility increment.  Candidate
    REs are never reused and a source symbol receives at most one copy in a
    frequency subband.  No copy index is anchored to a particular subband.
    """
    if profile != NR_LIKE_R4:
        raise ValueError("jammer-aware allocator currently requires nr_like_r4")
    gains, interference, noise_power = _reports_for_tti(
        tx_tti, csi_report, interference_report, profile.candidate_data_re,
        use_interference=use_interference,
    )
    if not torch.isfinite(gains).all() or not torch.isfinite(interference).all():
        raise ValueError("allocation reports must be finite")
    repetition = torch.tensor(uep_profile.repetition, dtype=torch.long)
    max_copies = int(repetition.max())
    subband_count = max_copies
    per_re_power = uep_profile.per_re_layer_power().double()
    weights = importance_weights(layer_importance_order)
    gamma = torch.stack(
        [allocation_sinr(gains, interference, noise_power=noise_power, per_re_power=per_re_power[layer], eps=eps)
         for layer in range(LAYERS)]
    ).double()
    if not torch.isfinite(gamma).all():
        raise FloatingPointError("nonfinite allocation SINR")

    coordinates = active_grid_masks(profile).candidate_data.nonzero()
    bands = (coordinates[:, 0].long() * subband_count // profile.active_subcarriers).long()
    # Per layer/subband ordered lists allow a lazy heap to recover the best
    # still-free RE without a fixed triplet/anchor structure.
    gamma_values = gamma.tolist()
    weight_values = weights.tolist()
    ordered: list[list[list[int]]] = []
    for layer in range(LAYERS):
        entries: list[list[int]] = []
        for band in range(subband_count):
            members = (bands == band).nonzero().flatten()
            entries.append(
                members[torch.argsort(gamma[layer, members], descending=True, stable=True)].tolist()
            )
        ordered.append(entries)
    cursor = [[0 for _ in range(subband_count)] for _ in range(LAYERS)]
    used = [False] * profile.candidate_data_re

    layer_of_source = [source // SYMBOLS_PER_LAYER for source in range(SOURCE_SYMBOLS)]
    desired = [int(repetition[layer]) for layer in layer_of_source]
    selected = torch.full((max_copies, SOURCE_SYMBOLS), -1, dtype=torch.long)
    powers = torch.zeros((max_copies, SOURCE_SYMBOLS), dtype=torch.float32)
    assignment_sinr = torch.zeros((max_copies, SOURCE_SYMBOLS), dtype=torch.float64)
    source_bands = torch.full((max_copies, SOURCE_SYMBOLS), -1, dtype=torch.long)
    combined = [0.0] * SOURCE_SYMBOLS
    assigned = [0] * SOURCE_SYMBOLS
    used_bands = [[False] * subband_count for _ in range(SOURCE_SYMBOLS)]

    def best_free(layer: int, band: int) -> int | None:
        values = ordered[layer][band]
        pointer = cursor[layer][band]
        while pointer < len(values) and used[values[pointer]]:
            pointer += 1
        cursor[layer][band] = pointer
        return None if pointer == len(values) else values[pointer]

    def push(source_index: int, band: int, heap: list[tuple[float, int, int, int, float]]) -> None:
        if assigned[source_index] >= desired[source_index] or used_bands[source_index][band]:
            return
        layer = layer_of_source[source_index]
        candidate = best_free(layer, band)
        if candidate is None:
            return
        current = combined[source_index]
        marginal = weight_values[layer] * math.log1p(
            gamma_values[layer][candidate] / (1.0 + current)
        )
        heapq.heappush(heap, (-marginal, source_index, band, candidate, current))

    heap: list[tuple[float, int, int, int, float]] = []
    for source in range(SOURCE_SYMBOLS):
        for band in range(subband_count):
            push(source, band, heap)

    total = sum(desired)
    placed = 0
    while heap:
        neg, source, band, candidate, prior_combined = heapq.heappop(heap)
        if assigned[source] >= desired[source] or used_bands[source][band]:
            continue
        layer = layer_of_source[source]
        current_candidate = best_free(layer, band)
        if current_candidate is None:
            continue
        current = combined[source]
        current_marginal = weight_values[layer] * math.log1p(
            gamma_values[layer][current_candidate] / (1.0 + current)
        )
        if current_candidate != candidate or abs(current - prior_combined) > 0.0 or abs(-neg - current_marginal) > 1e-14:
            heapq.heappush(heap, (-current_marginal, source, band, current_candidate, current))
            continue
        slot = assigned[source]
        selected[slot, source] = candidate
        powers[slot, source] = per_re_power[layer].float()
        assignment_sinr[slot, source] = gamma_values[layer][candidate]
        source_bands[slot, source] = band
        used[candidate] = True
        used_bands[source][band] = True
        combined[source] += gamma_values[layer][candidate]
        assigned[source] += 1
        placed += 1
        # Entries for this source's remaining bands stay in the heap with an
        # optimistic key based on its preceding MRC sum.  When one reaches the
        # top, the stale-entry branch above recomputes and re-inserts exactly
        # that *popped* entry.  Re-pushing every remaining band here created
        # duplicate live proposals and let a long evaluation grow the heap.
    if placed != total or assigned != desired:
        raise AssertionError("insufficient feasible REs for exact UEP copy counts")
    if int(torch.unique(selected[selected.ge(0)]).numel()) != total:
        raise AssertionError("jammer-aware allocator reused a data RE")
    if abs(float(powers.sum()) - float(total)) > 2e-3:
        raise AssertionError("jammer-aware allocation did not preserve packet energy")
    allocation = JammerAwareVariableCopyAllocation(
        profile=profile,
        selected_candidate_indices=selected,
        power_per_source_copy=powers,
        uep_profile=uep_profile,
        normalized_layer_power=per_re_power.float(),
        assignment_sinr=assignment_sinr.float(),
        source_subband_ids=source_bands,
        layer_weights=weights.float(),
        subband_count=subband_count,
        allocation_mode=allocation_mode,
    )
    if not allocation.distinct_subband_per_source:
        raise AssertionError("source copies must occupy distinct subbands")
    return allocation


def allocate_r4_jammer_aware_sinr(
    *,
    profile: PhysicalOFDMProfile,
    tx_tti: int,
    csi_report: GlobalTripletCSIReport | None,
    interference_report: REInterferenceReport | None,
    layer_importance_order: list[int],
    uep_profile: UEPProfile,
    eps: float = 1e-12,
) -> JammerAwareVariableCopyAllocation:
    """Allocate using delayed CSI and delayed RX-estimated interference."""
    return _allocate_r4_fixed_uep_utility(
        profile=profile, tx_tti=tx_tti, csi_report=csi_report,
        interference_report=interference_report,
        layer_importance_order=layer_importance_order, uep_profile=uep_profile,
        use_interference=True, allocation_mode="delayed_rx_interference", eps=eps,
    )


def allocate_r4_csi_only(
    *,
    profile: PhysicalOFDMProfile,
    tx_tti: int,
    csi_report: GlobalTripletCSIReport | None,
    layer_importance_order: list[int],
    uep_profile: UEPProfile,
    eps: float = 1e-12,
) -> JammerAwareVariableCopyAllocation:
    """Allocate from delayed CSI reliability alone; no risk map is consumed."""
    return _allocate_r4_fixed_uep_utility(
        profile=profile, tx_tti=tx_tti, csi_report=csi_report,
        interference_report=None,
        layer_importance_order=layer_importance_order, uep_profile=uep_profile,
        use_interference=False, allocation_mode="csi_only", eps=eps,
    )


__all__ = [
    "JammerAwareVariableCopyAllocation",
    "allocate_r4_csi_only",
    "allocate_r4_jammer_aware_sinr",
    "allocation_sinr",
    "importance_weights",
]
