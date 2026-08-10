from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from channels.physical_ofdm import NR_LIKE_R4, PhysicalOFDMProfile, active_grid_masks


SOURCE_SYMBOLS = 1920
COPIES = 3
_QUOTAS = (
    (90, 80, 70, 65, 55, 50, 40, 30),
    (70, 65, 62, 61, 59, 58, 55, 50),
    (50, 55, 58, 59, 61, 62, 65, 70),
    (30, 40, 50, 55, 65, 70, 80, 90),
)


@dataclass(frozen=True)
class GlobalTripletCSIReport:
    generated_tti: int
    available_tti: int
    reliability: Tensor

    @classmethod
    def from_reliability(
        cls, generated_tti: int, reliability: Tensor, delay_ttis: int = 1
    ) -> "GlobalTripletCSIReport":
        if delay_ttis < 1 or reliability.ndim != 1:
            raise ValueError("invalid global-triplet CSI report")
        return cls(
            int(generated_tti),
            int(generated_tti) + int(delay_ttis),
            reliability.detach().clone(),
        )


@dataclass(frozen=True)
class GlobalTripletAllocation:
    profile: PhysicalOFDMProfile
    selected_candidate_indices: Tensor
    resource_to_source: Tensor
    power_per_resource: Tensor
    predicted_triplet_gain: Tensor
    triplet_power_multiplier: Tensor
    branch_power_fractions: Tensor
    separation_levels: Tensor
    before_triplet_gain: Tensor
    min_selected_re_per_subcarrier: int
    max_selected_re_per_subcarrier: int
    minimum_time_separation_symbols: int = 0
    time_separation_levels: Tensor | None = None

    @property
    def unused_candidate_re(self) -> int:
        return self.profile.candidate_data_re - int(self.selected_candidate_indices.numel())

    @property
    def source_to_candidate_indices(self) -> Tensor:
        output = torch.empty_like(self.selected_candidate_indices)
        for branch in range(COPIES):
            output[branch, self.resource_to_source] = self.selected_candidate_indices[branch]
        return output

    @property
    def power_source_order(self) -> Tensor:
        output = torch.empty_like(self.power_per_resource)
        for branch in range(COPIES):
            output[branch, self.resource_to_source] = self.power_per_resource[branch]
        return output

    def place(self, source: Tensor) -> Tensor:
        if source.shape[-1] != SOURCE_SYMBOLS:
            raise ValueError("source must contain 1920 symbols")
        masks = active_grid_masks(self.profile, device=source.device)
        candidates = torch.zeros(
            *source.shape[:-1],
            self.profile.candidate_data_re,
            dtype=source.dtype,
            device=source.device,
        )
        mapped = source.index_select(-1, self.resource_to_source.to(source.device))
        for branch in range(COPIES):
            candidates[..., self.selected_candidate_indices[branch].to(source.device)] = (
                mapped * self.power_per_resource[branch].to(source.device).sqrt()
            )
        grid = torch.zeros(
            *source.shape[:-1],
            self.profile.active_subcarriers,
            self.profile.n_ofdm_symbols,
            dtype=source.dtype,
            device=source.device,
        )
        grid[..., masks.candidate_data] = candidates
        return grid

    def extract_source_order(self, active_grid: Tensor) -> Tensor:
        masks = active_grid_masks(self.profile, device=active_grid.device)
        candidates = active_grid[..., masks.candidate_data]
        output = torch.empty(
            *active_grid.shape[:-2],
            COPIES,
            SOURCE_SYMBOLS,
            dtype=active_grid.dtype,
            device=active_grid.device,
        )
        for branch in range(COPIES):
            selected = candidates.index_select(
                -1, self.selected_candidate_indices[branch].to(active_grid.device)
            )
            output[..., branch, self.resource_to_source.to(active_grid.device)] = selected
        return output


def _group_quotas(
    profile: PhysicalOFDMProfile,
    reliability: Tensor,
    coordinates: Tensor,
    bootstrap: bool,
    minimum: int,
    maximum: int,
) -> Tensor:
    if bootstrap:
        return torch.full((120,), 16, dtype=torch.long)
    frequency_score = torch.stack(
        [reliability[coordinates[:, 0] == frequency].mean() for frequency in range(360)]
    )
    group_score = (
        frequency_score[:120] + frequency_score[120:240] + frequency_score[240:]
    )
    quotas = torch.full((120,), minimum, dtype=torch.long)
    remaining = SOURCE_SYMBOLS - int(quotas.sum())
    order = torch.argsort(group_score, descending=True, stable=True)
    for item in order:
        group = int(item)
        addition = min(maximum - int(quotas[group]), remaining)
        quotas[group] += addition
        remaining -= addition
        if not remaining:
            break
    return quotas


def _select_per_subcarrier(
    profile: PhysicalOFDMProfile,
    reliability: Tensor,
    bootstrap: bool,
    minimum: int,
    maximum: int,
    minimum_time_separation_symbols: int = 0,
) -> tuple[list[Tensor], Tensor]:
    coordinates = active_grid_masks(profile).candidate_data.nonzero()
    quotas = _group_quotas(
        profile, reliability, coordinates, bootstrap, minimum, maximum
    )
    selected: list[Tensor] = []
    for frequency in range(profile.active_subcarriers):
        members = (coordinates[:, 0] == frequency).nonzero().flatten()
        times = coordinates[members, 1]
        spread_key = ((times * 11 + frequency * 7) % profile.n_ofdm_symbols).long()
        if minimum_time_separation_symbols > 0:
            # A time-interleaved policy chooses different symbol-time windows
            # for the three frequency branches of a triplet.  Reliability is
            # retained only as a deterministic tie breaker inside a window;
            # the temporal constraint is intentionally primary for this opt-in
            # mapping policy.
            branch = frequency // 120
            phase = branch * (profile.n_ofdm_symbols // COPIES)
            temporal_key = (times - phase) % profile.n_ofdm_symbols
            tie = -reliability[members].double() * 1e-12
            order = torch.argsort(temporal_key.double() + tie, stable=True)
        elif bootstrap:
            order = torch.argsort(spread_key, stable=True)
        else:
            tie = (profile.n_ofdm_symbols - spread_key).double() * 1e-12
            order = torch.argsort(reliability[members].double() + tie, descending=True, stable=True)
        chosen = members[order[: int(quotas[frequency % 120])]]
        chosen_times = coordinates[chosen, 1]
        selected.append(chosen[torch.argsort(chosen_times, stable=True)])
    return selected, quotas


def _circular_symbol_separation(first: Tensor, second: Tensor, n_symbols: int) -> Tensor:
    direct = (first - second).abs()
    return torch.minimum(direct, direct.new_full(direct.shape, n_symbols) - direct)


def _time_interleave_triplet_group(
    ordered: list[Tensor],
    coordinates: Tensor,
    n_symbols: int,
    minimum_time_separation_symbols: int,
) -> Tensor:
    """Deterministically rotate copy branches to maximize their time separation.

    The three frequency branches are intentionally left unchanged.  Only their
    pairing into a source-symbol triplet is permuted, so RE count, per-frequency
    occupancy, copy count, and the frequency-diversity guarantee are preserved.
    """
    if minimum_time_separation_symbols <= 0:
        return torch.stack(ordered)
    count = int(ordered[0].numel())
    if count == 0 or any(int(group.numel()) != count for group in ordered):
        raise AssertionError("triplet branches must have equal nonzero cardinality")
    times = [coordinates[group, 1] for group in ordered]
    best: tuple[tuple[int, int, int, int], Tensor] | None = None
    indices = torch.arange(count)
    # Pairing the first branch with cyclic rotations of the remaining branches
    # gives a deterministic, bounded search (at most 24^2 candidates in R4).
    for second_shift in range(count):
        second_index = (indices + second_shift) % count
        for third_shift in range(count):
            third_index = (indices + third_shift) % count
            candidate_times = (times[0], times[1][second_index], times[2][third_index])
            pairwise = torch.stack(
                [
                    _circular_symbol_separation(candidate_times[0], candidate_times[1], n_symbols),
                    _circular_symbol_separation(candidate_times[0], candidate_times[2], n_symbols),
                    _circular_symbol_separation(candidate_times[1], candidate_times[2], n_symbols),
                ]
            )
            minimum = int(pairwise.min())
            covered = int((pairwise.min(0).values >= minimum_time_separation_symbols).sum())
            # Prefer a fully feasible policy, then its worst pairwise spacing,
            # then aggregate spacing.  Negative shifts make the final tie break
            # choose the lexicographically earliest rotation deterministically.
            score = (covered, minimum, int(pairwise.sum()), -second_shift * count - third_shift)
            candidate = torch.stack((ordered[0], ordered[1][second_index], ordered[2][third_index]))
            if best is None or score > best[0]:
                best = (score, candidate)
    assert best is not None
    result = best[1]
    chosen_times = coordinates[result, 1]
    pairwise = torch.stack(
        [
            _circular_symbol_separation(chosen_times[0], chosen_times[1], n_symbols),
            _circular_symbol_separation(chosen_times[0], chosen_times[2], n_symbols),
            _circular_symbol_separation(chosen_times[1], chosen_times[2], n_symbols),
        ]
    ).min(0).values
    if int(pairwise.min()) < minimum_time_separation_symbols:
        raise ValueError(
            "requested minimum copy time separation is infeasible for the selected R4 resources"
        )
    return result


def _balanced_triplets(
    per_frequency: list[Tensor],
    reliability: Tensor,
    coordinates: Tensor,
    n_symbols: int,
    minimum_time_separation_symbols: int = 0,
) -> tuple[Tensor, Tensor]:
    # Three anchors separated by one third of the occupied active-index span.
    branches: list[list[Tensor]] = [[], [], []]
    before: list[Tensor] = []
    for anchor in range(120):
        groups = [per_frequency[anchor], per_frequency[anchor + 120], per_frequency[anchor + 240]]
        before.extend([reliability[groups[0]] + reliability[groups[1]] + reliability[groups[2]]])
        # Opposite reliability ordering prevents strong-strong-strong / weak-weak-weak triplets.
        orders = [
            torch.argsort(reliability[groups[0]], descending=True, stable=True),
            torch.argsort(reliability[groups[1]], stable=True),
            torch.roll(torch.argsort(reliability[groups[2]], stable=True), shifts=5),
        ]
        ordered = [groups[branch][orders[branch]] for branch in range(COPIES)]
        if minimum_time_separation_symbols > 0:
            # Start from chronological order; the rotation search below then
            # only changes which branch-copy shares each source symbol.
            ordered = [
                group[torch.argsort(coordinates[group, 1], stable=True)]
                for group in ordered
            ]
        paired = _time_interleave_triplet_group(
            ordered,
            coordinates,
            n_symbols,
            minimum_time_separation_symbols,
        )
        for branch in range(COPIES):
            branches[branch].append(paired[branch])
    return torch.stack([torch.cat(branch) for branch in branches]), torch.cat(before)


def _layer_mapping(selected: Tensor, reliability: Tensor, importance: list[int]) -> Tensor:
    score = reliability[selected].sum(0)
    order = torch.argsort(score, descending=True, stable=True)
    result = torch.empty(SOURCE_SYMBOLS, dtype=torch.long)
    offsets = [0] * 8
    for quantile, destinations in enumerate(order.chunk(4)):
        cursor = 0
        for rank, layer in enumerate(importance):
            count = _QUOTAS[quantile][rank]
            start = layer * 240 + offsets[layer]
            result[destinations[cursor : cursor + count]] = torch.arange(start, start + count)
            offsets[layer] += count
            cursor += count
    if offsets != [240] * 8 or torch.unique(result).numel() != SOURCE_SYMBOLS:
        raise AssertionError("layer-aware triplet assignment is not bijective")
    return result


def _bounded_mean_projection(raw: Tensor, lower: float, upper: float) -> Tensor:
    if not 0 < lower <= 1 <= upper:
        raise ValueError("power bounds must contain unit mean")
    lo, hi = 0.0, float(upper / raw.clamp_min(1e-20).min())
    for _ in range(80):
        scale = (lo + hi) / 2
        value = (raw * scale).clamp(lower, upper).mean()
        if value < 1:
            lo = scale
        else:
            hi = scale
    result = (raw * ((lo + hi) / 2)).clamp(lower, upper)
    if abs(float(result.mean()) - 1.0) > 2e-6:
        raise AssertionError("bounded projection did not preserve mean power")
    return result


def allocate_global_balanced_triplets(
    *,
    profile: PhysicalOFDMProfile,
    tx_tti: int,
    report: GlobalTripletCSIReport | None,
    layer_importance_order: list[int],
    min_selected_re_per_subcarrier: int = 8,
    max_selected_re_per_subcarrier: int = 24,
    minimum_frequency_separation_subcarriers: int = 60,
    minimum_time_separation_symbols: int = 0,
    q_min: float = 0.5,
    q_max: float = 2.0,
    branch_alpha: float = 0.5,
    branch_min_fraction: float = 0.15,
    epsilon: float = 1e-12,
) -> GlobalTripletAllocation:
    if profile != NR_LIKE_R4:
        raise ValueError("global balanced triplets currently require nr_like_r4")
    if sorted(layer_importance_order) != list(range(8)):
        raise ValueError("layer importance must be a permutation")
    if not min_selected_re_per_subcarrier <= 16 <= max_selected_re_per_subcarrier:
        raise ValueError("R4 balanced selection requires bounds containing 16")
    if not 0 <= minimum_time_separation_symbols < profile.n_ofdm_symbols:
        raise ValueError("minimum time separation must be in [0, n_ofdm_symbols)")
    bootstrap = tx_tti == 0
    if bootstrap:
        if report is not None:
            raise ValueError("TTI 0 must use deterministic bootstrap")
        reliability = torch.ones(profile.candidate_data_re)
    else:
        if report is None or report.available_tti != tx_tti or report.generated_tti >= tx_tti:
            raise ValueError("allocation requires causally available past CSI")
        if report.reliability.shape != (profile.candidate_data_re,):
            raise ValueError("CSI report does not match R4")
        reliability = report.reliability
    per_frequency, quotas = _select_per_subcarrier(
        profile,
        reliability,
        bootstrap,
        min_selected_re_per_subcarrier,
        max_selected_re_per_subcarrier,
        minimum_time_separation_symbols,
    )
    coordinates = active_grid_masks(profile).candidate_data.nonzero()
    selected, before_gain = _balanced_triplets(
        per_frequency,
        reliability,
        coordinates,
        profile.n_ofdm_symbols,
        minimum_time_separation_symbols,
    )
    if torch.unique(selected).numel() != COPIES * SOURCE_SYMBOLS:
        raise AssertionError("every selected resource must be unique")
    frequencies = coordinates[selected, 0]
    separations = torch.stack(
        [
            (frequencies[0] - frequencies[1]).abs(),
            (frequencies[0] - frequencies[2]).abs(),
            (frequencies[1] - frequencies[2]).abs(),
        ]
    ).min(0).values
    if int(separations.min()) < minimum_frequency_separation_subcarriers:
        raise AssertionError("frequency separation relaxation was required but not configured")
    times = coordinates[selected, 1]
    time_separations = torch.stack(
        [
            (times[0] - times[1]).abs(),
            (times[0] - times[2]).abs(),
            (times[1] - times[2]).abs(),
        ]
    ).min(0).values
    if int(time_separations.min()) < minimum_time_separation_symbols:
        raise AssertionError("minimum copy time separation was not met")
    mapping = _layer_mapping(selected, reliability, layer_importance_order)
    gain_destination = reliability[selected].clamp_min(0)
    triplet_gain_destination = gain_destination.sum(0)
    gain_source = torch.empty_like(gain_destination)
    gain_source[:, mapping] = gain_destination
    triplet_gain_source = gain_source.sum(0)
    q = _bounded_mean_projection(
        torch.rsqrt(triplet_gain_source + epsilon), q_min, q_max
    )
    branch_raw = (gain_source + epsilon).pow(branch_alpha)
    fractions = branch_min_fraction + (1 - 3 * branch_min_fraction) * (
        branch_raw / branch_raw.sum(0, keepdim=True)
    )
    powers_source = 3 * q[None] * fractions
    powers_destination = powers_source[:, mapping]
    if abs(float(powers_destination.sum()) - 5760.0) > 2e-3:
        raise AssertionError("three-copy packet energy is not 5760")
    return GlobalTripletAllocation(
        profile=profile,
        selected_candidate_indices=selected,
        resource_to_source=mapping,
        power_per_resource=powers_destination,
        predicted_triplet_gain=triplet_gain_source,
        triplet_power_multiplier=q,
        branch_power_fractions=fractions,
        separation_levels=separations,
        before_triplet_gain=before_gain,
        min_selected_re_per_subcarrier=min_selected_re_per_subcarrier,
        max_selected_re_per_subcarrier=max_selected_re_per_subcarrier,
        minimum_time_separation_symbols=minimum_time_separation_symbols,
        time_separation_levels=time_separations,
    )


__all__ = [
    "COPIES",
    "SOURCE_SYMBOLS",
    "GlobalTripletAllocation",
    "GlobalTripletCSIReport",
    "allocate_global_balanced_triplets",
]
