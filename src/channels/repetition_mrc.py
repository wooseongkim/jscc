from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from channels.physical_ofdm import NR_LIKE_R3, PhysicalOFDMProfile, active_grid_masks
from speech_jscc.diagnostics.physical_fdd import bounded_power_weights


SOURCE_SYMBOLS = 1920
COPIES = 3
GROUP_SUBCARRIERS = 72
_QUOTAS = (
    (90, 80, 70, 65, 55, 50, 40, 30),
    (70, 65, 62, 61, 59, 58, 55, 50),
    (50, 55, 58, 59, 61, 62, 65, 70),
    (30, 40, 50, 55, 65, 70, 80, 90),
)
_TIME_OFFSETS = (0, 9, 18)


@dataclass(frozen=True)
class RepetitionCSIReport:
    generated_tti: int
    available_tti: int
    reliability: Tensor

    @classmethod
    def from_reliability(
        cls, generated_tti: int, reliability: Tensor, delay_ttis: int = 1
    ) -> "RepetitionCSIReport":
        if delay_ttis < 1 or reliability.ndim != 1:
            raise ValueError("invalid repetition CSI report")
        return cls(
            int(generated_tti),
            int(generated_tti) + int(delay_ttis),
            reliability.detach().clone(),
        )


@dataclass(frozen=True)
class MRCResult:
    estimate: Tensor
    weights: Tensor
    denominator: Tensor
    theoretical_combined_sinr: Tensor


@dataclass(frozen=True)
class RepetitionAllocation:
    profile: PhysicalOFDMProfile
    selected_candidate_indices: Tensor
    resource_to_source: Tensor
    power_per_resource: Tensor
    energy_contract: str

    @property
    def unused_candidate_re(self) -> int:
        return self.profile.candidate_data_re - int(self.selected_candidate_indices.numel())

    @property
    def group_unused_counts(self) -> tuple[int, int, int]:
        return (24, 24, 24)

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
            *source.shape[:-1], self.profile.candidate_data_re,
            dtype=source.dtype, device=source.device,
        )
        mapped = source.index_select(-1, self.resource_to_source.to(source.device))
        for branch in range(COPIES):
            values = mapped * self.power_per_resource[branch].to(source.device).sqrt()
            candidates[..., self.selected_candidate_indices[branch].to(source.device)] = values
        grid = torch.zeros(
            *source.shape[:-1], self.profile.active_subcarriers,
            self.profile.n_ofdm_symbols, dtype=source.dtype, device=source.device,
        )
        grid[..., masks.candidate_data] = candidates
        return grid

    def extract_source_order(self, active_grid: Tensor) -> Tensor:
        masks = active_grid_masks(self.profile, device=active_grid.device)
        candidates = active_grid[..., masks.candidate_data]
        output = torch.empty(
            *active_grid.shape[:-2], COPIES, SOURCE_SYMBOLS,
            dtype=active_grid.dtype, device=active_grid.device,
        )
        for branch in range(COPIES):
            selected = candidates.index_select(
                -1, self.selected_candidate_indices[branch].to(active_grid.device)
            )
            output[..., branch, self.resource_to_source.to(active_grid.device)] = selected
        return output


def _group_candidate_indices(
    profile: PhysicalOFDMProfile, branch: int
) -> tuple[Tensor, Tensor]:
    if profile != NR_LIKE_R3:
        raise ValueError("three-copy mapping is defined only for nr_like_r3")
    coordinates = active_grid_masks(profile).candidate_data.nonzero()
    start = branch * GROUP_SUBCARRIERS
    member = (coordinates[:, 0] >= start) & (
        coordinates[:, 0] < start + GROUP_SUBCARRIERS
    )
    indices = member.nonzero().flatten()
    if indices.numel() != 1944:
        raise AssertionError("each R3 frequency group must contain 1944 candidate RE")
    return indices, coordinates[indices]


def _spread_order(coordinates: Tensor, branch: int) -> Tensor:
    local_frequency = coordinates[:, 0] - branch * GROUP_SUBCARRIERS
    shifted_time = (coordinates[:, 1] - _TIME_OFFSETS[branch]) % 28
    key = shifted_time * GROUP_SUBCARRIERS + (local_frequency * 17) % GROUP_SUBCARRIERS
    return torch.argsort(key, stable=True)


def _select_branch(
    profile: PhysicalOFDMProfile,
    branch: int,
    reliability: Tensor,
    bootstrap: bool,
) -> Tensor:
    group_indices, coordinates = _group_candidate_indices(profile, branch)
    spread = _spread_order(coordinates, branch)
    if bootstrap:
        positions = torch.floor(
            (torch.arange(SOURCE_SYMBOLS, dtype=torch.float64) + .5)
            * group_indices.numel()
            / SOURCE_SYMBOLS
        ).long()
        return group_indices[spread[positions]]
    group_reliability = reliability[group_indices]
    spread_rank = torch.empty_like(spread)
    spread_rank[spread] = torch.arange(spread.numel())
    score = group_reliability.double() + (
        spread.numel() - spread_rank
    ).double() * 1e-12
    chosen_local = torch.topk(score, SOURCE_SYMBOLS, sorted=False).indices
    chosen_local = chosen_local[torch.argsort(spread_rank[chosen_local], stable=True)]
    return group_indices[chosen_local]


def _triplet_mapping(
    selected: Tensor, reliability: Tensor, importance: list[int]
) -> Tensor:
    triplet_score = torch.log1p(reliability[selected].clamp_min(0)).sum(dim=0)
    order = torch.argsort(triplet_score, descending=True, stable=True)
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
        raise AssertionError("triplet mapping must preserve every source symbol")
    return result


def allocate_repetition3(
    *,
    profile: PhysicalOFDMProfile,
    tx_tti: int,
    report: RepetitionCSIReport | None,
    layer_importance_order: list[int],
    energy_contract: str = "fixed_power_per_copy",
    alpha: float = .5,
    minimum_relative_power: float = .5,
    maximum_relative_power: float = 2.0,
) -> RepetitionAllocation:
    if sorted(layer_importance_order) != list(range(8)):
        raise ValueError("layer importance must be a permutation")
    if energy_contract not in {"fixed_power_per_copy", "fixed_total_packet_energy"}:
        raise ValueError("unsupported repetition energy contract")
    bootstrap = tx_tti == 0
    if bootstrap:
        if report is not None:
            raise ValueError("TTI 0 must use deterministic bootstrap")
        reliability = torch.ones(profile.candidate_data_re)
    else:
        if report is None or report.available_tti != tx_tti or report.generated_tti >= tx_tti:
            raise ValueError("allocation requires causally available past CSI")
        if report.reliability.shape != (profile.candidate_data_re,):
            raise ValueError("CSI report does not match R3")
        reliability = report.reliability
    selected = torch.stack(
        [_select_branch(profile, branch, reliability, bootstrap) for branch in range(COPIES)]
    )
    if torch.unique(selected).numel() != COPIES * SOURCE_SYMBOLS:
        raise AssertionError("copy resources must be disjoint")
    mapping = _triplet_mapping(selected, reliability, layer_importance_order)
    powers = torch.stack([
        bounded_power_weights(
            reliability[selected[branch]], alpha=alpha,
            minimum_relative_power=minimum_relative_power,
            maximum_relative_power=maximum_relative_power,
        )
        for branch in range(COPIES)
    ])
    if energy_contract == "fixed_total_packet_energy":
        powers = powers / COPIES
    return RepetitionAllocation(profile, selected, mapping, powers, energy_contract)


def _noise_tensor(noise_variance: float | Tensor, reference: Tensor) -> Tensor:
    variance = torch.as_tensor(
        noise_variance, dtype=reference.real.dtype, device=reference.device
    )
    if variance.ndim == 0:
        return variance
    if variance.ndim == 1 and variance.numel() == reference.shape[1]:
        return variance[None, :, None]
    return torch.broadcast_to(variance, reference.shape)


def coherent_mrc(
    raw_observations: Tensor,
    channel_estimate: Tensor,
    power_source_order: Tensor,
    noise_variance: float | Tensor,
    *,
    source_power: float = 1.0,
    epsilon: float = 1e-12,
) -> MRCResult:
    if raw_observations.shape != channel_estimate.shape or raw_observations.ndim != 3:
        raise ValueError("MRC inputs must match [B,copies,1920]")
    if raw_observations.shape[1] < 1:
        raise ValueError("MRC requires at least one copy")
    if power_source_order.shape != raw_observations.shape[1:]:
        raise ValueError("power tensor must match [3,symbols]")
    amplitude = power_source_order.to(raw_observations.device).sqrt()
    effective_channel = channel_estimate * amplitude[None]
    variance = _noise_tensor(noise_variance, raw_observations)
    reliability = effective_channel.abs().square() / variance
    denominator = reliability.sum(dim=1)
    numerator = (
        effective_channel.conj() * raw_observations / variance
    ).sum(dim=1)
    estimate = numerator / (denominator + epsilon)
    weights = reliability / (denominator[:, None] + epsilon)
    if not (
        torch.isfinite(estimate).all()
        and torch.isfinite(weights).all()
        and torch.isfinite(denominator).all()
    ):
        raise FloatingPointError("nonfinite coherent MRC result")
    return MRCResult(
        estimate=estimate,
        weights=weights,
        denominator=denominator,
        theoretical_combined_sinr=denominator * float(source_power),
    )


def oracle_branch_sinr(
    true_channel: Tensor,
    power_source_order: Tensor,
    noise_variance: float | Tensor,
    *,
    source_power: float,
) -> Tensor:
    variance = _noise_tensor(noise_variance, true_channel)
    return (
        true_channel.abs().square()
        * power_source_order.to(true_channel.device)[None]
        * float(source_power)
        / variance
    )


__all__ = [
    "MRCResult",
    "RepetitionAllocation",
    "RepetitionCSIReport",
    "allocate_repetition3",
    "coherent_mrc",
    "oracle_branch_sinr",
]
