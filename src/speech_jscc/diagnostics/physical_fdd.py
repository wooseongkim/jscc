from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from channels.physical_ofdm import PhysicalOFDMProfile, active_grid_masks


SOURCE_SYMBOLS = 1920
SYMBOLS_PER_LAYER = 240
LAYER_ORDER_QUOTAS = (
    (90, 80, 70, 65, 55, 50, 40, 30),
    (70, 65, 62, 61, 59, 58, 55, 50),
    (50, 55, 58, 59, 61, 62, 65, 70),
    (30, 40, 50, 55, 65, 70, 80, 90),
)


@dataclass(frozen=True)
class PhysicalCSIReport:
    generated_tti: int
    available_tti: int
    reliability: Tensor

    @classmethod
    def from_reliability(
        cls, generated_tti: int, reliability: Tensor, delay_ttis: int = 1
    ) -> "PhysicalCSIReport":
        if delay_ttis < 1 or reliability.ndim != 1:
            raise ValueError("invalid causal CSI report")
        return cls(
            int(generated_tti),
            int(generated_tti) + int(delay_ttis),
            reliability.detach().clone(),
        )


@dataclass(frozen=True)
class PhysicalAllocation:
    profile: PhysicalOFDMProfile
    selected_candidate_indices: Tensor
    resource_to_source: Tensor
    relative_power: Tensor

    def place(self, source: Tensor) -> Tensor:
        if source.shape[-1] != SOURCE_SYMBOLS:
            raise ValueError("source must contain exactly 1920 symbols")
        masks = active_grid_masks(self.profile, device=source.device)
        candidates = torch.zeros(
            *source.shape[:-1],
            self.profile.candidate_data_re,
            dtype=source.dtype,
            device=source.device,
        )
        mapped = source.index_select(-1, self.resource_to_source.to(source.device))
        powered = mapped * self.relative_power.to(source.device).sqrt()
        candidates[..., self.selected_candidate_indices.to(source.device)] = powered
        grid = torch.zeros(
            *source.shape[:-1],
            self.profile.active_subcarriers,
            self.profile.n_ofdm_symbols,
            dtype=source.dtype,
            device=source.device,
        )
        grid[..., masks.candidate_data] = candidates
        return grid


def _spread_order(profile: PhysicalOFDMProfile) -> Tensor:
    masks = active_grid_masks(profile)
    coordinates = masks.candidate_data.nonzero()
    key = (
        ((coordinates[:, 1] * 13) % profile.n_ofdm_symbols) * profile.active_subcarriers
        + ((coordinates[:, 0] * 17) % profile.active_subcarriers)
    )
    return torch.argsort(key, stable=True)


def _uniform_selection(profile: PhysicalOFDMProfile) -> Tensor:
    order = _spread_order(profile)
    positions = torch.floor(
        (torch.arange(SOURCE_SYMBOLS, dtype=torch.float64) + 0.5)
        * profile.candidate_data_re
        / SOURCE_SYMBOLS
    ).long()
    return order[positions]


def bounded_power_weights(
    reliability: Tensor,
    *,
    alpha: float = .5,
    minimum_relative_power: float = .5,
    maximum_relative_power: float = 2.0,
    epsilon: float = 1e-8,
) -> Tensor:
    if not 0 < minimum_relative_power <= 1 <= maximum_relative_power:
        raise ValueError("power bounds must contain one")
    raw = (reliability.float().clamp_min(0) + epsilon).pow(alpha)
    low, high = 0.0, float(maximum_relative_power / raw.clamp_min(epsilon).min())
    for _ in range(80):
        scale = (low + high) / 2
        mean = raw.mul(scale).clamp(minimum_relative_power, maximum_relative_power).mean()
        if mean < 1:
            low = scale
        else:
            high = scale
    weights = raw.mul((low + high) / 2).clamp(
        minimum_relative_power, maximum_relative_power
    )
    # Close the finite-precision sum exactly enough for the fixed energy
    # contract without moving a bounded element outside its limits.
    residual = weights.new_tensor(float(weights.numel())) - weights.sum()
    adjustable = (
        (weights > minimum_relative_power + 1e-6)
        & (weights < maximum_relative_power - 1e-6)
    ).nonzero()
    if adjustable.numel():
        weights[adjustable[0, 0]] += residual
    return weights


def _mapping_for_selected(
    selected_reliability: Tensor, spread_rank: Tensor, importance: list[int]
) -> Tensor:
    reliability_order = torch.argsort(selected_reliability, descending=True, stable=True)
    result = torch.empty(SOURCE_SYMBOLS, dtype=torch.long)
    offsets = [0] * 8
    for quantile, destinations in enumerate(reliability_order.chunk(4)):
        destinations = destinations[
            torch.argsort(spread_rank[destinations], stable=True)
        ]
        cursor = 0
        for rank, layer in enumerate(importance):
            count = LAYER_ORDER_QUOTAS[quantile][rank]
            start = layer * SYMBOLS_PER_LAYER + offsets[layer]
            result[destinations[cursor : cursor + count]] = torch.arange(start, start + count)
            offsets[layer] += count
            cursor += count
    if offsets != [SYMBOLS_PER_LAYER] * 8 or torch.unique(result).numel() != SOURCE_SYMBOLS:
        raise AssertionError("physical allocation is not layer-preserving and bijective")
    return result


def allocate_physical_resources(
    *,
    profile: PhysicalOFDMProfile,
    tx_tti: int,
    report: PhysicalCSIReport | None,
    layer_importance_order: list[int],
    alpha: float = .5,
    minimum_relative_power: float = .5,
    maximum_relative_power: float = 2.0,
) -> PhysicalAllocation:
    if sorted(layer_importance_order) != list(range(8)):
        raise ValueError("layer importance must be a permutation")
    if tx_tti == 0:
        if report is not None:
            raise ValueError("TTI 0 must not have a CSI report")
        selected = _uniform_selection(profile)
        reliability = torch.ones(profile.candidate_data_re)
    else:
        if (
            report is None
            or report.available_tti != tx_tti
            or report.generated_tti >= tx_tti
        ):
            raise ValueError("allocation requires causally available past CSI")
        if report.reliability.shape != (profile.candidate_data_re,):
            raise ValueError("CSI report does not match profile candidate pool")
        reliability = report.reliability
        spread = _spread_order(profile)
        spread_rank = torch.empty_like(spread)
        spread_rank[spread] = torch.arange(spread.numel())
        # Reliability is primary; deterministic spread rank breaks ties.
        score = reliability.double() + (spread.numel() - spread_rank).double() * 1e-12
        selected = torch.topk(score, SOURCE_SYMBOLS, sorted=False).indices
        selected = selected[torch.argsort(spread_rank[selected], stable=True)]
    spread = _spread_order(profile)
    spread_rank_all = torch.empty_like(spread)
    spread_rank_all[spread] = torch.arange(spread.numel())
    selected_reliability = reliability[selected]
    mapping = _mapping_for_selected(
        selected_reliability, spread_rank_all[selected], layer_importance_order
    )
    power = bounded_power_weights(
        selected_reliability,
        alpha=alpha,
        minimum_relative_power=minimum_relative_power,
        maximum_relative_power=maximum_relative_power,
    )
    return PhysicalAllocation(profile, selected, mapping, power)


def recover_source_symbols(active_grid: Tensor, allocation: PhysicalAllocation) -> Tensor:
    masks = active_grid_masks(allocation.profile, device=active_grid.device)
    candidates = active_grid[..., masks.candidate_data]
    selected = candidates.index_select(
        -1, allocation.selected_candidate_indices.to(active_grid.device)
    )
    unpowered = selected / allocation.relative_power.to(active_grid.device).sqrt()
    restored = torch.empty_like(unpowered)
    restored[..., allocation.resource_to_source.to(active_grid.device)] = unpowered
    return restored


def lmmse_source_estimate(
    received: Tensor,
    channel_estimate: Tensor,
    relative_power: Tensor,
    *,
    noise_variance: float,
    source_power: float,
) -> tuple[Tensor, Tensor]:
    amplitude = relative_power.to(received.device).sqrt()
    effective_channel = channel_estimate * amplitude
    coefficient = effective_channel.conj() / (
        effective_channel.abs().square() + float(noise_variance) / float(source_power)
    )
    estimate = coefficient * received
    effective_gain = coefficient * effective_channel
    return estimate, effective_gain


__all__ = [
    "PhysicalAllocation",
    "PhysicalCSIReport",
    "allocate_physical_resources",
    "bounded_power_weights",
    "lmmse_source_estimate",
    "recover_source_symbols",
]
