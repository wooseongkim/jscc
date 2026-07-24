from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CSIReport:
    generated_slot: int
    available_slot: int
    reliability: Tensor

    @classmethod
    def from_reliability(
        cls, generated_slot: int, reliability: Tensor, delay_slots: int = 1
    ) -> "CSIReport":
        if delay_slots < 1:
            raise ValueError("causal FDD feedback delay must be at least one slot")
        if reliability.ndim != 1 or not torch.isfinite(reliability).all():
            raise ValueError("reliability must be a finite vector")
        return cls(
            int(generated_slot),
            int(generated_slot) + int(delay_slots),
            reliability.detach().clone(),
        )

    def clone(self) -> "CSIReport":
        return CSIReport(
            self.generated_slot, self.available_slot, self.reliability.clone()
        )


class DelayedCSIBuffer:
    def __init__(self, delay_slots: int = 1) -> None:
        if delay_slots < 1:
            raise ValueError("delay_slots must be at least one")
        self.delay_slots = int(delay_slots)
        self._reports: dict[int, CSIReport] = {}

    def submit(self, report: CSIReport) -> None:
        expected = report.generated_slot + self.delay_slots
        if report.available_slot != expected:
            raise ValueError("report delay does not match feedback buffer")
        self._reports[expected] = report.clone()

    def available_for_tx(self, slot: int) -> CSIReport | None:
        report = self._reports.get(int(slot))
        return None if report is None else report.clone()


def deterministic_interleaver(pilot_mask: Tensor) -> Tensor:
    if pilot_mask.ndim != 2 or pilot_mask.dtype != torch.bool:
        raise ValueError("pilot_mask must be a boolean [subcarrier,time] tensor")
    data_coordinates = (~pilot_mask).nonzero(as_tuple=False)
    resources = int(data_coordinates.shape[0])
    if resources != 1920:
        raise ValueError(f"expected 1920 data resources, got {resources}")
    k_count, n_count = pilot_mask.shape
    key = (
        ((data_coordinates[:, 1] * 13) % n_count) * k_count
        + ((data_coordinates[:, 0] * 17) % k_count)
    )
    destination_order = torch.argsort(key, stable=True)
    mapping = torch.empty(resources, dtype=torch.long, device=pilot_mask.device)
    mapping[destination_order] = torch.arange(resources, device=pilot_mask.device)
    return mapping


_QUOTAS = (
    (90, 80, 70, 65, 55, 50, 40, 30),
    (70, 65, 62, 61, 59, 58, 55, 50),
    (50, 55, 58, 59, 61, 62, 65, 70),
    (30, 40, 50, 55, 65, 70, 80, 90),
)


def allocate_from_report(
    tx_slot: int,
    report: CSIReport,
    base_mapping: Tensor,
    layer_importance_order: list[int] | tuple[int, ...],
) -> Tensor:
    if report.available_slot != tx_slot or report.generated_slot >= tx_slot:
        raise ValueError("TX allocation may use only a causally available past report")
    if base_mapping.shape != (1920,) or report.reliability.shape != (1920,):
        raise ValueError("mapping and reliability must contain 1920 resources")
    if sorted(layer_importance_order) != list(range(8)):
        raise ValueError("layer importance must be a permutation of 0..7")

    return _stratified_allocation(
        report.reliability, base_mapping, layer_importance_order
    )


def allocate_from_current_oracle(
    current_slot: int,
    current_true_reliability: Tensor,
    base_mapping: Tensor,
    layer_importance_order: list[int] | tuple[int, ...],
) -> Tensor:
    """Analysis-only current-CSI allocation upper bound.

    The explicit API prevents an oracle tensor from masquerading as a delayed
    report. Receiver equalization is intentionally outside this function.
    """
    if current_slot < 1:
        raise ValueError("slot 0 must use the uniform bootstrap allocation")
    return _stratified_allocation(
        current_true_reliability, base_mapping, layer_importance_order
    )


def _stratified_allocation(
    reliability: Tensor,
    base_mapping: Tensor,
    layer_importance_order: list[int] | tuple[int, ...],
) -> Tensor:
    if base_mapping.shape != (1920,) or reliability.shape != (1920,):
        raise ValueError("mapping and reliability must contain 1920 resources")
    if sorted(layer_importance_order) != list(range(8)):
        raise ValueError("layer importance must be a permutation of 0..7")
    reliability_order = torch.argsort(reliability, descending=True, stable=True)
    base_rank = torch.empty_like(base_mapping)
    base_rank[torch.argsort(base_mapping)] = torch.arange(
        base_mapping.numel(), device=base_mapping.device
    )
    source_offsets = {layer: 0 for layer in range(8)}
    result = torch.empty_like(base_mapping)
    for quantile, destinations in enumerate(reliability_order.chunk(4)):
        destinations = destinations[torch.argsort(base_rank[destinations], stable=True)]
        cursor = 0
        for importance_rank, layer in enumerate(layer_importance_order):
            count = _QUOTAS[quantile][importance_rank]
            start = layer * 240 + source_offsets[layer]
            result[destinations[cursor : cursor + count]] = torch.arange(
                start, start + count, device=result.device
            )
            source_offsets[layer] += count
            cursor += count
    if any(value != 240 for value in source_offsets.values()):
        raise AssertionError("allocation did not preserve 240 symbols per layer")
    if torch.unique(result).numel() != 1920:
        raise AssertionError("allocation must be bijective")
    return result


def apply_resource_map(source: Tensor, resource_to_source: Tensor) -> Tensor:
    return source.index_select(-1, resource_to_source.to(source.device))


def invert_resource_map(allocated: Tensor, resource_to_source: Tensor) -> Tensor:
    restored = torch.empty_like(allocated)
    restored[..., resource_to_source.to(allocated.device)] = allocated
    return restored


def mmse_equalize(
    received: Tensor, current_channel_estimate: Tensor, *, noise_power: float, signal_power: float
) -> Tensor:
    if noise_power < 0 or signal_power <= 0:
        raise ValueError("invalid signal/noise power")
    coefficient = current_channel_estimate.conj() / (
        current_channel_estimate.abs().square() + float(noise_power) / float(signal_power)
    )
    return coefficient * received


__all__ = [
    "CSIReport",
    "DelayedCSIBuffer",
    "allocate_from_current_oracle",
    "allocate_from_report",
    "apply_resource_map",
    "deterministic_interleaver",
    "invert_resource_map",
    "mmse_equalize",
]
