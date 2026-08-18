"""Receiver-estimated risk maps for causal R4 resource placement.

The map is deliberately a *receiver observation* only.  Oracle mask conversion
is separately named for upper-bound experiments and is never invoked by the
deployable residual-risk estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from channels.global_triplet_allocator import (
    GlobalTripletCSIReport,
    GlobalTripletAllocation,
    allocate_global_balanced_triplets,
)


@dataclass(frozen=True)
class RERiskReport:
    generated_tti: int
    available_tti: int
    risk: Tensor

    @classmethod
    def from_risk(cls, generated_tti: int, risk: Tensor, delay_ttis: int = 1) -> "RERiskReport":
        if delay_ttis < 1 or risk.ndim != 1:
            raise ValueError("invalid RE risk report")
        if not torch.isfinite(risk).all() or (risk < 0).any():
            raise ValueError("RE risk must be finite and nonnegative")
        # Delayed CSI reports are stored on CPU; match that contract so a
        # next-TTI CUDA risk observation can be combined without device drift.
        return cls(int(generated_tti), int(generated_tti) + int(delay_ttis), risk.detach().cpu().clone())


@dataclass(frozen=True)
class REInterferenceReport:
    """Causal receiver estimate of additive interference power per data RE.

    Unlike the historical normalized ``RERiskReport``, this quantity stays in
    power units so it can appear directly in an allocation SINR denominator.
    It contains no true jammer labels, masks, or jammer samples.
    """

    generated_tti: int
    available_tti: int
    interference_power: Tensor
    noise_power: float

    @classmethod
    def from_power(
        cls,
        generated_tti: int,
        interference_power: Tensor,
        *,
        noise_power: float,
        delay_ttis: int = 1,
    ) -> "REInterferenceReport":
        if delay_ttis < 1 or interference_power.ndim != 1:
            raise ValueError("invalid RE interference report")
        if not torch.isfinite(interference_power).all() or (interference_power < 0).any():
            raise ValueError("RE interference must be finite and nonnegative")
        if not float(noise_power) > 0 or not torch.isfinite(torch.tensor(float(noise_power))):
            raise ValueError("noise power must be finite and positive")
        return cls(
            int(generated_tti),
            int(generated_tti) + int(delay_ttis),
            interference_power.detach().cpu().clone(),
            float(noise_power),
        )


def require_available_interference_report(
    *, tx_tti: int, report: REInterferenceReport | None
) -> REInterferenceReport:
    if report is None or report.available_tti != tx_tti or report.generated_tti >= tx_tti:
        raise ValueError("interference report is not causally available")
    return report


def _normalize_risk(risk: Tensor, eps: float) -> Tensor:
    minimum = risk.min()
    return (risk - minimum) / (risk.max() - minimum + eps)


def combine_reliability_with_risk(reliability: Tensor, risk: Tensor, alpha: float, eps: float = 1e-12) -> Tensor:
    """Penalize high-risk candidates without changing a zero-alpha baseline."""
    if reliability.shape != risk.shape or reliability.ndim != 1:
        raise ValueError("reliability and risk must be equal one-dimensional tensors")
    if alpha < 0:
        raise ValueError("risk alpha must be nonnegative")
    if alpha == 0:
        return reliability
    if not torch.isfinite(reliability).all() or not torch.isfinite(risk).all():
        raise ValueError("reliability and risk must be finite")
    return reliability / (1.0 + float(alpha) * _normalize_risk(risk.clamp_min(0), eps))


def combine_csi_and_risk_report(
    tx_tti: int,
    csi_report: GlobalTripletCSIReport | None,
    risk_report: RERiskReport | None,
    risk_alpha: float,
    eps: float = 1e-12,
) -> GlobalTripletCSIReport | None:
    """Create a causal effective reliability report for a future transmission."""
    if risk_alpha == 0 or risk_report is None:
        return csi_report
    if csi_report is None:
        # Bootstrap TTI has no delayed CSI reliability to combine.
        return None
    if csi_report.available_tti != tx_tti or csi_report.generated_tti >= tx_tti:
        raise ValueError("CSI report is not causally available")
    if risk_report.available_tti != tx_tti or risk_report.generated_tti >= tx_tti:
        raise ValueError("RE risk report is not causally available")
    if csi_report.reliability.shape != risk_report.risk.shape:
        raise ValueError("CSI reliability and RE risk dimensions differ")
    effective = combine_reliability_with_risk(csi_report.reliability, risk_report.risk, risk_alpha, eps)
    return GlobalTripletCSIReport(int(csi_report.generated_tti), int(tx_tti), effective)


def allocate_risk_aware_global_balanced_triplets(
    *,
    profile,
    tx_tti: int,
    csi_report: GlobalTripletCSIReport | None,
    risk_report: RERiskReport | None,
    risk_alpha: float,
    layer_importance_order: list[int],
    **kwargs,
) -> GlobalTripletAllocation:
    """Thin risk-aware wrapper around the canonical global triplet allocator."""
    # Direct delegation is intentional: it preserves legacy mapping bitwise.
    if risk_alpha == 0 or risk_report is None:
        return allocate_global_balanced_triplets(
            profile=profile, tx_tti=tx_tti, report=csi_report,
            layer_importance_order=layer_importance_order, **kwargs,
        )
    effective_report = combine_csi_and_risk_report(tx_tti, csi_report, risk_report, risk_alpha)
    return allocate_global_balanced_triplets(
        profile=profile, tx_tti=tx_tti, report=effective_report,
        layer_importance_order=layer_importance_order, **kwargs,
    )


def estimate_rx_residual_risk_map(
    received_grid: Tensor,
    pilots: Tensor,
    pilot_mask: Tensor,
    estimated_channel: Tensor,
    noise_variance: Tensor | float,
    candidate_data_mask: Tensor,
    *,
    normalize: bool = True,
    eps: float = 1e-12,
) -> Tensor:
    """Estimate candidate-RE risk from receiver-observable residual features.

    It combines interpolated pilot residual energy, inverse estimated channel
    gain, and a local received-energy anomaly.  It never accepts a jammer
    label, jammer tensor, or true jammer mask.
    """
    if received_grid.ndim != 3 or not torch.is_complex(received_grid):
        raise ValueError("received_grid must be complex [B,K,N]")
    if pilots.shape != received_grid.shape or estimated_channel.shape != received_grid.shape:
        raise ValueError("pilots and estimated_channel must match received_grid")
    if pilot_mask.shape != received_grid.shape[-2:] or candidate_data_mask.shape != received_grid.shape[-2:]:
        raise ValueError("pilot and candidate masks must match [K,N]")
    noise = torch.as_tensor(noise_variance, device=received_grid.device, dtype=received_grid.real.dtype)
    if noise.ndim == 0:
        noise = noise.expand(received_grid.shape[0])
    if noise.shape != (received_grid.shape[0],):
        raise ValueError("noise_variance must be scalar or [B]")
    pilot = pilot_mask.to(device=received_grid.device, dtype=received_grid.real.dtype)[None]
    residual = (received_grid - estimated_channel * pilots).abs().square()
    pilot_residual = (residual * pilot).sum(-1) / pilot.sum(-1).clamp_min(1.0)
    pilot_feature = pilot_residual[:, :, None].expand_as(residual)
    inverse_gain = 1.0 / estimated_channel.abs().square().clamp_min(eps)
    energy = received_grid.abs().square()
    local_reference = energy.median(dim=-1, keepdim=True).values.clamp_min(eps)
    anomaly = (energy / local_reference - 1.0).clamp_min(0.0)
    risk_grid = pilot_feature / noise[:, None, None].clamp_min(eps) + inverse_gain + anomaly
    risk = risk_grid.mean(0)[candidate_data_mask.to(device=received_grid.device)]
    if normalize:
        risk = _normalize_risk(risk, eps)
    if not torch.isfinite(risk).all():
        raise FloatingPointError("nonfinite receiver-estimated RE risk")
    return risk


def estimate_rx_residual_interference_power(
    received_grid: Tensor,
    pilots: Tensor,
    pilot_mask: Tensor,
    estimated_channel: Tensor,
    noise_variance: Tensor | float,
    candidate_data_mask: Tensor,
    *,
    eps: float = 1e-12,
) -> Tensor:
    """Estimate additive interference power without channel-gain double count.

    Pilot residual power above the known noise floor is interpolated along the
    OFDM-symbol axis.  A positive local received-energy residual is added as a
    receiver-observable anomaly term.  The output is deliberately left in
    power units rather than normalized to [0, 1].
    """
    if received_grid.ndim != 3 or not torch.is_complex(received_grid):
        raise ValueError("received_grid must be complex [B,K,N]")
    if pilots.shape != received_grid.shape or estimated_channel.shape != received_grid.shape:
        raise ValueError("pilots and estimated_channel must match received_grid")
    if pilot_mask.shape != received_grid.shape[-2:] or candidate_data_mask.shape != received_grid.shape[-2:]:
        raise ValueError("pilot and candidate masks must match [K,N]")
    noise = torch.as_tensor(noise_variance, device=received_grid.device, dtype=received_grid.real.dtype)
    if noise.ndim == 0:
        noise = noise.expand(received_grid.shape[0])
    if noise.shape != (received_grid.shape[0],):
        raise ValueError("noise_variance must be scalar or [B]")
    pilot = pilot_mask.to(device=received_grid.device, dtype=received_grid.real.dtype)[None]
    residual_power = (received_grid - estimated_channel * pilots).abs().square()
    # A per-subcarrier pilot residual is causally observable and can be used
    # for every candidate time RE on that subcarrier.
    pilot_count = pilot.sum(-1).clamp_min(1.0)
    pilot_excess = ((residual_power * pilot).sum(-1) / pilot_count - noise[:, None]).clamp_min(0.0)
    pilot_feature = pilot_excess[:, :, None].expand_as(residual_power)
    energy = received_grid.abs().square()
    local_reference = energy.median(dim=-1, keepdim=True).values
    anomaly = (energy - local_reference).clamp_min(0.0)
    interference = (pilot_feature + anomaly).mean(0)[
        candidate_data_mask.to(device=received_grid.device)
    ]
    if not torch.isfinite(interference).all():
        raise FloatingPointError("nonfinite receiver-estimated RE interference")
    return interference


def oracle_jamming_mask_to_risk_report(
    jammer_mask: Tensor,
    candidate_data_mask: Tensor,
    *,
    generated_tti: int,
    delay_ttis: int = 1,
) -> RERiskReport:
    """Upper-bound-only conversion of a true jammer mask into an RE risk report."""
    if jammer_mask.ndim != 3 or jammer_mask.shape[-2:] != candidate_data_mask.shape:
        raise ValueError("jammer mask must be [B,K,N] matching candidate data mask")
    risk = jammer_mask.to(dtype=torch.float32).mean(0)[candidate_data_mask.to(jammer_mask.device)]
    return RERiskReport.from_risk(generated_tti, risk, delay_ttis)


def oracle_jammer_grid_to_interference_report(
    jammer_grid: Tensor,
    candidate_data_mask: Tensor,
    *,
    generated_tti: int,
    noise_power: float,
    delay_ttis: int = 1,
) -> REInterferenceReport:
    """Upper-bound-only jammer-power report; never use in deployable mode."""
    if jammer_grid.ndim != 3 or jammer_grid.shape[-2:] != candidate_data_mask.shape:
        raise ValueError("jammer grid must be [B,K,N] matching candidate data mask")
    power = jammer_grid.abs().square().mean(0)[
        candidate_data_mask.to(jammer_grid.device)
    ]
    return REInterferenceReport.from_power(
        generated_tti, power, noise_power=noise_power, delay_ttis=delay_ttis
    )


__all__ = [
    "RERiskReport", "allocate_risk_aware_global_balanced_triplets",
    "combine_csi_and_risk_report", "combine_reliability_with_risk",
    "estimate_rx_residual_risk_map", "oracle_jamming_mask_to_risk_report",
    "REInterferenceReport", "estimate_rx_residual_interference_power",
    "require_available_interference_report", "oracle_jammer_grid_to_interference_report",
]
