from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype == torch.complex128 else torch.float32


def _complex_normal_like(reference: Tensor, generator: torch.Generator | None = None) -> Tensor:
    real = torch.randn(
        reference.shape,
        device=reference.device,
        dtype=_real_dtype(reference.dtype),
        generator=generator,
    )
    imag = torch.randn(
        reference.shape,
        device=reference.device,
        dtype=_real_dtype(reference.dtype),
        generator=generator,
    )
    return torch.complex(real, imag) / math.sqrt(2.0)


def _validate_reference(reference: Tensor) -> None:
    if not torch.is_complex(reference) or reference.ndim not in (2, 3):
        raise ValueError("reference must be complex [B, M] or [B, K, N]")


def _batch_parameter(value: float | Tensor, reference: Tensor) -> Tensor:
    result = torch.as_tensor(value, device=reference.device, dtype=_real_dtype(reference.dtype))
    if result.ndim == 0:
        result = result.repeat(reference.shape[0])
    if result.shape != (reference.shape[0],):
        raise ValueError(f"expected a scalar or one value per batch item, got {tuple(result.shape)}")
    return result.reshape(reference.shape[0], *([1] * (reference.ndim - 1)))


def _random_interval_mask(
    mask: Tensor,
    axis: int,
    active: int,
    generator: torch.Generator | None,
) -> None:
    length = mask.shape[axis]
    for batch_index in range(mask.shape[0]):
        start = int(
            torch.randint(
                0,
                length - active + 1,
                (),
                device=mask.device,
                generator=generator,
            ).item()
        )
        selection = [batch_index] + [slice(None)] * (mask.ndim - 1)
        selection[axis] = slice(start, start + active)
        mask[tuple(selection)] = True


def make_jammer_mask(
    reference_or_shape: Tensor | Sequence[int],
    kind: str = "barrage",
    jammed_fraction: float = 0.25,
    *,
    pilot_mask: Tensor | None = None,
    pilot_spacing: int = 4,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Build a jammer mask for flat symbols or an OFDM resource grid.

    For `[B, K, N]`, narrowband jamming selects contiguous subcarriers and
    burst jamming selects contiguous OFDM symbols. A supplied `pilot_mask` is
    used exactly for pilot jamming; otherwise a regular pilot comb is created.
    """
    if isinstance(reference_or_shape, Tensor):
        shape = tuple(reference_or_shape.shape)
        device = reference_or_shape.device
    else:
        shape = tuple(reference_or_shape)
    if len(shape) not in (2, 3) or any(size <= 0 for size in shape):
        raise ValueError("shape must be [B, M] or [B, K, N] with positive dimensions")
    if not 0.0 < jammed_fraction <= 1.0:
        raise ValueError("jammed_fraction must be in (0, 1]")

    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    if kind == "barrage":
        return mask.fill_(True)
    if kind == "narrowband":
        axis = 1
        active = max(1, round(shape[axis] * jammed_fraction))
        _random_interval_mask(mask, axis, active, generator)
        return mask
    if kind == "burst":
        axis = len(shape) - 1
        active = max(1, round(shape[axis] * jammed_fraction))
        _random_interval_mask(mask, axis, active, generator)
        return mask
    if kind == "pilot":
        if pilot_mask is not None:
            supplied = pilot_mask.to(device=mask.device, dtype=torch.bool)
            try:
                return torch.broadcast_to(supplied, shape).clone()
            except RuntimeError as error:
                raise ValueError("pilot_mask is not broadcastable to the resource shape") from error
        if pilot_spacing <= 0:
            raise ValueError("pilot_spacing must be positive")
        if len(shape) == 2:
            mask[:, ::pilot_spacing] = True
        else:
            mask[:, ::pilot_spacing, ::pilot_spacing] = True
        return mask
    raise ValueError(f"unsupported jammer kind: {kind}")


def compute_jsr(signal: Tensor, jammer: Tensor, *, db: bool = False, eps: float = 1e-12) -> Tensor:
    """Return per-example jammer-to-signal power ratio."""
    if signal.shape != jammer.shape or not torch.is_complex(signal) or not torch.is_complex(jammer):
        raise ValueError("signal and jammer must be matching complex tensors")
    dimensions = tuple(range(1, signal.ndim))
    ratio = jammer.abs().square().mean(dimensions) / signal.abs().square().mean(dimensions).clamp_min(eps)
    return 10.0 * torch.log10(ratio.clamp_min(eps)) if db else ratio


def jammer_mask_statistics(mask: Tensor) -> dict[str, Tensor]:
    """Return per-example active counts and mask ratios plus aggregate ratio."""
    if mask.ndim not in (2, 3):
        raise ValueError("mask must have shape [B, M] or [B, K, N]")
    dimensions = tuple(range(1, mask.ndim))
    active_count = mask.to(torch.int64).sum(dimensions)
    total_count = torch.full_like(active_count, mask[0].numel())
    mask_ratio = active_count.to(torch.float32) / total_count
    return {
        "active_count": active_count,
        "total_count": total_count,
        "mask_ratio": mask_ratio,
        "mean_mask_ratio": mask_ratio.mean(),
    }


def make_jammer(
    reference: Tensor,
    jsr_db: float | Tensor,
    kind: str = "barrage",
    jammed_fraction: float = 0.25,
    *,
    pilot_mask: Tensor | None = None,
    pilot_spacing: int = 4,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Generate masked complex Gaussian interference at a requested JSR.

    Power is averaged across every channel use/resource element, including
    unjammed elements. Thus all jammer types have the same total JSR while
    sparse attacks concentrate more power on their active resources.
    """
    _validate_reference(reference)
    mask = make_jammer_mask(
        reference,
        kind,
        jammed_fraction,
        pilot_mask=pilot_mask,
        pilot_spacing=pilot_spacing,
        generator=generator,
    )
    raw = _complex_normal_like(reference, generator) * mask
    dimensions = tuple(range(1, reference.ndim))
    keep_dimensions = tuple(range(1, reference.ndim))
    signal_power = reference.abs().square().mean(dimensions, keepdim=True)
    target_power = signal_power * torch.pow(10.0, _batch_parameter(jsr_db, reference) / 10.0)
    raw_power = raw.abs().square().mean(keep_dimensions, keepdim=True).clamp_min(1e-12)
    jammer = raw * torch.sqrt(target_power / raw_power)
    return jammer, mask

