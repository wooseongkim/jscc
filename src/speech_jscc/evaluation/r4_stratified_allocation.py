"""Conservative fixed-tier power profiles for paired R4 allocation evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import torch
from torch import Tensor


LAYERS = 8
SYMBOLS_PER_LAYER = 240
SOURCE_SYMBOLS = LAYERS * SYMBOLS_PER_LAYER
COPIES = 3
PRIMARY_PROFILES = ("uniform", "core_protection", "layer1_focused")


@dataclass(frozen=True)
class AllocationProfile:
    name: str
    raw_weights: tuple[float, ...]
    mapping_policy: str = "balanced_triplet"
    repetition_policy: str = "fixed_three"
    power_policy: str = "uniform"


_PROFILES = {
    "uniform": AllocationProfile("uniform", (1.0,) * LAYERS, power_policy="uniform"),
    "core_protection": AllocationProfile(
        "core_protection", (1.25,) * 4 + (0.75,) * 4, power_policy="core_protection"
    ),
    "layer1_focused": AllocationProfile(
        "layer1_focused", (1.10, 1.45, 1.10, 1.10, 0.80, 0.80, 0.80, 0.80),
        power_policy="layer1_focused",
    ),
}


def allocation_profile(name: str) -> AllocationProfile:
    try:
        return _PROFILES[name]
    except KeyError as error:
        raise ValueError(f"unknown allocation profile: {name}") from error


def source_layer_indices(*, device: torch.device | None = None) -> Tensor:
    return torch.arange(LAYERS, device=device).repeat_interleave(SYMBOLS_PER_LAYER)


def layer_symbol_indices(layer: int, *, device: torch.device | None = None) -> Tensor:
    if not 0 <= int(layer) < LAYERS:
        raise ValueError(f"layer must be in [0,{LAYERS})")
    return torch.arange(layer * SYMBOLS_PER_LAYER, (layer + 1) * SYMBOLS_PER_LAYER, device=device)


def normalize_layer_power_weights(
    raw_weights: Sequence[float], symbol_counts: Sequence[int] | None = None
) -> Tensor:
    if len(raw_weights) != LAYERS:
        raise ValueError(f"profile must have exactly {LAYERS} weights")
    raw = torch.tensor(raw_weights, dtype=torch.float64)
    if not torch.isfinite(raw).all() or (raw <= 0).any():
        raise ValueError("profile power weights must be finite and positive")
    counts = torch.tensor(
        symbol_counts if symbol_counts is not None else (SYMBOLS_PER_LAYER,) * LAYERS,
        dtype=torch.float64,
    )
    if counts.shape != (LAYERS,) or (counts <= 0).any():
        raise ValueError("symbol counts must be eight positive values")
    return raw / (torch.dot(counts, raw) / counts.sum())


def source_order_power_multiplier(layer_weights: Tensor, source_layers: Tensor | None = None) -> Tensor:
    weights = torch.as_tensor(layer_weights)
    if weights.shape != (LAYERS,):
        raise ValueError(f"layer weights must have shape [{LAYERS}]")
    layers = source_layer_indices(device=weights.device) if source_layers is None else source_layers.to(weights.device)
    if layers.shape != (SOURCE_SYMBOLS,) or int(layers.min()) < 0 or int(layers.max()) >= LAYERS:
        raise ValueError("source layer indices must map every one of 1920 symbols to a layer")
    return weights.index_select(0, layers.long())


def compose_source_power(
    allocation_power_source_order: Tensor, layer_weights: Tensor, source_layers: Tensor | None = None
) -> Tensor:
    if allocation_power_source_order.shape != (COPIES, SOURCE_SYMBOLS):
        raise ValueError("allocation power must have shape [3,1920]")
    multiplier = source_order_power_multiplier(layer_weights, source_layers).to(
        allocation_power_source_order
    )
    result = allocation_power_source_order * multiplier[None]
    result = result * (allocation_power_source_order.sum() / result.sum())
    if not torch.isfinite(result).all() or (result <= 0).any():
        raise FloatingPointError("composed source power is nonfinite or nonpositive")
    return result


def destination_power_from_source_order(source_power: Tensor, resource_to_source: Tensor) -> Tensor:
    """Invert allocation.extract_source_order for a source-order power tensor."""
    if source_power.shape != (COPIES, SOURCE_SYMBOLS):
        raise ValueError("source power must have shape [3,1920]")
    if resource_to_source.shape != (SOURCE_SYMBOLS,) or torch.unique(resource_to_source).numel() != SOURCE_SYMBOLS:
        raise ValueError("resource-to-source mapping must be a 1920-element permutation")
    return source_power.index_select(-1, resource_to_source.to(source_power.device))


def sqrt_power_amplitude(power: Tensor) -> Tensor:
    if not torch.isfinite(power).all() or (power < 0).any():
        raise ValueError("power must be finite and nonnegative")
    return power.sqrt()


def measured_average_power(power_source_order: Tensor) -> float:
    if power_source_order.shape != (COPIES, SOURCE_SYMBOLS):
        raise ValueError("power must have shape [3,1920]")
    return float(power_source_order.mean())


def measured_transmit_symbol_power(source: Tensor, power_source_order: Tensor) -> Tensor:
    if source.shape[-1] != SOURCE_SYMBOLS or not source.is_complex():
        raise ValueError("source must be complex with final dimension 1920")
    if power_source_order.shape != (COPIES, SOURCE_SYMBOLS):
        raise ValueError("power must have shape [3,1920]")
    return (source.abs().square() * power_source_order.to(source.device).sum(0)[None]).mean()


def preserve_measured_transmit_power(source: Tensor, base_power: Tensor, profile_power: Tensor) -> Tensor:
    baseline = measured_transmit_symbol_power(source, base_power)
    current = measured_transmit_symbol_power(source, profile_power)
    if not torch.isfinite(baseline + current) or float(current) <= 0:
        raise FloatingPointError("nonfinite measured transmit power")
    return profile_power.to(source.device) * (baseline / current)


def validate_paired_profile_rows(rows: Iterable[dict]) -> None:
    grouped: dict[tuple[object, object], set[str]] = {}
    for row in rows:
        try:
            key = (row["utterance_id"], row["realization"])
            profile = str(row["profile"])
        except KeyError as error:
            raise ValueError(f"paired row missing {error.args[0]}") from error
        grouped.setdefault(key, set()).add(profile)
    expected = set(PRIMARY_PROFILES)
    for key, present in grouped.items():
        if present != expected:
            raise ValueError(f"paired profiles missing for {key}: expected={sorted(expected)} present={sorted(present)}")


def _percentile(values: Tensor, percentile: float) -> float:
    return float(torch.quantile(values, percentile))


def paired_bootstrap_summary(
    rows: Sequence[dict], *, profile: str, metric: str, samples: int, seed: int
) -> dict[str, float | int | list[float]]:
    """Summarize profile-minus-uniform deltas after averaging realizations per utterance."""
    if profile not in PRIMARY_PROFILES or profile == "uniform":
        raise ValueError("paired summary requires a non-uniform primary profile")
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    validate_paired_profile_rows(rows)
    paired: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        paired.setdefault(str(row["utterance_id"]), {}).setdefault(str(row["profile"]), []).append(float(row[metric]))
    deltas = torch.tensor([
        sum(member[profile]) / len(member[profile]) - sum(member["uniform"]) / len(member["uniform"])
        for member in paired.values()
    ], dtype=torch.float64)
    generator = torch.Generator().manual_seed(int(seed))
    boot = torch.stack([
        deltas[torch.randint(len(deltas), (len(deltas),), generator=generator)].mean()
        for _ in range(samples)
    ])
    mean = float(deltas.mean())
    return {
        "n_utterances": int(deltas.numel()), "mean": mean, "median": _percentile(deltas, .5),
        "std": float(deltas.std(unbiased=False)),
        "standard_error": float(deltas.std(unbiased=True) / math.sqrt(len(deltas))) if len(deltas) > 1 else 0.0,
        "p5": _percentile(deltas, .05), "p10": _percentile(deltas, .10), "p25": _percentile(deltas, .25),
        "p75": _percentile(deltas, .75), "p90": _percentile(deltas, .90), "p95": _percentile(deltas, .95),
        "min": float(deltas.min()), "max": float(deltas.max()),
        "positive_gain_fraction": float((deltas > 0).double().mean()),
        "bootstrap_95_ci": [_percentile(boot, .025), _percentile(boot, .975)],
    }


def reference_check(
    measured: dict[str, float], expected: dict[str, float], *,
    si_sdr_tolerance: float, waveform_snr_tolerance: float, stft_tolerance: float,
) -> dict[str, object]:
    tolerances = {
        "si_sdr_db": float(si_sdr_tolerance), "waveform_snr_db": float(waveform_snr_tolerance),
        "stft_l1": float(stft_tolerance),
    }
    differences = {key: abs(float(measured[key]) - float(expected[key])) for key in tolerances}
    return {"expected": expected, "measured": measured, "absolute_differences": differences,
            "tolerances": tolerances, "passed": all(differences[key] <= tolerances[key] for key in tolerances)}


__all__ = [
    "AllocationProfile", "COPIES", "LAYERS", "PRIMARY_PROFILES", "SOURCE_SYMBOLS",
    "SYMBOLS_PER_LAYER", "allocation_profile", "compose_source_power", "layer_symbol_indices",
    "destination_power_from_source_order",
    "measured_average_power", "normalize_layer_power_weights", "source_layer_indices",
    "source_order_power_multiplier", "sqrt_power_amplitude", "validate_paired_profile_rows",
    "paired_bootstrap_summary", "reference_check", "measured_transmit_symbol_power",
    "preserve_measured_transmit_power",
]
