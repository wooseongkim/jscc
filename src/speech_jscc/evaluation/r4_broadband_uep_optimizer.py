"""Pure, deterministic core for broadband R4 UEP black-box optimization.

The module deliberately has no waveform-forward, manifest, or legacy-final
dependency.  The CLI supplies paired summaries produced by the existing R4
UEP evaluator; this module only constructs feasible decisions and ranks them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import itertools
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


LAYERS = 8
SYMBOLS_PER_LAYER = 240
TOTAL_REPETITIONS = 24
TOTAL_DATA_RE = SYMBOLS_PER_LAYER * TOTAL_REPETITIONS


@dataclass(frozen=True)
class UEPCandidate:
    repetition: tuple[int, ...]
    logits: tuple[float, ...]
    power_share: tuple[float, ...]
    per_re_power: tuple[float, ...]

    @property
    def candidate_id(self) -> str:
        return candidate_hash(self)

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["candidate_id"] = self.candidate_id
        return payload


def validate_repetition(repetition: Sequence[int]) -> tuple[int, ...]:
    value = tuple(int(item) for item in repetition)
    if len(value) != LAYERS:
        raise ValueError("repetition must contain eight layer values")
    if any(item < 1 or item > 5 for item in value):
        raise ValueError("each repetition value must be between 1 and 5")
    if sum(value) != TOTAL_REPETITIONS:
        raise ValueError("repetition values must sum to 24")
    return value


def enumerate_feasible_repetitions(
    *, layers: int = LAYERS, total: int = TOTAL_REPETITIONS,
    minimum: int = 1, maximum: int = 5,
) -> Iterable[tuple[int, ...]]:
    """Yield every feasible integer repetition vector in lexical order."""
    if layers <= 0 or minimum > maximum:
        return

    def recurse(prefix: tuple[int, ...], remaining_layers: int, remaining: int):
        if remaining_layers == 0:
            if remaining == 0:
                yield prefix
            return
        low = max(minimum, remaining - maximum * (remaining_layers - 1))
        high = min(maximum, remaining - minimum * (remaining_layers - 1))
        for item in range(low, high + 1):
            yield from recurse(prefix + (item,), remaining_layers - 1, remaining - item)

    yield from recurse((), layers, total)


def softmax_power(logits: Sequence[float]) -> tuple[float, ...]:
    if len(logits) != LAYERS:
        raise ValueError("power logits must contain eight values")
    values = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("power logits must be finite")
    shifted = values - values.max()
    exp = np.exp(shifted)
    power = exp / exp.sum()
    return tuple(float(item) for item in power)


def per_re_power(repetition: Sequence[int], power_share: Sequence[float]) -> tuple[float, ...]:
    repetitions = validate_repetition(repetition)
    if len(power_share) != LAYERS:
        raise ValueError("power share must contain eight values")
    shares = tuple(float(item) for item in power_share)
    if any(not math.isfinite(item) or item <= 0 for item in shares):
        raise ValueError("power shares must be finite and positive")
    if not math.isclose(sum(shares), 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("power shares must sum to one")
    return tuple(TOTAL_REPETITIONS * share / copies for copies, share in zip(repetitions, shares))


def total_re(repetition: Sequence[int]) -> int:
    return SYMBOLS_PER_LAYER * sum(validate_repetition(repetition))


def total_energy(repetition: Sequence[int], per_re: Sequence[float]) -> float:
    repetitions = validate_repetition(repetition)
    if len(per_re) != LAYERS:
        raise ValueError("per-RE power must contain eight values")
    return float(SYMBOLS_PER_LAYER * sum(copies * float(power) for copies, power in zip(repetitions, per_re)))


def make_candidate(repetition: Sequence[int], logits: Sequence[float]) -> UEPCandidate:
    repetitions = validate_repetition(repetition)
    logit_tuple = tuple(float(item) for item in logits)
    shares = softmax_power(logit_tuple)
    powers = per_re_power(repetitions, shares)
    if not math.isclose(total_energy(repetitions, powers), float(TOTAL_DATA_RE), abs_tol=1e-7):
        raise AssertionError("candidate does not preserve total transmit energy")
    return UEPCandidate(repetitions, logit_tuple, shares, powers)


def candidate_hash(candidate: UEPCandidate) -> str:
    payload = {
        "repetition": list(candidate.repetition),
        "power_share": [round(value, 12) for value in candidate.power_share],
        "per_re_power": [round(value, 12) for value in candidate.per_re_power],
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]


def sample_random_candidates(*, count: int, seed: int, alpha_values: Sequence[float] = (0.5, 1.0, 2.0, 5.0)) -> list[UEPCandidate]:
    """Sample feasible decisions without a layer-importance input or bias."""
    if count < 0:
        raise ValueError("count must be nonnegative")
    if not alpha_values or any(value <= 0 for value in alpha_values):
        raise ValueError("Dirichlet alpha values must be positive")
    feasible = tuple(enumerate_feasible_repetitions())
    generator = np.random.default_rng(seed)
    candidates: list[UEPCandidate] = [make_candidate((3,) * LAYERS, (0.0,) * LAYERS)]
    seen = {candidate_hash(candidates[0])}
    while len(candidates) < count:
        repetition = feasible[int(generator.integers(0, len(feasible)))]
        alpha = float(alpha_values[(len(candidates) - 1) % len(alpha_values)])
        shares = generator.dirichlet(np.full(LAYERS, alpha, dtype=np.float64))
        logits = np.log(shares)
        candidate = make_candidate(repetition, logits)
        if candidate_hash(candidate) not in seen:
            candidates.append(candidate)
            seen.add(candidate_hash(candidate))
    return candidates[:count]


def propose_repetition_moves(repetition: Sequence[int]) -> list[tuple[int, ...]]:
    baseline = validate_repetition(repetition)
    proposals: list[tuple[int, ...]] = []
    for receive in range(LAYERS):
        for donate in range(LAYERS):
            if receive == donate or baseline[receive] == 5 or baseline[donate] == 1:
                continue
            proposal = list(baseline)
            proposal[receive] += 1
            proposal[donate] -= 1
            proposals.append(tuple(proposal))
    return proposals


def propose_power_transfers(power_share: Sequence[float], *, deltas: Sequence[float] = (0.05, 0.025, 0.01, 0.005), eps: float = 1e-9) -> list[tuple[float, ...]]:
    if len(power_share) != LAYERS or any(item <= 0 for item in power_share):
        raise ValueError("power shares must be positive eight-vector")
    if not math.isclose(sum(power_share), 1.0, abs_tol=1e-10):
        raise ValueError("power shares must sum to one")
    proposals: list[tuple[float, ...]] = []
    for delta in deltas:
        if delta <= 0:
            raise ValueError("power transfer must be positive")
        for receive in range(LAYERS):
            for donate in range(LAYERS):
                if receive == donate or power_share[donate] - delta <= eps:
                    continue
                proposal = list(float(item) for item in power_share)
                proposal[receive] += delta
                proposal[donate] -= delta
                proposals.append(tuple(proposal))
    return proposals


def _safe(summary: Mapping[str, float], key: str, default: float = 0.0) -> float:
    value = summary.get(key, default)
    return float(value) if value is not None else default


def objective_from_summary(
    summary: Mapping[str, float], *, lambda_clean: float = 1.0,
    lambda_tail: float = 1.0, lambda_stft: float = 1.0,
) -> dict[str, float]:
    """Calculate scalar value from *paired U0* aggregate metrics only."""
    paired_delta = _safe(summary, "weighted_mean_delta_si_sdr_db")
    clean_cost = _safe(summary, "clean_cost_db")
    p5_delta = _safe(summary, "p5_delta_si_sdr_db")
    severe_tail = max(0.0, _safe(summary, "severe_tail_increase_fraction"))
    stft_regression = max(0.0, _safe(summary, "stft_l1_delta"))
    clean_penalty = max(0.0, clean_cost - 0.2) * lambda_clean
    tail_penalty = (max(0.0, -1.0 - p5_delta) + severe_tail) * lambda_tail
    stft_penalty = stft_regression * lambda_stft
    return {
        "paired_weighted_delta_si_sdr_db": paired_delta,
        "clean_cost_penalty": clean_penalty,
        "tail_risk_penalty": tail_penalty,
        "stft_penalty": stft_penalty,
        "score": paired_delta - clean_penalty - tail_penalty - stft_penalty,
    }


def _pareto_metrics(record: Mapping) -> tuple[float, float, float, float, float, float]:
    summary = record["summary"]
    return (
        _safe(summary, "mean_delta_si_sdr_db"),
        _safe(summary, "p5_delta_si_sdr_db"),
        _safe(summary, "absolute_catastrophic_reduction_fraction"),
        -_safe(summary, "clean_cost_db"),
        -_safe(summary, "stft_l1_delta"),
        -max(0.0, _safe(summary, "severe_tail_increase_fraction")),
    )


def pareto_front(records: Sequence[Mapping]) -> list[Mapping]:
    front: list[Mapping] = []
    for index, candidate in enumerate(records):
        values = _pareto_metrics(candidate)
        dominated = False
        for other_index, other in enumerate(records):
            if index == other_index:
                continue
            other_values = _pareto_metrics(other)
            if all(left >= right for left, right in zip(other_values, values)) and any(left > right for left, right in zip(other_values, values)):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


def _with_objective(record: Mapping) -> dict:
    output = dict(record)
    output["objective"] = objective_from_summary(output["summary"])
    return output


def rank_by_objective(records: Sequence[Mapping]) -> list[Mapping]:
    """Return every evaluated candidate in current scalar-objective order."""
    return sorted(
        records,
        key=lambda record: (_safe(record["objective"], "score"), record["candidate_id"]),
        reverse=True,
    )


def select_profiles(records: Sequence[Mapping], front: Sequence[Mapping] | None = None) -> dict[str, Mapping | None]:
    if not records:
        return {"x_best": None, "x_stable": None, "x_aggressive": None}
    scored = [_with_objective(record) for record in records]
    front_ids = {record["candidate_id"] for record in (front if front is not None else pareto_front(records))}
    pareto_scored = [record for record in scored if record["candidate_id"] in front_ids]
    best = max(scored, key=lambda record: (record["objective"]["score"], record["candidate_id"]))
    stable_options = [
        record for record in pareto_scored
        if _safe(record["summary"], "clean_cost_db") <= 0.2
        and _safe(record["summary"], "p5_delta_si_sdr_db") >= -1.0
        and _safe(record["summary"], "severe_tail_increase_fraction") <= 0.02
    ]
    aggressive_options = [
        record for record in pareto_scored
        if _safe(record["summary"], "clean_cost_db") <= 0.3
        and (
            _safe(record["summary"], "mean_delta_si_sdr_db") > 0.0
            or _safe(record["summary"], "absolute_catastrophic_reduction_fraction") > 0.0
        )
    ]
    key_mean = lambda record: (_safe(record["summary"], "mean_delta_si_sdr_db"), record["candidate_id"])
    return {
        "x_best": best,
        "x_stable": max(stable_options, key=key_mean) if stable_options else None,
        "x_aggressive": max(aggressive_options, key=key_mean) if aggressive_options else None,
    }


def validate_search_split(split: str) -> str:
    if split != "expanded_selection":
        raise ValueError("search is restricted to expanded_selection")
    return split


def build_selected_profiles_artifact(*, checkpoint: Mapping, selected: Mapping[str, Mapping | None], candidate_count: int) -> dict:
    def serialized(kind: str, record: Mapping | None):
        if record is None:
            return {
                "status": "NOT_FOUND", "reason": "no_candidate_satisfies_selection_rule",
                "candidate": None, "mean_delta_si_sdr": None, "scalar_objective": None,
                "clean_cost": None, "p5_delta": None, "relative_tail_fraction": None,
                "absolute_catastrophic_fraction": None, "selected_for_final_eval": False,
            }
        summary = record["summary"]
        score = _safe(record.get("objective", {}), "score")
        mean = _safe(summary, "mean_delta_si_sdr_db")
        if record.get("candidate_id") == "U0":
            status, reason = "SELECTED", "u0_baseline_no_nonuniform_gain_found"
        elif kind == "x_best" and score <= 0.0:
            status, reason = "NEGATIVE_OBJECTIVE", "scalar_objective_is_not_positive"
        elif kind == "x_best" and mean <= 0.0:
            status, reason = "NO_GAIN", "mean_broadband_delta_is_not_positive"
        else:
            status, reason = "SELECTED", "satisfies_selection_rule"
        return {
            "status": status, "reason": reason, "candidate": record,
            "mean_delta_si_sdr": mean, "scalar_objective": score,
            "clean_cost": _safe(summary, "clean_cost_db"),
            "p5_delta": _safe(summary, "p5_delta_si_sdr_db"),
            "relative_tail_fraction": _safe(summary, "severe_tail_increase_fraction"),
            "absolute_catastrophic_fraction": _safe(summary, "absolute_catastrophic_fraction"),
            "selected_for_final_eval": status == "SELECTED",
        }

    return {
        "schema_version": 1,
        "artifact_type": "r4_broadband_uep_optimization_selection",
        "selection_split": "expanded_selection",
        "selection_uses_legacy_metrics": False,
        "checkpoint": dict(checkpoint),
        "candidate_count": int(candidate_count),
        "selected": {key: serialized(key, value) for key, value in selected.items()},
    }


def load_selected_profiles(path: str | Path) -> dict:
    selection_path = Path(path)
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    payload = json.loads(selection_path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported selected_profiles schema")
    if payload.get("artifact_type") != "r4_broadband_uep_optimization_selection":
        raise ValueError("selected_profiles artifact type is invalid")
    if payload.get("selection_split") != "expanded_selection":
        raise ValueError("selected_profiles must originate from expanded_selection")
    if payload.get("selection_uses_legacy_metrics") is not False:
        raise ValueError("selected_profiles must not use legacy metrics")
    return payload


__all__ = [
    "LAYERS", "SYMBOLS_PER_LAYER", "TOTAL_DATA_RE", "UEPCandidate",
    "build_selected_profiles_artifact", "candidate_hash", "enumerate_feasible_repetitions",
    "make_candidate", "objective_from_summary", "pareto_front", "per_re_power",
    "propose_power_transfers", "propose_repetition_moves", "sample_random_candidates",
    "select_profiles", "softmax_power", "total_energy", "total_re", "validate_repetition",
    "validate_search_split", "load_selected_profiles", "rank_by_objective",
]
