from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def classify_overfit_result(stage: str, result: dict[str, float]) -> tuple[bool, list[str]]:
    """Apply data-path learnability thresholds; these are not performance targets."""
    loss = float(result["final_loss"])
    improvement = float(result["relative_improvement_over_zero"])
    if stage in {"O0", "O1", "O2", "O2-P"}:
        loss_pass = improvement >= 0.80 or loss < 0.20
        loss_reason = "requires >=80% zero improvement or loss <0.2"
    elif stage in {"O3", "O4", "O5"}:
        loss_pass = improvement >= 0.50 or loss < 0.50
        loss_reason = "requires >=50% zero improvement or loss <0.5"
    elif stage in {"O6", "O7"}:
        loss_pass = improvement >= 0.05
        loss_reason = "requires >=5% zero improvement"
    else:
        raise ValueError(f"unknown overfit stage: {stage}")
    guards = {
        "power ratio >=0.01": float(result["power_ratio"]) >= 0.01,
        "cosine >0": float(result["cosine_similarity"]) > 0.0,
        "correlation >0": float(result["pearson_correlation"]) > 0.0,
    }
    reasons = ([] if loss_pass else [loss_reason]) + [name for name, passed in guards.items() if not passed]
    return loss_pass and all(guards.values()), reasons


def stages_to_run(
    requested: Sequence[str], prior_results: Sequence[dict[str, Any]], continue_after_failure: bool
) -> tuple[str, ...]:
    completed = {str(item["stage"]) for item in prior_results}
    remaining = tuple(stage for stage in requested if stage not in completed)
    if prior_results and not bool(prior_results[-1]["passed"]) and not continue_after_failure:
        return ()
    return remaining
