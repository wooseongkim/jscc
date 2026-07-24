from __future__ import annotations

import math
import hashlib
import statistics
from collections import defaultdict
from dataclasses import replace
from typing import Any, Iterable

import torch

from evaluation.paired import PairedEvaluationBatch


J4_TAIL_VERSION = "j4_tail_failure_v1"
ROOT_CAUSE_THRESHOLDS = {
    "relative_negative_rate_reduction": 0.50,
    "absolute_p10_improvement": 0.02,
    "minimum_confirmed_negative_rate": 0.01,
    "mean_degradation_tolerance": 0.02,
}


def select_strongest_condition(distribution: dict[str, Any]) -> dict[str, float]:
    """Select the observed weakest boundary row, not the widest burst."""
    row = distribution.get("worst_tail_condition")
    if not isinstance(row, dict):
        fractions = distribution.get("selected_burst_fractions") or []
        snrs = distribution.get("selected_snr_range_db") or []
        jsrs = distribution.get("selected_global_jsr_range_db") or []
        if not fractions or not snrs or not jsrs:
            raise ValueError("selected distribution lacks boundary evidence and severity ranges")
        return {"snr_db":float(min(snrs)),"jsr_db":float(max(jsrs)),
                "burst_fraction":float(min(fractions))}
    fraction = row.get("requested_fraction", row.get("burst_fraction"))
    if fraction is None:
        raise ValueError("worst_tail_condition lacks burst fraction")
    return {"snr_db": float(row["snr_db"]), "jsr_db": float(row["jsr_db"]),
            "burst_fraction": float(fraction)}


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval (normal z=1.9599639845 at 95%)."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("successes/total are invalid")
    if confidence != 0.95:
        raise ValueError("only the documented 95% Wilson interval is supported")
    z = 1.959963984540054
    p = successes / total; z2 = z * z; denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _rate(values: list[float], predicate) -> dict[str, Any]:
    count = sum(bool(predicate(value)) for value in values); low, high = wilson_interval(count, len(values))
    return {"count": count, "total": len(values), "rate": count / len(values),
            "confidence_interval_95": [low, high], "method": "Wilson score interval"}


def summarize_failure_rates(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    if not rows: raise ValueError("failure-rate summary requires rows")
    realization = [float(row["layer7_improvement"]) for row in rows]
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows: grouped[str(row["utterance_id"])].append(float(row["layer7_improvement"]))
    # A content item fails if any paired stochastic realization fails. This does
    # not pretend repeated realizations are independent utterances.
    utterance = [min(values) for values in grouped.values()]
    def summary(values):
        return {"negative": _rate(values, lambda x: x < 0),
                "below_2_percent": _rate(values, lambda x: x < .02),
                "below_5_percent": _rate(values, lambda x: x < .05)}
    return {"realization": summary(realization), "utterance": summary(utterance)}


def data_only_batch(batch: PairedEvaluationBatch) -> PairedEvaluationBatch:
    """Suppress pilot jammer samples and retain the original global jammer power."""
    mask = batch.jammer_mask & ~batch.pilot_mask
    jammer = batch.jammer * mask
    original = batch.jammer.abs().square().mean(tuple(range(1, batch.jammer.ndim)), keepdim=True)
    current = jammer.abs().square().mean(tuple(range(1, jammer.ndim)), keepdim=True).clamp_min(1e-12)
    jammer = jammer * torch.sqrt(original / current)
    return replace(batch, jammer=jammer, jammer_mask=mask,
                   metadata={**batch.metadata, "diagnostic_mode": "data_only_burst"})


def no_jammer_batch(batch: PairedEvaluationBatch) -> PairedEvaluationBatch:
    return replace(batch, jammer=torch.zeros_like(batch.jammer),
                   jammer_mask=torch.zeros_like(batch.jammer_mask),
                   metadata={**batch.metadata, "diagnostic_mode": "no_jammer_control"})


def pair_stochastic_environment(
    baseline: PairedEvaluationBatch, jammer_variant: PairedEvaluationBatch
) -> PairedEvaluationBatch:
    """Use a variant jammer while retaining every non-jammer paired tensor."""
    return replace(jammer_variant, seed=baseline.seed, waveform=baseline.waveform,
        representation=baseline.representation, snr_db=baseline.snr_db,
        pilot_mask=baseline.pilot_mask, pilots=baseline.pilots, noise=baseline.noise,
        signal_fading=baseline.signal_fading, jammer_fading=baseline.jammer_fading,
        metadata={**jammer_variant.metadata, "paired_environment_seed":baseline.seed})


def paired_burst_barrage_batches(batch: PairedEvaluationBatch, *, seed: int):
    """Build burst and barrage interventions from one canonical Gaussian field."""
    generator=torch.Generator(device=batch.jammer.device).manual_seed(int(seed))
    real=torch.randn(batch.jammer.shape,device=batch.jammer.device,dtype=batch.jammer.real.dtype,generator=generator)
    imag=torch.randn(batch.jammer.shape,device=batch.jammer.device,dtype=batch.jammer.real.dtype,generator=generator)
    raw=torch.complex(real,imag)/math.sqrt(2.0)
    dims=tuple(range(1,raw.ndim));global_jsr=float(batch.jsr_db.mean());fraction=float(batch.jammer_mask.float().mean())
    def scaled(mask,jsr):
        value=raw*mask;power=value.abs().square().mean(dims,keepdim=True).clamp_min(1e-12)
        target=batch.target_power*(10**(jsr/10));return value*torch.sqrt(value.real.new_tensor(target)/power)
    burst_jammer=scaled(batch.jammer_mask,global_jsr);full=torch.ones_like(batch.jammer_mask)
    equal_global=scaled(full,global_jsr);active_jsr=global_jsr-10*math.log10(fraction);equal_active=scaled(full,active_jsr)
    digest=hashlib.sha256(raw.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    common={**batch.metadata,"canonical_jammer_source_hash":digest}
    return (replace(batch,jammer=burst_jammer,metadata={**common,"diagnostic_mode":"burst_baseline"}),
            replace(batch,jammer_type="barrage",jammer=equal_global,jammer_mask=full,
                    metadata={**common,"diagnostic_mode":"barrage_equal_global"}),
            replace(batch,jammer_type="barrage",jammer=equal_active,jammer_mask=full,
                    jsr_db=torch.full_like(batch.jsr_db,active_jsr),metadata={**common,"diagnostic_mode":"barrage_equal_active"}),digest)


def _effect(baseline: dict[str, float], candidate: dict[str, float], thresholds: dict[str, float]) -> dict[str, Any]:
    base_rate = float(baseline["negative_rate"]); candidate_rate = float(candidate["negative_rate"])
    relative = (base_rate - candidate_rate) / max(base_rate, 1e-12)
    p10_gain = float(candidate["p10"]) - float(baseline["p10"])
    return {"relative_negative_rate_reduction": relative, "p10_improvement": p10_gain,
            "mean_change": float(candidate.get("mean", 0)) - float(baseline.get("mean", 0)),
            "substantial": relative >= thresholds["relative_negative_rate_reduction"] or
                           p10_gain >= thresholds["absolute_p10_improvement"]}


def classify_root_cause(baseline: dict[str, float], *, oracle: dict[str, float],
                        data_only: dict[str, float], clipped: dict[int, dict[str, float]],
                        oracle_clipped: dict[int, dict[str, float]],
                        equal_global_barrage: dict[str, float],
                        thresholds: dict[str, float] = ROOT_CAUSE_THRESHOLDS) -> dict[str, Any]:
    evidence = {"oracle": _effect(baseline, oracle, thresholds),
                "data_only": _effect(baseline, data_only, thresholds),
                "clipped": {str(k): _effect(baseline, v, thresholds) for k, v in clipped.items()},
                "oracle_clipped": {str(k): _effect(baseline, v, thresholds) for k, v in oracle_clipped.items()},
                "equal_global_barrage": _effect(baseline, equal_global_barrage, thresholds)}
    if float(baseline["negative_rate"]) < thresholds["minimum_confirmed_negative_rate"]:
        label = "INSUFFICIENT_SAMPLE_EVIDENCE"
    else:
        oracle_help = evidence["oracle"]["substantial"] or evidence["data_only"]["substantial"]
        clip_help = any(v["substantial"] and v["mean_change"] >= -thresholds["mean_degradation_tolerance"] for v in evidence["clipped"].values())
        combined = any(v["substantial"] for v in evidence["oracle_clipped"].values())
        burst_worse = evidence["equal_global_barrage"]["substantial"]
        if combined and not oracle_help and not clip_help: label = "COMBINED_CSI_EQUALIZER"
        elif oracle_help: label = "PILOT_CSI_DOMINANT"
        elif clip_help: label = "EQUALIZER_AMPLIFICATION_DOMINANT"
        elif burst_worse: label = "TEMPORAL_CONCENTRATION_DOMINANT"
        else: label = "REPRESENTATION_LAYER7_DOMINANT"
    return {"classification": label, "thresholds": dict(thresholds), "evidence": evidence}


def distribution(values: Iterable[float]) -> dict[str, float]:
    values = sorted(map(float, values)); n = len(values)
    if not n: raise ValueError("distribution requires values")
    def percentile(q):
        p=(n-1)*q; lo=math.floor(p); hi=math.ceil(p)
        return values[lo] if lo==hi else values[lo]*(hi-p)+values[hi]*(p-lo)
    return {"mean": statistics.fmean(values), "std": statistics.pstdev(values),
            "median": statistics.median(values), "p10": percentile(.10),
            "p05": percentile(.05), "minimum": values[0],
            "negative_rate": sum(x < 0 for x in values)/n,
            "below_2_percent_rate": sum(x < .02 for x in values)/n,
            "below_5_percent_rate": sum(x < .05 for x in values)/n}
