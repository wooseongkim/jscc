"""Discovery, paired summaries, constraints, and ranking for expanded selection.

No legacy-final input is accepted by this module.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import torch


_STEP = re.compile(r"local_step_(\d+)\.pt$")
_PREFERENCE = {"best_5db_si_sdr.pt": 2, "best_validation_average.pt": 3, "last.pt": 4}


def model_state_hash(model: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(model):
        value = model[key].detach().cpu().contiguous()
        digest.update(key.encode()); digest.update(str(value.dtype).encode()); digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_step(payload: dict, name: str):
    for key in ("local_step", "step"):
        if payload.get(key) is not None:
            return int(payload[key])
    match = _STEP.match(name)
    return int(match.group(1)) if match else None


def discover_checkpoints(root: Path, experiments: list[str], weights: dict[str, float]):
    records = []
    for experiment in experiments:
        directory = root / experiment
        for path in sorted(directory.glob("*.pt")) if directory.is_dir() else []:
            item = {"checkpoint_id": f"{experiment}:{path.name}", "experiment": experiment, "checkpoint_path": str(path), "filename": path.name, "file_size": path.stat().st_size, "target_si_sdr_weight": float(weights.get(experiment, 0.0)), "loadable": False, "failure_reason": None, "aliases": []}
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
                    raise ValueError("checkpoint has no model state")
                item.update({"loadable": True, "sha256": _sha(path), "model_state_hash": model_state_hash(payload["model"]), "local_step": _local_step(payload, path.name), "source_global_step": payload.get("source_global_step", 5750), "checkpoint_metadata": {key: payload.get(key) for key in ("global_step", "local_step", "resume_mode") if key in payload}})
            except Exception as error:
                item["failure_reason"] = str(error)
            records.append(item)
    groups = {}
    for item in records:
        if item["loadable"]:
            groups.setdefault((item["experiment"], item["model_state_hash"]), []).append(item)
    representatives = []
    for values in groups.values():
        def key(item):
            return (0 if _STEP.match(item["filename"]) else _PREFERENCE.get(item["filename"], 9), -(item["local_step"] if item["local_step"] is not None else -1), item["filename"])
        values.sort(key=key)
        representative = dict(values[0]); representative["aliases"] = sorted(value["filename"] for value in values[1:])
        representatives.append(representative)
    return records, representatives


def paired_statistics(values, *, samples: int, seed: int):
    data = np.asarray(values, dtype=float)
    if not np.isfinite(data).all():
        raise FloatingPointError("nonfinite paired delta")
    rng = np.random.default_rng(seed)
    means = [float(data[rng.integers(0, len(data), len(data))].mean()) for _ in range(samples)] if len(data) else []
    qs = lambda q: float(np.percentile(data, q)) if len(data) else None
    return {"count": int(len(data)), "mean": float(data.mean()) if len(data) else None, "median": float(np.median(data)) if len(data) else None, "std": float(data.std(ddof=1)) if len(data) > 1 else 0.0, "standard_error": float(data.std(ddof=1)/math.sqrt(len(data))) if len(data)>1 else 0.0, "p1": qs(1), "p5": qs(5), "p10": qs(10), "p25": qs(25), "p75": qs(75), "p90": qs(90), "p95": qs(95), "minimum": float(data.min()) if len(data) else None, "maximum": float(data.max()) if len(data) else None, "positive_gain_fraction": float((data>0).mean()) if len(data) else None, "zero_gain_fraction": float((data==0).mean()) if len(data) else None, "negative_gain_fraction": float((data<0).mean()) if len(data) else None, "bootstrap_ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means,97.5))] if means else [None,None], "insufficient_for_inference": len(data)<5, "catastrophic_fractions": {str(v): float((data < v).mean()) for v in (-1.,-3.,-5.,-10.)}}


def statistical_tests(groups: dict[str, list[float]]):
    """Wilcoxon/t-test with deterministic Holm correction; no ranking input."""
    from scipy import stats
    result = {}; eligible=[]
    for name, values in groups.items():
        data=np.asarray(values,dtype=float)
        if len(data)<5: result[name]={"status":"insufficient_data","wilcoxon_p_value":None,"paired_t_p_value":None}; continue
        if np.allclose(data, data[0]): result[name]={"status":"degenerate","wilcoxon_p_value":None,"paired_t_p_value":None}; continue
        try: p=float(stats.wilcoxon(data).pvalue)
        except ValueError: p=None
        t=float(stats.ttest_1samp(data,0.0).pvalue)
        result[name]={"status":"ok","wilcoxon_p_value":p,"paired_t_p_value":t};
        if p is not None: eligible.append((name,p))
    m=len(eligible)
    for rank,(name,p) in enumerate(sorted(eligible,key=lambda x:x[1])):
        result[name]["holm_corrected_p_value"]=min(1.0,p*(m-rank))
    for value in result.values(): value.setdefault("holm_corrected_p_value",None)
    return result


def smoke_constraints(*, rows):
    valid = bool(rows) and all(row.get("finite") for row in rows) and all(row.get("condition_hash") and row.get("mapping_hash") for row in rows)
    return {"mode": "smoke", "pass": valid, "constraints": [{"name":"load_finite_complete_paired", "applicable":True, "observed_value":valid, "reference_value":None, "threshold":True, "pass":valid, "failure_reason":None if valid else "missing/nonfinite/unpaired row"}, {"name":"full_only_regression_constraints", "applicable":False, "observed_value":None, "reference_value":None, "threshold":None, "pass":None, "failure_reason":"not_evaluated_in_smoke"}]}


def full_constraints(*, rows, by_snr, catastrophic_3_max: float, catastrophic_5_max: float):
    base=smoke_constraints(rows=rows); checks=list(base["constraints"][:1])
    five=by_snr["5.0"]; ten=by_snr["10.0"]; fifteen=by_snr["15.0"]
    values=[("5db_waveform_snr_delta", five["waveform_snr_mean"]-five["source_waveform_snr_mean"], -0.2, "gte"), ("5db_stft_ratio", five["stft_mean"]/max(five["source_stft_mean"],1e-12), 1.05, "lte"), ("10db_si_sdr_delta",ten["statistics"]["mean"],-0.3,"gte"), ("15db_si_sdr_delta",fifteen["statistics"]["mean"],-0.3,"gte"), ("5db_catastrophic_lt_minus_3",five["statistics"]["catastrophic_fractions"]["-3.0"],catastrophic_3_max,"lte"), ("5db_catastrophic_lt_minus_5",five["statistics"]["catastrophic_fractions"]["-5.0"],catastrophic_5_max,"lte")]
    for name, observed, threshold, direction in values:
        passed=observed>=threshold if direction=="gte" else observed<=threshold
        checks.append({"name":name,"applicable":True,"observed_value":observed,"reference_value":0.0,"threshold":threshold,"pass":passed,"failure_reason":None if passed else f"{observed} violates {direction} {threshold}"})
    return {"mode":"full_selection","pass":all(c["pass"] for c in checks if c["pass"] is not None),"constraints":checks}


def rank_experiments(candidates, *, tolerance: float):
    results = {}
    for experiment in {candidate["experiment"] for candidate in candidates}:
        members = [item for item in candidates if item["experiment"] == experiment and item.get("constraint_pass")]
        if not members:
            results[experiment] = {"status":"no_passing_candidate", "selected_checkpoint":None}; continue
        def key(item):
            metric = item["metrics"]
            return (-metric.get("si_sdr_5db", -math.inf), -metric.get("delta_vs_source_5db", -math.inf), -metric.get("average_si_sdr", -math.inf), -metric.get("positive_gain_fraction_5db", -math.inf), -metric.get("p5_delta", -math.inf), metric.get("catastrophic_lt_minus_3", math.inf), metric.get("catastrophic_lt_minus_5", math.inf), metric.get("stft_ratio", math.inf), -metric.get("waveform_snr", -math.inf), item.get("local_step") if item.get("local_step") is not None else math.inf)
        selected = sorted(members, key=key)[0]
        results[experiment] = {"status":"selected", **selected}
    return results


def write_selection_artifact(path: Path, *, source: dict, selected: dict, constraint_mode: str="smoke"):
    artifact = {"schema_version":1, "artifact_type":"expanded_selection_checkpoint_selection", "selection_split":"expanded_selection", "selection_uses_legacy_metrics":False, "constraint_mode":constraint_mode, "selection_metric":"5db_mean_si_sdr", "source_checkpoint":source, "experiments":selected, "overall_best_si_sdr_experiment":None, "created_from":{"checkpoint_ranking":"checkpoint_ranking.json", "checkpoint_constraints":"checkpoint_constraints.json", "paired_statistics":"paired_statistics.json"}}
    choices = [value for value in selected.values() if value.get("status")=="selected" and value.get("experiment") != "control_no_si_sdr"]
    if choices:
        best = max(choices, key=lambda item: item["metrics"]["si_sdr_5db"]); artifact["overall_best_si_sdr_experiment"]={"experiment":best["experiment"],"checkpoint":best["checkpoint_path"]}
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True)+"\n")
    return artifact
