from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from speech_jscc.diagnostics.random_distribution import SeedDeriver


J2_VERSION = "j2_strong_barrage_boundary_v1"
J2_SNR_GRID = (0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0)
J2_JSR_GRID = (-15.0, -12.5, -10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0)
J2_THRESHOLDS = {
    "aggregate_improvement": .10, "aggregate_correlation": .30,
    "enhancement_improvement": .075, "enhancement_correlation": .25,
    "layers6_to_7_improvement": .05, "layer7_improvement": .03,
    "aggregate_power_ratio": .12, "enhancement_power_ratio": .08,
    "layer7_power_ratio": .04, "worst_aggregate_improvement": .075,
    "worst_enhancement_improvement": .05, "worst_layer7_improvement": .02,
}


def file_sha256(path: str | Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def verify_j1_artifact(summary_path: str | Path, checkpoint_path: str | Path, *, expected: dict[str, Any] | None=None) -> dict[str, Any]:
    summary_path=Path(summary_path); checkpoint_path=Path(checkpoint_path)
    if not summary_path.is_file() or not checkpoint_path.is_file(): raise ValueError("accepted J1 summary and checkpoint are required")
    payload=json.loads(summary_path.read_text()); provenance=payload.get("provenance",{})
    if not payload.get("gate",{}).get("stage_pass"): raise ValueError("accepted J1 gate is not PASS")
    if provenance.get("stage_name")!="j1_weak_random_barrage" or provenance.get("model_architecture")!="conv_conformer_v1":
        raise ValueError("accepted J1 stage or architecture mismatch")
    preprocessing=provenance.get("preprocessing",{})
    preprocessing_hash=preprocessing.get("preprocessing_hash") or hashlib.sha256(json.dumps(preprocessing,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    result={"summary_path":str(summary_path),"checkpoint_path":str(checkpoint_path),
        "summary_sha256":file_sha256(summary_path),"checkpoint_sha256":file_sha256(checkpoint_path),
        "stage_name":provenance.get("stage_name"),"architecture":provenance.get("model_architecture"),
        "architecture_version":provenance.get("architecture_version"),"subset_size":provenance.get("subset_size"),
        "preprocessing_hash":preprocessing_hash,
        "latent_cache_hash":provenance.get("latent_cache_hash"),"train_manifest_hash":provenance.get("train_manifest_hash"),
        "validation_suite_hash":provenance.get("validation_suite_hash"),"git_commit":provenance.get("git_commit")}
    if expected:
        for key in ("summary_sha256","checkpoint_sha256"):
            if expected.get(key)!=result[key]: raise ValueError(f"J1 artifact hash changed: {key}")
    return result


def derive_sweep_seed(root_seed: int, snr_db: float, jsr_db: float, realization: int, group: str="all") -> int:
    return SeedDeriver(root_seed).seed(f"j2_sweep|{group}|snr={snr_db:g}|jsr={jsr_db:g}",realization)


def sweep_grid(*,snr_values: Iterable[float]=J2_SNR_GRID,jsr_values: Iterable[float]=J2_JSR_GRID,realizations: int=16) -> list[dict[str,Any]]:
    if realizations<=0: raise ValueError("realizations must be positive")
    return [{"snr_db":float(snr),"jsr_db":float(jsr),"realization":index}
        for snr in snr_values for jsr in jsr_values for index in range(realizations)]


def _percentile(values: list[float], fraction: float) -> float:
    ordered=sorted(values); position=(len(ordered)-1)*fraction; low=math.floor(position); high=math.ceil(position)
    return ordered[low] if low==high else ordered[low]*(high-position)+ordered[high]*(position-low)


def aggregate_realizations(rows: list[dict[str,Any]], metric_names: Iterable[str]) -> list[dict[str,Any]]:
    grouped=defaultdict(list)
    for row in rows: grouped[(float(row["snr_db"]),float(row["jsr_db"]),str(row.get("group","all")))].append(row)
    output=[]
    lower_is_worse_tokens=("improvement","correlation","cosine","power_ratio","sinr")
    for (snr,jsr,group),members in sorted(grouped.items()):
        item={"snr_db":snr,"jsr_db":jsr,"group":group,"realization_count":len(members)}
        for name in metric_names:
            values=[float(row[name]) for row in members if row.get(name) is not None]
            if not values: continue
            lower_is_worse=any(token in name for token in lower_is_worse_tokens)
            item[f"{name}_mean"]=statistics.fmean(values)
            item[f"{name}_std"]=statistics.pstdev(values)
            item[f"{name}_worst_decile"]=_percentile(values,.10 if lower_is_worse else .90)
            worst=min(values) if lower_is_worse else max(values)
            item[f"{name}_worst_sample"]=worst
        output.append(item)
    return output


def select_training_range(aggregated_rows: list[dict[str,Any]]) -> dict[str,Any]:
    candidates=sorted({float(row["jsr_db"]) for row in aggregated_rows if float(row["jsr_db"])>=-10.0})
    failing=None
    for jsr in candidates:
        rows=[row for row in aggregated_rows if float(row["jsr_db"])==jsr]
        enhancement=min((float(row.get("layers1_to_7_relative_improvement_over_zero_mean",1.0)) for row in rows),default=1.0)
        aggregate=min((float(row.get("aggregate_relative_improvement_over_zero_mean",1.0)) for row in rows),default=1.0)
        if enhancement<.05 and aggregate>0:
            failing=(jsr,rows);break
    if failing is None:
        return {"defined":False,"reason":"no enhancement-layer transition with noncollapsed aggregate reconstruction","selected_snr_range_db":None,"selected_jsr_range_db":None}
    upper,rows=failing
    layers=[(layer,min(float(row.get(f"layer{layer}_relative_improvement_over_zero_mean",1.0)) for row in rows)) for layer in range(1,8)]
    first_layer=min(layers,key=lambda pair:pair[1])[0]
    csi=max(float(row.get("csi_nmse_mean",0)) for row in rows); gain=max(float(row.get("maximum_equalizer_gain_mean",0)) for row in rows)
    return {"defined":True,"selected_snr_range_db":[5.0,15.0],"selected_jsr_range_db":[-10.0,float(upper)],
        "lower_boundary_evidence":"accepted J1 strongest requested JSR -10 dB passed",
        "upper_boundary_evidence":{"jsr_db":float(upper),"rows":rows},
        "first_failing_metric":"enhancement_layers_improvement","first_failing_layer":first_layer,
        "failure_appearance":"channel_driven" if csi>.25 or gain>100 else "model_or_mixed"}


def summarize_layer_groups(metrics: dict[str,Any]) -> dict[str,Any]:
    layers=metrics["per_layer"]
    def mean(indices):
        keys=("normalized_mse","raw_mse","relative_improvement_over_zero","cosine_similarity","pearson_correlation","power_ratio","optimal_scalar_rescaled_normalized_loss")
        return {key:statistics.fmean(float(layers[i][key]) for i in indices) for key in keys if key in layers[indices[0]]}
    return {"aggregate":metrics["aggregate"],"layer0":dict(layers[0]),"layers1_to_7":mean(range(1,8)),
        "layers6_to_7":mean((6,7)),"layer7":dict(layers[7])}


def j2_gate(unseen: dict[str,Any], strongest: dict[str,Any], *, randomness_pass: bool, coverage_pass: bool, parameters_finite: bool, thresholds: dict[str,float]=J2_THRESHOLDS) -> dict[str,Any]:
    a=unseen["aggregate"]; e=unseen["layers1_to_7"]; deep=unseen["layers6_to_7"]; l7=unseen["layer7"]; w_a=strongest["aggregate"]; w_e=strongest["layers1_to_7"]; w_l7=strongest["layer7"]
    checks={
        "aggregate_pass":a["relative_improvement_over_zero"]>=thresholds["aggregate_improvement"] and a["pearson_correlation"]>=thresholds["aggregate_correlation"],
        "enhancement_pass":e["relative_improvement_over_zero"]>=thresholds["enhancement_improvement"] and e["pearson_correlation"]>=thresholds["enhancement_correlation"],
        "deep_enhancement_pass":deep["relative_improvement_over_zero"]>=thresholds["layers6_to_7_improvement"] and l7["relative_improvement_over_zero"]>=thresholds["layer7_improvement"] and l7["pearson_correlation"]>0,
        "power_pass":a["power_ratio"]>=thresholds["aggregate_power_ratio"] and e["power_ratio"]>=thresholds["enhancement_power_ratio"] and l7["power_ratio"]>=thresholds["layer7_power_ratio"],
        "worst_condition_pass":w_a["relative_improvement_over_zero"]>=thresholds["worst_aggregate_improvement"] and w_e["relative_improvement_over_zero"]>=thresholds["worst_enhancement_improvement"] and w_l7["relative_improvement_over_zero"]>=thresholds["worst_layer7_improvement"] and min(w_a["pearson_correlation"],w_e["pearson_correlation"],w_l7["pearson_correlation"])>0,
        "finite_pass":bool(parameters_finite and a.get("finite",True) and strongest.get("channel",{}).get("finite",True)),
        "randomness_pass":bool(randomness_pass),"coverage_pass":bool(coverage_pass),
    }
    checks["channel_estimation_stable"]=float(strongest.get("channel",{}).get("csi_nmse",0))<.5
    checks["equalizer_stable"]=float(strongest.get("channel",{}).get("maximum_equalizer_gain",0))<1000
    checks["clearly_violated"]=a["relative_improvement_over_zero"]<0 or e["relative_improvement_over_zero"]<0 or l7["pearson_correlation"]<=0
    checks["passed"]=all(value for key,value in checks.items() if key not in {"clearly_violated"})
    checks["thresholds"]=dict(thresholds)
    return checks


def classify_j2(gate: dict[str,Any], *, final_is_best: bool, loss_slope: float) -> str:
    if gate["passed"]: return "PASS"
    if not gate["finite_pass"]: return "FAIL_NONFINITE"
    if final_is_best and loss_slope<0 and not gate.get("clearly_violated",False): return "INCONCLUSIVE_NOT_CONVERGED"
    if not gate.get("channel_estimation_stable",True): return "FAIL_CHANNEL_ESTIMATION"
    if not gate.get("equalizer_stable",True): return "FAIL_EQUALIZER_INSTABILITY"
    if not gate.get("enhancement_pass",True) or not gate.get("deep_enhancement_pass",True): return "FAIL_ENHANCEMENT_COLLAPSE"
    return "FAIL_MODEL_RECONSTRUCTION"


def select_initialization(fresh: dict[str,Any], transfer: dict[str,Any]) -> dict[str,Any]:
    required=("unseen_loss","layers1_to_7_improvement","layers6_to_7_improvement","layer7_improvement","gradient_finite_ratio","output_power_ratio")
    if any(key not in row for row in (fresh,transfer) for key in required): raise ValueError("initialization summaries are incomplete")
    if min(float(fresh["gradient_finite_ratio"]),float(transfer["gradient_finite_ratio"]))<1: raise ValueError("nonfinite initialization comparison")
    def score(row):
        return (-float(row["unseen_loss"])+float(row["layers1_to_7_improvement"])+
            float(row["layers6_to_7_improvement"])+float(row["layer7_improvement"])+.1*float(row["output_power_ratio"]))
    selected="j1_transfer" if score(transfer)>score(fresh) else "fresh"
    return {"selected_initialization":selected,"fresh":fresh,"j1_transfer":transfer,
        "decision_rule":"lower unseen loss plus enhancement/deep-layer improvement and output-power evidence",
        "score":{"fresh":score(fresh),"j1_transfer":score(transfer)}}


def validate_j2_resume_metadata(saved: dict[str,Any],current: dict[str,Any]) -> None:
    for key in ("stage","initialization_mode","selected_range_hash"):
        if saved.get(key)!=current.get(key): raise ValueError(f"J2 resume mismatch: {key}")
