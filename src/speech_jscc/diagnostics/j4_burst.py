from __future__ import annotations
import json,math,random,statistics
from pathlib import Path
from typing import Any
import torch
from speech_jscc.diagnostics.j2_barrage import file_sha256
from speech_jscc.diagnostics.random_distribution import SeedDeriver

J4_VERSION="j4_random_burst_v1";J4_SNR_GRID=(5.,10.,15.);J4_JSR_GRID=(-10.,-5.,0.);J4_FRACTIONS=(.125,.25,.5)
J4_THRESHOLDS={"aggregate_improvement":.10,"aggregate_correlation":.30,"aggregate_power_ratio":.12,"enhancement_improvement":.075,"enhancement_correlation":.25,"enhancement_power_ratio":.08,"layers6_to_7_improvement":.05,"layer7_improvement":.03,"layer7_power_ratio":.04,"worst_aggregate_improvement":.075,"worst_enhancement_improvement":.05,"worst_layer7_improvement":.02,"layer7_p10_min":0.,"layer7_negative_rate_max":.10}

def active_window_jsr_db(global_jsr_db,fraction):
    if not 0<float(fraction)<=1:raise ValueError("fraction must be in (0,1]")
    return float(global_jsr_db)-10*math.log10(float(fraction))

def burst_diagnostics(signal,jammer,mask,pilot_mask,*,requested_fraction,requested_global_jsr_db,faded_signal=None,faded_jammer=None):
    active=mask.any(dim=1);counts=active.sum(dim=1);indices=torch.where(active[0])[0];actual=float(counts.float().mean()/mask.shape[2])
    contiguous=all(bool(torch.equal(torch.where(row)[0],torch.arange(torch.where(row)[0][0],torch.where(row)[0][0]+row.sum(),device=row.device))) for row in active)
    full_band=all(bool(item[:,torch.where(item.any(dim=0))[0]].all()) for item in mask)
    outside=jammer[~mask].abs().square().mean() if (~mask).any() else jammer.real.new_tensor(0.);inside=jammer[mask].abs().square().mean();dims=(1,2)
    tx=jammer.abs().square().mean(dims)/signal.abs().square().mean(dims).clamp_min(1e-12);fs=signal if faded_signal is None else faded_signal;fj=jammer if faded_jammer is None else faded_jammer
    received=fj.abs().square().mean(dims)/fs.abs().square().mean(dims).clamp_min(1e-12);local=[fj[i][mask[i]].abs().square().mean()/fs[i][mask[i]].abs().square().mean().clamp_min(1e-12) for i in range(mask.shape[0])]
    data=~pilot_mask;packed=[]
    for i in range(mask.shape[0]):packed.append((mask[i]&data[i])[data[i]].reshape(8,-1).float())
    chunks=torch.stack(packed);layer_counts=chunks.sum(2).mean(0);layer_fraction=chunks.mean(2).mean(0)
    return {"requested_global_jsr_db":float(requested_global_jsr_db),"measured_transmit_global_jsr_db":float((10*torch.log10(tx)).detach().mean()),"realized_received_global_jsr_db":float((10*torch.log10(received)).detach().mean()),"realized_received_active_window_jsr_db":float((10*torch.log10(torch.stack(local))).detach().mean()),"active_window_jsr_db":active_window_jsr_db(requested_global_jsr_db,actual),"requested_burst_fraction":float(requested_fraction),"actual_burst_fraction":actual,"burst_symbol_count":int(counts[0]),"burst_start_symbol":int(indices[0]),"burst_end_symbol":int(indices[-1]),"burst_symbol_indices":[int(x) for x in indices],"contiguous_burst_verified":contiguous,"full_band_inside_burst_verified":full_band,"wraparound_used":False,"per_layer_burst_data_count":[float(x) for x in layer_counts],"per_layer_burst_data_fraction":[float(x) for x in layer_fraction],"pilot_overlap_fraction":float(((mask&pilot_mask).sum()/pilot_mask.sum().clamp_min(1)).detach()),"data_overlap_fraction":float(((mask&data).sum()/data.sum().clamp_min(1)).detach()),"jammer_power_inside_burst":float(inside.detach()),"leakage_power_outside_burst":float(outside.detach())}

def j4_policy(root_seed,step,snr_range,jsr_range,fractions):
    d=SeedDeriver(root_seed);rng=random.Random(d.seed("j4_distribution",step));values=list(map(float,fractions))
    return {"seed":d.seed("j4_channel_jammer_noise",step),"snr_db":rng.uniform(*map(float,snr_range)),"jsr_db":rng.uniform(*map(float,jsr_range)),"burst_fraction":values[rng.randrange(len(values))],"jammer_type":"burst"}

def _percentile(values,q):
    s=sorted(map(float,values));p=(len(s)-1)*q;lo=math.floor(p);hi=math.ceil(p);return s[lo] if lo==hi else s[lo]*(hi-p)+s[hi]*(p-lo)
def tail_statistics(values):
    v=list(map(float,values));return {"mean":statistics.fmean(v),"std":statistics.pstdev(v),"median":statistics.median(v),"p10":_percentile(v,.1),"p05":_percentile(v,.05),"worst":min(v),"below_5_percent_rate":sum(x<.05 for x in v)/len(v),"below_2_percent_rate":sum(x<.02 for x in v)/len(v),"negative_rate":sum(x<0 for x in v)/len(v),"nonpositive_rate":sum(x<=0 for x in v)/len(v)}

def verify_j3_accepted(manifest_path,checkpoint_path):
    value=json.loads(Path(manifest_path).read_text())
    if value.get("classification")!="ACCEPTED_PASS" or not value.get("initial_weights_loaded") or value.get("checkpoint_sha256")!=file_sha256(checkpoint_path):raise ValueError("corrected accepted J3 artifact required")
    return {**value,"accepted_manifest_path":str(manifest_path),"accepted_manifest_sha256":file_sha256(manifest_path)}

def j4_gate(unseen,strongest,*,infrastructure,tail,thresholds=J4_THRESHOLDS):
    a=unseen["aggregate"];e=unseen["layers1_to_7"];d=unseen["layers6_to_7"];l=unseen["layer7"];wa=strongest["aggregate"];we=strongest["layers1_to_7"];wl=strongest["layer7"]
    g={"aggregate_pass":a["relative_improvement_over_zero"]>=thresholds["aggregate_improvement"] and a["pearson_correlation"]>=thresholds["aggregate_correlation"] and a["power_ratio"]>=thresholds["aggregate_power_ratio"],"enhancement_pass":e["relative_improvement_over_zero"]>=thresholds["enhancement_improvement"] and e["pearson_correlation"]>=thresholds["enhancement_correlation"] and e["power_ratio"]>=thresholds["enhancement_power_ratio"],"deep_pass":d["relative_improvement_over_zero"]>=thresholds["layers6_to_7_improvement"],"layer7_pass":l["relative_improvement_over_zero"]>=thresholds["layer7_improvement"] and l["pearson_correlation"]>0 and l["power_ratio"]>=thresholds["layer7_power_ratio"],"worst_pass":wa["relative_improvement_over_zero"]>=thresholds["worst_aggregate_improvement"] and we["relative_improvement_over_zero"]>=thresholds["worst_enhancement_improvement"] and wl["relative_improvement_over_zero"]>=thresholds["worst_layer7_improvement"] and min(wa["pearson_correlation"],we["pearson_correlation"],wl["pearson_correlation"])>0,"tail_pass":tail["layer7_improvement_p10"]>=thresholds["layer7_p10_min"] and tail["layer7_negative_rate"]<=thresholds["layer7_negative_rate_max"],"channel_stable":strongest.get("channel",{}).get("csi_nmse",0)<.5,"equalizer_stable":strongest.get("channel",{}).get("maximum_equalizer_gain",0)<1000,**{f"infrastructure_{k}":bool(v) for k,v in infrastructure.items()},"thresholds":dict(thresholds)}
    g["mean_passed"]=all(v for k,v in g.items() if k not in {"thresholds","tail_pass"});g["passed"]=g["mean_passed"] and g["tail_pass"];return g

def classify_j4(gate,final_is_best,loss_slope):
    if gate["passed"]:return "PASS"
    if not gate.get("infrastructure_finite",True):return "FAIL_NONFINITE"
    if not gate.get("infrastructure_no_leakage",True) or not gate.get("infrastructure_contiguous",True) or not gate.get("infrastructure_full_band",True):return "FAIL_MASK_IMPLEMENTATION"
    if gate.get("mean_passed") and not gate.get("tail_pass"):return "MARGINAL_TAIL"
    if not gate.get("channel_stable",True):return "FAIL_CHANNEL_ESTIMATION"
    if not gate.get("equalizer_stable",True):return "FAIL_EQUALIZER_INSTABILITY"
    if final_is_best and loss_slope<0:return "INCONCLUSIVE_NOT_CONVERGED"
    if not gate.get("layer7_pass",True):return "FAIL_LAYER7"
    if not gate.get("enhancement_pass",True) or not gate.get("deep_pass",True):return "FAIL_ENHANCEMENT"
    return "FAIL_AGGREGATE"

def validate_j4_resume_metadata(saved,current):
    for key in ("stage","selected_distribution_hash","parent_checkpoint_hash","accepted_manifest_hash"):
        if saved.get(key)!=current.get(key):raise ValueError(f"J4 resume mismatch: {key}")
