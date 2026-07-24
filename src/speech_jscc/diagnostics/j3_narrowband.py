from __future__ import annotations
import hashlib,json,math,random
from pathlib import Path
from typing import Any
import torch
from speech_jscc.diagnostics.random_distribution import SeedDeriver
from speech_jscc.diagnostics.j2_barrage import file_sha256

J3_VERSION="j3_random_narrowband_v1"
J3_SNR_GRID=(5.,10.,15.);J3_JSR_GRID=(-10.,-5.,0.);J3_FRACTIONS=(.125,.25,.5)
J3_THRESHOLDS={"aggregate_improvement":.10,"aggregate_correlation":.30,"aggregate_power_ratio":.12,
 "enhancement_improvement":.075,"enhancement_correlation":.25,"enhancement_power_ratio":.08,
 "layers6_to_7_improvement":.05,"layer7_improvement":.03,"layer7_power_ratio":.04,
 "worst_aggregate_improvement":.075,"worst_enhancement_improvement":.05,"worst_layer7_improvement":.02}

def j3_initialization_metadata(parent_checkpoint_path,parent_checkpoint_sha256,parent_summary_path,parent_summary_sha256):
    return {"initialization_mode":"j2_transfer","initialization_source_stage":"j2_strong_barrage_boundary","initial_weights_loaded":True,
      "parent_checkpoint_path":str(parent_checkpoint_path),"parent_checkpoint_sha256":str(parent_checkpoint_sha256),
      "parent_summary_path":str(parent_summary_path),"parent_summary_sha256":str(parent_summary_sha256)}

def aggregate_channel_diagnostics(rows:list[dict[str,Any]])->dict[str,Any]:
    """Average scalar diagnostics while retaining realization-specific structures."""
    if not rows:return {}
    output:dict[str,Any]={}
    for key in rows[0]:
        values=[row[key] for row in rows if key in row]
        if all(isinstance(value,bool) for value in values):
            output[key]=all(values)
        elif all(isinstance(value,(int,float)) and not isinstance(value,bool) for value in values):
            output[key]=sum(float(value) for value in values)/len(values)
        else:
            output[key]=values
    return output

def write_training_curves(history:list[dict[str,Any]],output_dir:str|Path)->None:
    if not history:return
    import matplotlib.pyplot as plt
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    series={"loss_vs_step":("loss","training loss"),"power_ratio_vs_step":("power_ratio","aggregate power ratio"),"correlation_vs_step":("pearson_correlation","aggregate correlation")}
    for filename,(key,label) in series.items():
        values=[float(row[key] if key=="loss" else row["aggregate"][key]) for row in history]
        figure,axis=plt.subplots();axis.plot([int(row["step"]) for row in history],values);axis.set_xlabel("optimizer step");axis.set_ylabel(label);figure.tight_layout();figure.savefig(output/f"{filename}.png");plt.close(figure)

def local_inband_jsr_db(global_jsr_db:float,fraction:float)->float:
    if not 0<fraction<=1:raise ValueError("fraction must be in (0,1]")
    return float(global_jsr_db)-10*math.log10(float(fraction))

def sinr_fields(linear)->dict[str,float]:
    value=float(linear) if isinstance(linear,(int,float)) else float(torch.as_tensor(linear).detach().mean());return {"post_equalization_sinr_linear":value,"post_equalization_sinr_db":10*math.log10(max(value,1e-12))}

def narrowband_diagnostics(signal,jammer,mask,pilot_mask,*,requested_fraction,requested_global_jsr_db,faded_signal=None,faded_jammer=None):
    active=mask.any(dim=2);counts=active.sum(dim=1);indices=torch.where(active[0])[0];actual=float(counts.float().mean()/mask.shape[1])
    contiguous=all(bool(torch.equal(torch.where(row)[0],torch.arange(torch.where(row)[0][0],torch.where(row)[0][0]+row.sum(),device=row.device))) for row in active)
    outside=jammer[~mask].abs().square().mean() if (~mask).any() else jammer.real.new_tensor(0.)
    inside=jammer[mask].abs().square().mean();dims=(1,2);global_ratio=jammer.abs().square().mean(dims)/signal.abs().square().mean(dims).clamp_min(1e-12)
    fs=signal if faded_signal is None else faded_signal;fj=jammer if faded_jammer is None else faded_jammer
    received_global=fj.abs().square().mean(dims)/fs.abs().square().mean(dims).clamp_min(1e-12)
    inband=[]
    for i in range(mask.shape[0]):inband.append(fj[i][mask[i]].abs().square().mean()/fs[i][mask[i]].abs().square().mean().clamp_min(1e-12))
    pilot_overlap=(mask&pilot_mask).sum().float()/pilot_mask.sum().clamp_min(1);data=(~pilot_mask);data_overlap=(mask&data).sum().float()/data.sum().clamp_min(1)
    per_layer_counts=[];per_layer_fractions=[]
    for batch_index in range(mask.shape[0]):
        packed=(mask[batch_index]&data[batch_index])[data[batch_index]]
        if packed.numel()%8:raise ValueError("data resources are not divisible into eight layer partitions")
        chunks=packed.reshape(8,-1);per_layer_counts.append(chunks.sum(dim=1).float());per_layer_fractions.append(chunks.float().mean(dim=1))
    layer_counts=torch.stack(per_layer_counts).mean(dim=0);layer_fractions=torch.stack(per_layer_fractions).mean(dim=0)
    return {"requested_global_jsr_db":float(requested_global_jsr_db),"measured_transmit_global_jsr_db":float((10*torch.log10(global_ratio)).detach().mean()),
      "realized_received_global_jsr_db":float((10*torch.log10(received_global)).detach().mean()),"realized_received_inband_jsr_db":float((10*torch.log10(torch.stack(inband))).detach().mean()),
      "local_inband_jsr_db":local_inband_jsr_db(requested_global_jsr_db,actual),"requested_jammed_subcarrier_fraction":float(requested_fraction),
      "jammed_subcarrier_fraction":actual,"actual_jammed_subcarrier_count":int(counts[0]),"narrowband_start_index":int(indices[0]),"narrowband_end_index":int(indices[-1]),
      "narrowband_subcarrier_indices":[int(x) for x in indices],"contiguous_band_verified":contiguous,
      "per_layer_jammed_data_count":[float(x) for x in layer_counts.detach().cpu()],"per_layer_jammed_data_fraction":[float(x) for x in layer_fractions.detach().cpu()],
      "pilot_resource_overlap_ratio":float(pilot_overlap.detach()),"data_resource_overlap_ratio":float(data_overlap.detach()),
      "jammer_power_inside_band":float(inside.detach()),"leakage_power_outside_band":float(outside.detach())}

def j3_policy(root_seed,step,snr_range,jsr_range,fractions):
    derive=SeedDeriver(root_seed);rng=random.Random(derive.seed("j3_distribution",step));values=list(map(float,fractions))
    return {"seed":derive.seed("j3_channel_jammer_noise",step),"snr_db":rng.uniform(*map(float,snr_range)),"jsr_db":rng.uniform(*map(float,jsr_range)),"jammed_fraction":values[rng.randrange(len(values))],"jammer_type":"narrowband"}

def validate_j3_resume_metadata(saved:dict[str,Any],current:dict[str,Any])->None:
    for key in ("stage","selected_distribution_hash","parent_checkpoint_hash"):
        if saved.get(key)!=current.get(key):raise ValueError(f"J3 resume mismatch: {key}")

def verify_j2_artifact(summary_path,checkpoint_path,expected=None):
    summary=json.loads(Path(summary_path).read_text());p=summary.get("provenance",{})
    if summary.get("classification")!="PASS" or p.get("stage_name")!="j2_strong_barrage" or p.get("model_architecture")!="conv_conformer_v1":raise ValueError("accepted J2 PASS artifact required")
    result={"summary_path":str(summary_path),"checkpoint_path":str(checkpoint_path),"summary_sha256":file_sha256(summary_path),"checkpoint_sha256":file_sha256(checkpoint_path),
      "stage_name":p.get("stage_name"),"architecture":p.get("model_architecture"),"architecture_version":p.get("architecture_version"),"subset_size":p.get("subset_size"),
      "latent_cache_hash":p.get("latent_cache_hash"),"train_manifest_hash":p.get("train_manifest_hash"),"validation_suite_hash":p.get("validation_suite_hash"),"git_commit":p.get("git_commit")}
    if expected and any(expected[k]!=result[k] for k in ("summary_sha256","checkpoint_sha256")):raise ValueError("accepted J2 artifact hash changed")
    return result

def j3_gate(unseen,strongest,*,infrastructure,thresholds=J3_THRESHOLDS):
    a=unseen["aggregate"];e=unseen["layers1_to_7"];d=unseen["layers6_to_7"];l=unseen["layer7"];wa=strongest["aggregate"];we=strongest["layers1_to_7"];wl=strongest["layer7"]
    g={"aggregate_pass":a["relative_improvement_over_zero"]>=thresholds["aggregate_improvement"] and a["pearson_correlation"]>=thresholds["aggregate_correlation"] and a["power_ratio"]>=thresholds["aggregate_power_ratio"],
      "enhancement_pass":e["relative_improvement_over_zero"]>=thresholds["enhancement_improvement"] and e["pearson_correlation"]>=thresholds["enhancement_correlation"] and e["power_ratio"]>=thresholds["enhancement_power_ratio"],
      "deep_pass":d["relative_improvement_over_zero"]>=thresholds["layers6_to_7_improvement"],
      "layer7_pass":l["relative_improvement_over_zero"]>=thresholds["layer7_improvement"] and l["pearson_correlation"]>0 and l["power_ratio"]>=thresholds["layer7_power_ratio"],
      "worst_pass":wa["relative_improvement_over_zero"]>=thresholds["worst_aggregate_improvement"] and we["relative_improvement_over_zero"]>=thresholds["worst_enhancement_improvement"] and wl["relative_improvement_over_zero"]>=thresholds["worst_layer7_improvement"] and min(wa["pearson_correlation"],we["pearson_correlation"],wl["pearson_correlation"])>0,
      **{f"infrastructure_{k}":bool(v) for k,v in infrastructure.items()},"channel_stable":strongest.get("channel",{}).get("csi_nmse",0)<.5,"equalizer_stable":strongest.get("channel",{}).get("maximum_equalizer_gain",0)<1000,"thresholds":dict(thresholds)}
    g["layer7_improvement"]=float(l["relative_improvement_over_zero"])
    g["passed"]=all(v for k,v in g.items() if k not in {"thresholds","layer7_improvement"});return g

def classify_j3(gate,final_is_best,loss_slope):
    if gate["passed"]:return "PASS"
    if not gate.get("infrastructure_finite",True):return "FAIL_NONFINITE"
    if not gate.get("infrastructure_no_leakage",True) or not gate.get("infrastructure_contiguous",True):return "FAIL_MASK_IMPLEMENTATION"
    if gate.get("pilot_overlap_dominant",False):return "FAIL_PILOT_OVERLAP"
    if not gate.get("channel_stable",True):return "FAIL_CHANNEL_ESTIMATION"
    if not gate.get("equalizer_stable",True):return "FAIL_EQUALIZER_INSTABILITY"
    if final_is_best and loss_slope<0:return "INCONCLUSIVE_NOT_CONVERGED"
    if not gate.get("layer7_pass",True):
        improvement=float(gate.get("layer7_improvement",float("-inf")))
        return "MARGINAL_LAYER7" if improvement>=J3_THRESHOLDS["worst_layer7_improvement"] else "FAIL_LAYER7"
    if not gate.get("enhancement_pass",True) or not gate.get("deep_pass",True):return "FAIL_ENHANCEMENT"
    return "FAIL_AGGREGATE"
