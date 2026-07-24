from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

from channels.pilot import equalize_with_csi
from evaluation.paired import run_mode_on_paired_batch
from speech_jscc.config import load_config, resolve_device
from speech_jscc.diagnostics.metrics import latent_metric_rows, normalized_layer_loss, zero_predictor_loss
from speech_jscc.diagnostics.o5_root_cause import CONDITIONS, condition_batch, fixed_realization_hashes, linear_slope, optimal_scale_diagnostics, stable_tensor_hash, run_offline_condition, restore_rng_state
from speech_jscc.experiment import build_components
from train_latent_jscc import RepresentationSource, layer_weighted_latent_mse
from train_stage1_fixed_tx import _make_batch


def parse_args():
    p=argparse.ArgumentParser(description="Offline-only fixed-realization O5 root-cause diagnostic")
    p.add_argument("--config",required=True); p.add_argument("--condition",required=True,choices=tuple(CONDITIONS))
    p.add_argument("--steps",required=True,type=int); p.add_argument("--seed",required=True,type=int); p.add_argument("--output_dir",required=True)
    p.add_argument("--device"); p.add_argument("--learning_rate",type=float); p.add_argument("--checkpoint_every",type=int,default=100)
    p.add_argument("--log_every",type=int,default=10); p.add_argument("--resume"); p.add_argument("--overwrite",action="store_true")
    p.add_argument("--dry_run",action="store_true"); p.add_argument("--allow_long_run",action="store_true")
    p.add_argument("--max_runtime_seconds",type=float); p.add_argument("--requested_jsr_db",type=float,default=0.0)
    p.add_argument("--plateau_tolerance",type=float,default=1e-4)
    return p.parse_args()


def _jsonable(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().tolist()
    if isinstance(value,dict): return {k:_jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_jsonable(v) for v in value]
    return value


def _metric_record(step, loss, best, target, reconstruction, model, lr, epsilon):
    rows=latent_metric_rows(reconstruction,target,epsilon=epsilon,predictor="trained",scenario="o5")
    per=[]
    for layer in range(target.shape[1]):
        selected=[r for r in rows if r["layer"]==layer]
        per.append({"layer":layer,**{key:sum(float(r[key]) for r in selected)/len(selected) for key in
            ("raw_mse","normalized_mse","target_power","reconstruction_power","power_ratio","cosine_similarity","pearson_correlation")}})
    def grad(parameters):
        values=[p.grad for p in parameters if p.grad is not None]
        return float(torch.sqrt(sum((g.square().sum() for g in values),torch.tensor(0.,device=target.device)))) if values else 0.0
    scale=optimal_scale_diagnostics(
        reconstruction, target, epsilon, torch.ones(target.shape[1], device=target.device)
    )
    zero=float(zero_predictor_loss(target,torch.ones(target.shape[1],device=target.device),epsilon)[0])
    return {"step":step,"loss":float(loss),"best_loss":float(best),"zero_predictor_loss":zero,
        "relative_improvement_over_zero":(zero-float(loss))/zero,"per_layer":per,
        "aggregate_power_ratio":sum(x["power_ratio"] for x in per)/len(per),
        "aggregate_cosine_similarity":sum(x["cosine_similarity"] for x in per)/len(per),
        "aggregate_pearson_correlation":sum(x["pearson_correlation"] for x in per)/len(per),
        "reconstruction_mean":float(reconstruction.mean()),"reconstruction_std":float(reconstruction.std(unbiased=False)),
        "encoder_gradient_norm":grad(model.encoder.parameters()),"decoder_gradient_norm":grad(model.decoder.parameters()),
        "learning_rate":lr,"optimal_scale":scale}


def _channel_metrics(batch, result, requested_jsr_db):
    pilots=batch.pilot_mask.bool(); data=~pilots; jammer_mask=batch.jammer_mask.bool()
    dims=tuple(range(1,batch.noise.ndim)); eps=1e-12
    tx=result["transmitted"]; signal_power=tx.abs().square().mean(dims); noise_power=batch.noise.abs().square().mean(dims)
    jammer_power=batch.jammer.abs().square().mean(dims); total_jsr=10*torch.log10((jammer_power/signal_power.clamp_min(eps)).clamp_min(eps))
    active_jsr=[]
    for i in range(tx.shape[0]):
        active=jammer_mask[i]; active_jsr.append(float((10*torch.log10((batch.jammer[i][active].abs().square().mean()/tx[i][active].detach().abs().square().mean().clamp_min(eps)).clamp_min(eps))).detach()) if active.any() else -120.0)
    oracle=equalize_with_csi(result["received"],batch.signal_fading); estimated=result["equalized_estimated"]
    def data_sinr(channel):
        desired=equalize_with_csi(result["faded_signal"],channel)[data]
        interference=equalize_with_csi(result["faded_jammer"],channel)[data]+equalize_with_csi(result["noise"],channel)[data]
        return float(10*torch.log10((desired.abs().square().mean()/interference.abs().square().mean().clamp_min(eps)).clamp_min(eps)))
    data_error=(estimated[data]-tx[data]).abs().square().mean(); data_ref=tx[data].abs().square().mean().clamp_min(eps)
    eq_gain=(result["estimated_channel"].abs().square().clamp_min(1e-6).rsqrt())
    return {"requested_snr_db":10.0,"measured_pre_channel_snr_db":float((10*torch.log10(signal_power/noise_power)).mean()),
        "requested_total_grid_jsr_db":requested_jsr_db,"measured_pre_channel_total_grid_jsr_db":float(total_jsr.mean()),
        "active_resource_jsr_db":sum(active_jsr)/len(active_jsr),"jammer_type":batch.jammer_type,
        "jammer_active_resource_count":int(jammer_mask[0].sum()),"pilot_jammer_overlap_count":int((jammer_mask&pilots).sum()/jammer_mask.shape[0]),
        "data_jammer_overlap_count":int((jammer_mask&data).sum()/jammer_mask.shape[0]),"pilot_overlap_fraction":float((jammer_mask&pilots).sum()/pilots.sum().clamp_min(1)),
        "data_overlap_fraction":float((jammer_mask&data).sum()/data.sum().clamp_min(1)),"legitimate_channel_average_power":float(batch.signal_fading.abs().square().mean()),
        "jammer_channel_average_power":float(batch.jammer_fading.abs().square().mean()),"csi_nmse":float(result["csi_nmse"].mean()),
        "pilot_evm":float(result["pilot_evm"].mean()),"data_evm":float(torch.sqrt(data_error/data_ref)),
        "equalized_symbol_mse":float(data_error),"estimated_csi_post_equalization_sinr_db":data_sinr(result["estimated_channel"]),"oracle_csi_post_equalization_sinr_db":data_sinr(batch.signal_fading),
        "oracle_csi_equalized_data_mse":float((oracle[data]-tx[data]).abs().square().mean()),"maximum_equalizer_gain":float(eq_gain.max()),
        "fifth_percentile_legitimate_channel_power":float(torch.quantile(batch.signal_fading.abs().square(),.05)),
        "post_channel_total_grid_jsr_db":float(10*torch.log10((result["faded_jammer"].abs().square().mean()/result["faded_signal"].abs().square().mean().clamp_min(eps)).clamp_min(eps)))}


def _checkpoint_payload(model, optimizer, step, condition, config, hashes, history):
    return {"diagnostic_type":"o5_root_cause_fixed_realization","model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":step,"condition":condition,"config":config,"fixed_realization_specification":{"seed":config["seed"],"batch_seed":config["seed"]+23000,"condition":condition},"resource_mapping":{"version":"pilot_reserved_v1"},"hashes":hashes,"metric_history_summary":history,"rng_state":{"torch":torch.get_rng_state(),"python":random.getstate()}}


def main():
    args=parse_args()
    if args.steps<0: raise SystemExit("--steps must be nonnegative")
    out=Path(args.output_dir)
    command=" ".join(sys.argv)
    if args.dry_run:
        print(json.dumps({"dry_run":True,"condition":args.condition,"steps":args.steps,"output_dir":str(out),"command":command},indent=2)); return
    if args.steps>5 and not args.allow_long_run: raise SystemExit("steps > 5 require --allow_long_run")
    if out.exists() and not args.overwrite and not args.resume: raise SystemExit(f"refusing existing output directory: {out}")
    out.mkdir(parents=True,exist_ok=True)
    config=load_config(args.config); config["seed"]=args.seed
    if args.device: config["device"]=args.device
    device=resolve_device(config.get("device","auto")); torch.manual_seed(args.seed); random.seed(args.seed)
    codec,model=build_components(config,device); codec.eval()
    for parameter in codec.parameters(): parameter.requires_grad_(False)
    target,_=RepresentationSource(config,codec,device,"train").next_batch(1)
    base=_make_batch(codec,model,config,target=target,waveform=None,snr_db=10.0,jsr_db=args.requested_jsr_db,jammer_type="barrage",seed=args.seed+23000,device=device)
    batch=condition_batch(base,args.condition,args.requested_jsr_db)
    hashes=fixed_realization_hashes(batch,target,model)
    immutable_hashes={"legitimate_channel":stable_tensor_hash(batch.signal_fading),"jammer_channel":stable_tensor_hash(batch.jammer_fading),"awgn":stable_tensor_hash(batch.noise),"raw_jammer_waveform":stable_tensor_hash(batch.jammer),"jammer_mask":stable_tensor_hash(batch.jammer_mask),"pilot_mask":stable_tensor_hash(batch.pilot_mask)}
    lr=args.learning_rate or float(config["train"]["learning_rate"]); optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    start=0; history=[]
    if args.resume:
        # Keep RNG state on CPU; model/optimizer state_dict loading copies tensors
        # to the devices owned by their destination parameters.
        payload=torch.load(args.resume,map_location="cpu",weights_only=False)
        if payload.get("diagnostic_type")!="o5_root_cause_fixed_realization": raise SystemExit("not an O5 diagnostic checkpoint")
        if payload["condition"]!=args.condition or any(payload["hashes"].get(key)!=value for key,value in hashes.items()): raise SystemExit("resume condition/fixed realization hash mismatch")
        hashes=payload["hashes"]
        model.load_state_dict(payload["model"],strict=True); optimizer.load_state_dict(payload["optimizer"]); start=int(payload["step"]); history=payload["metric_history_summary"]
        if args.steps < start: raise SystemExit("--steps must be >= resumed checkpoint step")
        restore_rng_state(payload["rng_state"])
    config_resolved={**config,"o5_diagnostic":{"condition":args.condition,"requested_jsr_db":args.requested_jsr_db}}
    (out/"resolved_config.yaml").write_text(yaml.safe_dump(config_resolved,sort_keys=True)); (out/"command.txt").write_text(command+"\n")
    git=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True,check=False).stdout.strip()
    (out/"environment.json").write_text(json.dumps({"python":sys.version,"torch":torch.__version__,"platform":platform.platform(),"device":str(device),"git_commit":git},indent=2))
    (out/"fixed_realization_hashes.json").write_text(json.dumps(hashes,indent=2))
    state=torch.zeros(1,model.encoder.channel_state_dim,device=device); gates=torch.ones(1,model.encoder.num_layers,device=device)
    best=min([x["loss"] for x in history],default=float("inf")); began=time.monotonic(); metrics_path=out/"metrics.jsonl"
    mode=CONDITIONS[args.condition]
    with metrics_path.open("a" if args.resume else "w") as handle:
        for step in range(start,args.steps+1):
            current={"legitimate_channel":stable_tensor_hash(batch.signal_fading),"jammer_channel":stable_tensor_hash(batch.jammer_fading),"awgn":stable_tensor_hash(batch.noise),"raw_jammer_waveform":stable_tensor_hash(batch.jammer),"jammer_mask":stable_tensor_hash(batch.jammer_mask),"pilot_mask":stable_tensor_hash(batch.pilot_mask)}
            if current != immutable_hashes: raise RuntimeError("fixed realization changed across optimization steps")
            optimizer.zero_grad(set_to_none=True)
            result=run_offline_condition(codec,model,batch,state,gates,args.condition,config)
            loss,_=layer_weighted_latent_mse(result["reconstruction"],target,torch.ones(8,device=device),config["train"]["latent_normalization"])
            loss.backward(); best=min(best,float(loss.detach()))
            if step==0 or step%args.log_every==0 or step==args.steps:
                record=_metric_record(step,float(loss.detach()),best,target,result["reconstruction"].detach(),model,lr,float(config["train"]["latent_normalization"]["epsilon"])); record["channel"]=_channel_metrics(batch,result,args.requested_jsr_db); history.append(record); handle.write(json.dumps(record)+"\n"); handle.flush()
                if step==0:
                    hashes["transmitted_data_symbols_at_initialization"]=stable_tensor_hash(result["data_symbols"])
                    (out/"fixed_realization_hashes.json").write_text(json.dumps(hashes,indent=2))
            if step<args.steps: optimizer.step()
            if step>0 and step%args.checkpoint_every==0:
                torch.save(_checkpoint_payload(model,optimizer,step,args.condition,config_resolved,hashes,history),out/"diagnostic_last.pt")
            if args.max_runtime_seconds and time.monotonic()-began>args.max_runtime_seconds: raise TimeoutError("max runtime exceeded")
    window=history[max(0,int(len(history)*.8)):]; losses=[x["loss"] for x in window]; powers=[x["aggregate_power_ratio"] for x in window]; corrs=[x["aggregate_pearson_correlation"] for x in window]
    final=history[-1]; ls=linear_slope(losses); ps=linear_slope(powers); cs=linear_slope(corrs)
    status="insufficient_history" if len(history)<5 else ("amplitude_suppression" if final["aggregate_power_ratio"]<.1 and final["aggregate_pearson_correlation"]>0 else ("optimization_still_progressing" if final["loss"]>.5 and ls < -args.plateau_tolerance and ps>0 else ("plateaued_failure" if final["loss"]>.5 and abs(ls)<=args.plateau_tolerance and abs(ps)<=args.plateau_tolerance else "inconclusive")))
    summary={"condition":args.condition,"steps":args.steps,"best_step":min(history,key=lambda x:x["loss"])["step"],"final_loss":final["loss"],"final_window_mean_loss":sum(losses)/len(losses),"loss_slope":ls,"power_ratio_slope":ps,"correlation_slope":cs,"plateau_status":status,
        "final_power_ratio":final["aggregate_power_ratio"],"final_correlation":final["aggregate_pearson_correlation"],
        "global_power_weighted_rescaled_nmse":final["optimal_scale"]["global_power_weighted_rescaled_nmse"],
        "stage1_layerwise_rescaled_loss":final["optimal_scale"]["stage1_layerwise_rescaled_loss"],
        "csi_nmse":final["channel"]["csi_nmse"],"data_evm":final["channel"]["data_evm"],"channel_diagnostics":final["channel"],
        "jsr_convention":"requested JSR is total-grid; active-resource JSR is reported separately",
        "hashes":hashes,"diagnostic_only_oracle_jammer_subtraction":bool(mode.get("diagnostic_only_oracle_jammer_subtraction",False))}
    (out/"summary.json").write_text(json.dumps(summary,indent=2));
    with (out/"per_layer_summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=final["per_layer"][0].keys()); w.writeheader(); w.writerows(final["per_layer"])
    checkpoint=_checkpoint_payload(model,optimizer,args.steps,args.condition,config_resolved,hashes,history)
    torch.save(checkpoint,out/"diagnostic_last.pt")

if __name__=="__main__": main()
