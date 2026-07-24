from __future__ import annotations

import argparse, hashlib, json, platform, random, subprocess, sys
from pathlib import Path
from typing import Any

import torch, yaml
import csv

from speech_jscc.config import load_config, resolve_device
from speech_jscc.diagnostics.content_generalization import build_content_subsets, build_content_validation_suite, forward_content_path, parse_speaker_id
from speech_jscc.diagnostics.g0_exposure import (CHECKPOINT_EPOCHS, EXPOSURE_ENGINE_VERSION, EpochSampler,
    compute_train_baselines, evaluate_baselines, exposure_metric_summary, gradient_norms, steps_for_epochs)
from speech_jscc.diagnostics.g0_exposure import should_continue_exposure, verify_resume_replay
from speech_jscc.diagnostics.o5_root_cause import linear_slope, stable_tensor_hash
from speech_jscc.experiment import build_components
from speech_jscc.diagnostics.latent_normalization import fit_latent_normalizer
from speech_jscc.diagnostics.architecture_screening import direct_bypass, revised_g0_gate, parameter_report
from speech_jscc.models.architecture_checkpoint import architecture_metadata, validate_architecture_metadata
from train_latent_jscc import RepresentationSource, layer_weighted_latent_mse


def args_parser():
    p=argparse.ArgumentParser(description="Exposure-normalized G0 direct-bypass diagnostic")
    p.add_argument("--config",required=True); p.add_argument("--subset-size",required=True,choices=("16","64","256","full"))
    p.add_argument("--batch-size",type=int,default=4); p.add_argument("--max-epochs",type=int,default=64); p.add_argument("--seed",type=int,default=23)
    p.add_argument("--output-dir",required=True); p.add_argument("--device"); p.add_argument("--resume"); p.add_argument("--overwrite",action="store_true")
    p.add_argument("--dry-run",action="store_true"); p.add_argument("--allow-long-run",action="store_true"); p.add_argument("--plateau-tolerance",type=float,default=1e-4)
    p.add_argument("--architecture",choices=("flat_mlp","normalized_flat_mlp","conv_conformer_v1"),default="flat_mlp")
    p.add_argument("--normalization-mode",choices=("per_layer_scalar","per_layer_per_dimension"),default="per_layer_per_dimension")
    p.add_argument("--num-workers",type=int,default=0); p.add_argument("--continue-on-pass",action="store_true")
    return p


def _git():
    c=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True,check=False).stdout.strip()
    d=bool(subprocess.run(["git","status","--porcelain"],capture_output=True,text=True,check=False).stdout.strip()); return c,d


def _index(source, identifier):
    if source.dataset is None: raise ValueError("manifest-backed real latent source required")
    found=[i for i,p in enumerate(source.dataset.paths) if p.as_posix().endswith(identifier) or p.name==Path(identifier).name]
    if len(found)!=1: raise ValueError(f"utterance does not uniquely resolve: {identifier}")
    return found[0]


def _one(source, identifier): return source.dataset[_index(source,identifier)]


def _batch(source, identifiers):
    values=[_one(source,x) for x in identifiers]
    return torch.stack([x[0] for x in values]), torch.stack([x[1] for x in values])


def _forward(model,target,config,normalizer=None):
    state=torch.zeros(target.shape[0],model.encoder.channel_state_dim,device=target.device,dtype=target.dtype)
    gates=torch.ones(target.shape[0],model.encoder.num_layers,device=target.device,dtype=target.dtype)
    power=torch.ones(model.encoder.num_layers,device=target.device,dtype=target.dtype)
    return direct_bypass(model,target,state,layer_gates=gates,layer_power_allocation=power,normalizer=normalizer)

def _validation(model,codec,config,sources,suite,baselines,normalizer=None):
    model.eval(); groups={}
    with torch.no_grad():
        for group in sorted({x["group"] for x in suite["scenarios"]}):
            scenarios=[x for x in suite["scenarios"] if x["group"]==group]; ids=[x["utterance_id"] for x in scenarios]
            source=sources["val"] if group.startswith("unseen_speaker") else sources["train"]
            target,_=_batch(source,ids); reconstruction=_forward(model,target,config,normalizer)
            metrics=exposure_metric_summary(reconstruction,target,group=group)
            baseline=evaluate_baselines(target,ids,baselines,group=group); model_loss=metrics["aggregate"]["normalized_mse"]
            metrics["baselines"]={name:(value["aggregate"]["normalized_mse"] if value.get("aggregate") else None) for name,value in baseline.items()}
            for name in ("zero","global_mean","layerwise_mean"):
                base=metrics["baselines"][name]; metrics[f"relative_improvement_over_{name}"]=(base-model_loss)/max(base,1e-12)
            groups[group]=metrics
    model.train(); return groups


def main():
    a=args_parser().parse_args()
    if a.max_epochs not in CHECKPOINT_EPOCHS: raise SystemExit(f"--max-epochs must be one of {CHECKPOINT_EPOCHS}")
    if a.batch_size<=0: raise SystemExit("--batch-size must be positive")
    if a.dry_run:
        print(json.dumps({"dry_run":True,"subset_size":a.subset_size,"batch_size":a.batch_size,"max_epochs":a.max_epochs,"command":" ".join(sys.argv)},indent=2)); return
    if a.max_epochs>2 and not a.allow_long_run: raise SystemExit("max epochs > 2 require --allow-long-run")
    out=Path(a.output_dir)
    if out.exists() and not a.overwrite and not a.resume: raise SystemExit(f"refusing existing output directory: {out}")
    out.mkdir(parents=True,exist_ok=True); config=load_config(a.config); config["seed"]=a.seed
    config["model"]["architecture"]="flat_mlp" if a.architecture=="normalized_flat_mlp" else a.architecture
    if a.device: config["device"]=a.device
    device=resolve_device(config.get("device","auto")); torch.manual_seed(a.seed); random.seed(a.seed)
    codec,model=build_components(config,device); codec.eval()
    for p in codec.parameters(): p.requires_grad_(False)
    sources={"train":RepresentationSource(config,codec,device,"train"),"val":RepresentationSource(config,codec,device,"val")}
    data=config["data"]; manifests=build_content_subsets(Path(data["train_manifest"]),Path(data["valid_manifest"]),Path(data["latent_cache_dir"]),seed=a.seed)
    subset=manifests["subsets"][a.subset_size]; suite=build_content_validation_suite(subset,a.seed); train_ids=subset["train_ids"]
    normalizer=None
    if a.architecture=="normalized_flat_mlp":
        normalizer=fit_latent_normalizer((_one(sources["train"],identifier)[0] for identifier in train_ids),mode=a.normalization_mode,
            epsilon=1e-6,split="train",manifest_hash=manifests["train_manifest_hash"],cache_hash=manifests["latent_cache_hash"])
    initial=torch.cat([p.detach().flatten().cpu() for p in model.parameters()]); init_hash=stable_tensor_hash(initial)
    baselines=compute_train_baselines(((identifier,_one(sources["train"],identifier)[0]) for identifier in train_ids),min_speaker_samples=2)
    baseline_hashes={name:stable_tensor_hash(value) for name,value in baselines.items() if isinstance(value,torch.Tensor)}
    optimizer=torch.optim.Adam(model.parameters(),lr=float(config["train"]["learning_rate"])); weights=torch.ones(8,device=device)
    commit,dirty=_git(); target_steps=steps_for_epochs(a.max_epochs,len(train_ids),a.batch_size)
    provenance={"diagnostic_type":"g0_exposure_normalized","diagnostic_engine_version":EXPOSURE_ENGINE_VERSION,"subset_size":a.subset_size,
      "train_subset_size":len(train_ids),"batch_size":a.batch_size,"max_epochs":a.max_epochs,"target_optimizer_steps":target_steps,
      "checkpoint_epochs":[x for x in CHECKPOINT_EPOCHS if x<=a.max_epochs],"seed":a.seed,"model_initialization_hash":init_hash,
      "validation_suite_hash":suite["validation_suite_hash"],"train_manifest_hash":manifests["train_manifest_hash"],"validation_manifest_hash":manifests["validation_manifest_hash"],
      "latent_cache_hash":manifests["latent_cache_hash"],"baseline_hashes":baseline_hashes,"learning_rate":config["train"]["learning_rate"],
      "normalization":config["train"]["latent_normalization"],"preprocessing_normalization":normalizer.metadata if normalizer else {"mode":"none","normalization_stats_hash":"none"},
      "model_architecture":a.architecture,"architecture_version":getattr(model,"architecture_version","flat_mlp_v1"),"git_commit":commit,"working_tree_dirty":dirty}
    checkpoint_meta=architecture_metadata(model,{"normalization_mode":a.normalization_mode if normalizer else "none","normalization_stats_hash":normalizer.metadata["normalization_stats_hash"] if normalizer else "none"},manifests["train_manifest_hash"],manifests["latent_cache_hash"])
    start=0; history=[]; saved_sampler=None; best=float("inf")
    if a.resume:
        payload=torch.load(a.resume,map_location="cpu",weights_only=False); saved=payload["provenance"]
        validate_architecture_metadata(payload.get("architecture_metadata"),checkpoint_meta)
        for key in ("subset_size","batch_size","seed","model_initialization_hash","validation_suite_hash","train_manifest_hash","validation_manifest_hash","latent_cache_hash","learning_rate","normalization","model_architecture","preprocessing_normalization"):
            if saved.get(key)!=provenance.get(key): raise SystemExit(f"resume provenance mismatch: {key}")
        model.load_state_dict(payload["model"],strict=True); optimizer.load_state_dict(payload["optimizer"]); start=int(payload["optimizer_step"])
        history=payload["history"]; saved_sampler=payload["sampler_state"]; best=float(payload.get("best_validation",float("inf")))
    (out/"resolved_config.yaml").write_text(yaml.safe_dump({**config,"g0_exposure":provenance},sort_keys=True)); (out/"command.txt").write_text(" ".join(sys.argv)+"\n")
    (out/"environment.json").write_text(json.dumps({"python":sys.version,"torch":torch.__version__,"platform":platform.platform(),"git_commit":commit,"working_tree_dirty":dirty},indent=2))
    (out/"validation_suite.json").write_text(json.dumps(suite,indent=2)); torch.save(baselines,out/"constant_predictors.pt")
    sampler=EpochSampler(train_ids,batch_size=a.batch_size,seed=a.seed,subset_key=a.subset_size); checkpoints={steps_for_epochs(e,len(train_ids),a.batch_size):e for e in CHECKPOINT_EPOCHS if e<=a.max_epochs}
    metrics_path=out/"metrics.jsonl"; model.train()
    stopped_for_plateau=False
    with metrics_path.open("a" if a.resume else "w") as handle:
      for step,ids in sampler.batch_stream(a.max_epochs):
        if step<=start: continue
        if step==start+1 and saved_sampler is not None:
            # Replayed deterministic batches must reproduce the saved exposure state.
            try: verify_resume_replay(dict(sampler.presentation_counts),ids,saved_sampler["presentation_counts"])
            except ValueError as error: raise SystemExit(str(error)) from error
        target,_=_batch(sources["train"],ids); optimizer.zero_grad(set_to_none=True)
        reconstruction=_forward(model,target,config,normalizer); loss,per=layer_weighted_latent_mse(reconstruction,target,weights,config["train"]["latent_normalization"])
        loss.backward(); grads=gradient_norms(model); torch.nn.utils.clip_grad_norm_(model.parameters(),float(config["train"].get("gradient_clip_norm",5.0))); optimizer.step()
        if step in checkpoints:
            epoch=checkpoints[step]; validation=_validation(model,codec,config,sources,suite,baselines,normalizer); train_metrics=exposure_metric_summary(reconstruction.detach(),target,group="train")
            record={"epoch":epoch,"optimizer_step":step,"sample_presentations":sum(sampler.presentation_counts.values()),"presentation_count_min":min(sampler.presentation_counts.values()),
              "presentation_count_max":max(sampler.presentation_counts.values()),"presentation_counts":dict(sampler.presentation_counts),"train_loss":float(loss.detach()),
              "train_metrics":train_metrics,"validation":validation,"gradient_norms":grads,"train_sample_count":len(train_ids),"validation_sample_count":len(suite["scenarios"])}
            history.append(record); handle.write(json.dumps(record)+"\n"); handle.flush()
            selection=sum(validation[g]["aggregate"]["normalized_mse"] for g in validation if not g.startswith("seen_utterance"))/2
            payload={"diagnostic_type":"g0_architecture_screening","provenance":provenance,"architecture_metadata":checkpoint_meta,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"optimizer_step":step,
              "completed_epoch":epoch,"sampler_state":sampler.state_dict(),"history":history,"best_validation":min(best,selection)}
            if selection<best: best=selection; torch.save(payload,out/"diagnostic_best.pt"); torch.save(payload,out/"best.pt")
            torch.save(payload,out/f"diagnostic_epoch_{epoch}.pt"); print(json.dumps({"epoch":epoch,"step":step,"train_loss":record["train_loss"],"selection_loss":selection}),flush=True)
            if epoch>=16 and len(history)>=3:
                recent={"train_loss":linear_slope([x["train_loss"] for x in history][-3:]),
                  "unseen_loss":linear_slope([x["validation"]["unseen_speaker_unseen_utterance_unseen_channel"]["aggregate"]["normalized_mse"] for x in history][-3:]),
                  "unseen_correlation":linear_slope([x["validation"]["unseen_speaker_unseen_utterance_unseen_channel"]["aggregate"]["pearson_correlation"] for x in history][-3:])}
                if not should_continue_exposure(epoch,recent,a.plateau_tolerance): stopped_for_plateau=True; break
    actual_step=sampler.optimizer_steps; actual_epoch=history[-1]["epoch"]
    final_payload={"diagnostic_type":"g0_architecture_screening","provenance":provenance,"architecture_metadata":checkpoint_meta,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"optimizer_step":actual_step,
      "completed_epoch":actual_epoch,"sampler_state":sampler.state_dict(),"history":history,"best_validation":best}
    torch.save(final_payload,out/"diagnostic_final.pt"); torch.save(final_payload,out/"last.pt")
    final=history[-1]; loss_series=[x["validation"]["unseen_speaker_unseen_utterance_unseen_channel"]["aggregate"]["normalized_mse"] for x in history]
    corr_series=[x["validation"]["unseen_speaker_unseen_utterance_unseen_channel"]["aggregate"]["pearson_correlation"] for x in history]
    slopes={"unseen_loss":linear_slope(loss_series[-3:]),"unseen_correlation":linear_slope(corr_series[-3:]),"train_loss":linear_slope([x["train_loss"] for x in history][-3:])}
    plateau=all(abs(slopes[k])<=a.plateau_tolerance for k in ("unseen_loss","unseen_correlation")) if len(history)>=3 else False
    groups=("same_speaker_unseen_utterance_unseen_channel","unseen_speaker_unseen_utterance_unseen_channel")
    gate_detail=revised_g0_gate(final["validation"],same_group=groups[0],unseen_group=groups[1]); gate=gate_detail["architecture_screening_pass"] and a.subset_size in {"256","full"}
    summary={"provenance":provenance,"history":history,"final":final,"best_validation":best,"slopes":slopes,"plateau":plateau,"gate":{"passed":gate,**gate_detail},"parameter_report":parameter_report(model,codec),
      "presentation_counts":dict(sampler.presentation_counts),"actual_optimizer_steps":sampler.optimizer_steps,"actual_sample_presentations":sum(sampler.presentation_counts.values()),"stopped_for_plateau":stopped_for_plateau}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)); (out/"provenance.json").write_text(json.dumps(provenance,indent=2)); (out/"presentation_counts.json").write_text(json.dumps(dict(sampler.presentation_counts),indent=2))
    with (out/"per_layer_summary.csv").open("w",newline="") as handle:
        rows=[]
        for group,value in final["validation"].items():
            for layer,row in enumerate(value["per_layer"]): rows.append({"group":group,"layer":layer,**{k:v for k,v in row.items() if isinstance(v,(int,float,bool))}})
        writer=csv.DictWriter(handle,fieldnames=sorted({key for row in rows for key in row})); writer.writeheader(); writer.writerows(rows)


if __name__=="__main__": main()
