from __future__ import annotations
import argparse, csv, hashlib, json, subprocess, sys
from pathlib import Path
import torch, yaml

from speech_jscc.config import load_config, resolve_device
from speech_jscc.diagnostics.architecture_screening import revised_g0_gate
from speech_jscc.diagnostics.content_generalization import build_content_subsets, build_content_validation_suite
from speech_jscc.diagnostics.g0_exposure import compute_train_baselines, evaluate_baselines, exposure_metric_summary
from speech_jscc.diagnostics.pca_reference import PerLayerPCAReference
from speech_jscc.experiment import build_components
from train_latent_jscc import RepresentationSource
import diagnose_g0_exposure_normalized as exposure

ARCHITECTURES=("flat_mlp","normalized_flat_mlp","conv_conformer_v1","per_layer_pca_480")

def parser():
    p=argparse.ArgumentParser(description="G0 architecture screening")
    p.add_argument("--config",required=True); p.add_argument("--architecture",required=True,choices=ARCHITECTURES)
    p.add_argument("--subset-size",required=True,choices=("16","64","256","full")); p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--max-epochs",type=int,default=64); p.add_argument("--seed",type=int,default=23); p.add_argument("--output-dir",required=True)
    p.add_argument("--device",default="auto"); p.add_argument("--num-workers",type=int,default=0); p.add_argument("--resume"); p.add_argument("--overwrite",action="store_true")
    p.add_argument("--continue-on-pass",action="store_true"); p.add_argument("--dry-run",action="store_true"); p.add_argument("--allow-long-run",action="store_true")
    return p

def _root(out): return out.parents[1] if out.name.startswith("subset_") else out.parent

def _aggregate(out,summary):
    root=_root(out); root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for path in root.glob("*/subset_*/summary.json"):
        try:
            value=json.loads(path.read_text()); final=value.get("final",{}); validation=final.get("validation",{})
            for group,metric in validation.items(): rows.append({"architecture":value["provenance"].get("model_architecture",path.parent.parent.name),"subset":path.parent.name.removeprefix("subset_"),"group":group,"loss":metric["aggregate"]["normalized_mse"],"correlation":metric["aggregate"]["pearson_correlation"],"power_ratio":metric["aggregate"]["power_ratio"],"passed":value.get("gate",{}).get("passed",False)})
        except (KeyError,ValueError): continue
    (root/"aggregate_results.json").write_text(json.dumps(rows,indent=2)); (root/"architecture_manifest.json").write_text(json.dumps({"architectures":list(ARCHITECTURES),"resource_mapping_version":"pilot_reserved_v1","representation_shape":[8,50,1024],"total_data_channel_uses":1920,"per_layer_channel_uses":[240]*8},indent=2))
    (root/"comparison_manifest.json").write_text(json.dumps({"primary_loss":"uniform_stage1_layer_normalized_mse","validation_groups":["seen_utterance_unseen_channel","same_speaker_unseen_utterance_unseen_channel","unseen_speaker_unseen_utterance_unseen_channel"]},indent=2))
    if rows:
        for name in ("aggregate_by_architecture.csv","aggregate_by_subset.csv","aggregate_by_epoch.csv"):
            with (root/name).open("w",newline="") as h: writer=csv.DictWriter(h,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    reports=[]; layer_rows=[]
    for path in root.glob("*/subset_*/summary.json"):
        try:
            value=json.loads(path.read_text()); reports.append({"architecture":path.parent.parent.name,"subset":path.parent.name,"parameter_report":value.get("parameter_report")})
            for group,metric in value.get("final",{}).get("validation",{}).items():
                for layer,row in enumerate(metric.get("per_layer",[])): layer_rows.append({"architecture":path.parent.parent.name,"subset":path.parent.name,"group":group,"layer":layer,**{k:v for k,v in row.items() if isinstance(v,(int,float,bool))}})
        except (KeyError,ValueError): continue
    (root/"model_parameter_report.json").write_text(json.dumps(reports,indent=2))
    if layer_rows:
        with (root/"per_layer_results.csv").open("w",newline="") as h: writer=csv.DictWriter(h,fieldnames=sorted({k for row in layer_rows for k in row})); writer.writeheader(); writer.writerows(layer_rows)
    commands=[f"bash scripts/run_g0_architecture_screening_external.sh --architecture {architecture} --subset-size {subset} --max-epochs 64 --device cuda" for architecture,subset in (("per_layer_pca_480","full"),("normalized_flat_mlp","16"),("conv_conformer_v1","16"),("conv_conformer_v1","64"),("conv_conformer_v1","256"),("conv_conformer_v1","full"))]
    (root/"external_commands.md").write_text("# External commands\n\n"+"\n".join(f"```bash\n{x}\n```" for x in commands)+"\n")
    (root/"g0_architecture_screening_report.md").write_text("# G0 architecture screening\n\nLong experiments are complete only when their per-architecture summary exists. Revised gates require Layer 0, layers 1–7, same-speaker and unseen-speaker generalization, and beating the global mean.\n")

def run_pca(a):
    if not a.allow_long_run: raise SystemExit("full PCA fitting requires --allow-long-run external acknowledgement")
    out=Path(a.output_dir)
    if out.exists() and not a.overwrite: raise SystemExit(f"refusing existing output directory: {out}")
    out.mkdir(parents=True,exist_ok=True); config=load_config(a.config); config["device"]=a.device; device=resolve_device(a.device)
    codec,_=build_components(config,device); codec.eval(); sources={"train":RepresentationSource(config,codec,device,"train"),"val":RepresentationSource(config,codec,device,"val")}
    manifests=build_content_subsets(Path(config["data"]["train_manifest"]),Path(config["data"]["valid_manifest"]),Path(config["data"]["latent_cache_dir"]),seed=a.seed)
    subset=manifests["subsets"][a.subset_size]; suite=build_content_validation_suite(subset,a.seed); train_ids=subset["train_ids"]
    train=torch.stack([exposure._one(sources["train"],identifier)[0] for identifier in train_ids])
    pca=PerLayerPCAReference(480,a.seed).fit(train,split="train"); baselines=compute_train_baselines(zip(train_ids,train))
    validation={}
    for group in sorted({row["group"] for row in suite["scenarios"]}):
        ids=[row["utterance_id"] for row in suite["scenarios"] if row["group"]==group]; source=sources["val"] if group.startswith("unseen_speaker") else sources["train"]
        target=torch.stack([exposure._one(source,x)[0] for x in ids]); metrics=exposure_metric_summary(pca.reconstruct(target),target,group=group); base=evaluate_baselines(target,ids,baselines,group=group)
        metrics["baselines"]={name:(value["aggregate"]["normalized_mse"] if value.get("aggregate") else None) for name,value in base.items()}
        validation[group]=metrics
    groups=("same_speaker_unseen_utterance_unseen_channel","unseen_speaker_unseen_utterance_unseen_channel"); gate=revised_g0_gate(validation,same_group=groups[0],unseen_group=groups[1])
    provenance={"model_architecture":"per_layer_pca_480","diagnostic_type":"offline_linear_reference","subset_size":a.subset_size,"train_manifest_hash":manifests["train_manifest_hash"],"latent_cache_hash":manifests["latent_cache_hash"],"validation_suite_hash":suite["validation_suite_hash"]}
    summary={"provenance":provenance,"final":{"validation":validation},"gate":{"passed":gate["architecture_screening_pass"],**gate}}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)); (out/"provenance.json").write_text(json.dumps(provenance,indent=2)); (out/"resolved_config.yaml").write_text(yaml.safe_dump(config,sort_keys=True)); (out/"pca_manifest.json").write_text(json.dumps({**pca.metadata,**provenance},indent=2)); torch.save({"mean":pca.mean,"components":pca.components},out/"pca_components.pt")
    with (out/"explained_variance.csv").open("w",newline="") as h:
        w=csv.writer(h); w.writerow(("layer","component","explained_variance_ratio","cumulative_explained_variance"))
        for layer,row in enumerate(pca.explained_variance_ratio):
            cumulative=0.;
            for component,value in enumerate(row): cumulative+=float(value); w.writerow((layer,component,float(value),cumulative))
    _aggregate(out,summary)

def main():
    a=parser().parse_args()
    command=[sys.executable,"diagnose_g0_exposure_normalized.py","--config",a.config,"--architecture",a.architecture,"--subset-size",a.subset_size,"--batch-size",str(a.batch_size),"--max-epochs",str(a.max_epochs),"--seed",str(a.seed),"--output-dir",a.output_dir,"--device",a.device,"--num-workers",str(a.num_workers)]
    if a.allow_long_run: command.append("--allow-long-run")
    if a.overwrite: command.append("--overwrite")
    if a.resume: command.extend(("--resume",a.resume))
    if a.dry_run: print(json.dumps({"dry_run":True,"architecture":a.architecture,"command":command},indent=2)); return
    if a.architecture=="per_layer_pca_480": run_pca(a); return
    result=subprocess.run(command,check=False); 
    if result.returncode: raise SystemExit(result.returncode)
    out=Path(a.output_dir); summary=json.loads((out/"summary.json").read_text()); _aggregate(out,summary)

if __name__=="__main__": main()
