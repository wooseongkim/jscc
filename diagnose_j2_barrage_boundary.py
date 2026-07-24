from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
import yaml

from speech_jscc.config import load_config, resolve_device
from speech_jscc.diagnostics.conv_conformer_integration import jammer_power_diagnostics
from speech_jscc.diagnostics.g0_exposure import exposure_metric_summary
from speech_jscc.diagnostics.j2_barrage import (
    J2_JSR_GRID, J2_SNR_GRID, J2_VERSION, aggregate_realizations,
    derive_sweep_seed, select_training_range, summarize_layer_groups, sweep_grid,
    verify_j1_artifact,
)
from speech_jscc.diagnostics.o5_root_cause import stable_tensor_hash
from speech_jscc.experiment import build_components
from train_latent_jscc import RepresentationSource
from train_stage1_fixed_tx import _make_batch
from speech_jscc.diagnostics.conv_conformer_integration import forward_integration_path


GROUPS=("seen_utterance_unseen_channel","same_speaker_unseen_utterance_unseen_channel","unseen_speaker_unseen_utterance_unseen_channel")
METRICS=("aggregate_normalized_mse","aggregate_raw_mse","aggregate_relative_improvement_over_zero",
    "aggregate_cosine_similarity","aggregate_pearson_correlation","aggregate_power_ratio",
    "aggregate_optimal_scalar_rescaled_normalized_loss","layers1_to_7_relative_improvement_over_zero",
    "layers6_to_7_relative_improvement_over_zero","layer7_relative_improvement_over_zero",
    "post_equalization_sinr_db","csi_nmse","pilot_evm","maximum_equalizer_gain")


def parse_values(text: str) -> list[float]: return [float(value) for value in text.split(",") if value.strip()]


def parse_args():
    p=argparse.ArgumentParser(description="No-training J2 barrage boundary sweep")
    p.add_argument("--config",required=True);p.add_argument("--j1-summary",required=True);p.add_argument("--j1-checkpoint",required=True)
    p.add_argument("--output-dir",required=True);p.add_argument("--device",default="cuda");p.add_argument("--seed",type=int,default=23)
    p.add_argument("--snr-values",default=",".join(map(str,J2_SNR_GRID)));p.add_argument("--jsr-values",default=",".join(map(str,J2_JSR_GRID)))
    p.add_argument("--realizations",type=int,default=16);p.add_argument("--overwrite",action="store_true");p.add_argument("--dry-run",action="store_true");p.add_argument("--allow-long-run",action="store_true")
    return p.parse_args()


def _find(source,identifier):
    matches=[i for i,path in enumerate(source.dataset.paths) if path.as_posix().endswith(identifier) or path.name==Path(identifier).name]
    if len(matches)!=1: raise ValueError(f"utterance does not uniquely resolve: {identifier}")
    latent,waveform=source.dataset[matches[0]];return latent.unsqueeze(0),waveform.unsqueeze(0)


def _source(sources,identifier):
    for source in sources.values():
        try:_find(source,identifier);return source
        except ValueError:pass
    raise ValueError(f"utterance unavailable: {identifier}")


def _flatten_metrics(summary):
    groups=summarize_layer_groups(summary); row={}
    for prefix,value in groups.items():
        for key,number in value.items():
            if isinstance(number,(int,float)):row[f"{prefix}_{key}"]=number
    for layer,value in enumerate(summary["per_layer"]):
        for key,number in value.items():
            if isinstance(number,(int,float)) and key!="layer":row[f"layer{layer}_{key}"]=number
    return row


def _channel_metrics(result,batch):
    dims=tuple(range(1,result["faded_signal"].ndim));eps=1e-12
    signal=result["faded_signal"].abs().square().mean(dims);jammer=result["faded_jammer"].abs().square().mean(dims);noise=result["noise"].abs().square().mean(dims)
    data_error=(result["decoder_input"]-result["data_symbols"]).abs().square().mean()
    values={"realized_received_snr_db":float((10*torch.log10(signal/noise.clamp_min(eps))).mean()),
        "realized_received_jsr_db":float((10*torch.log10(jammer/signal.clamp_min(eps))).mean()),
        "post_equalization_sinr_db":float((10*torch.log10(result["post_equalization_sinr"].clamp_min(eps))).mean()),
        "csi_nmse":float(result["csi_nmse"].mean()),"pilot_evm":float(result["pilot_evm"].mean()),
        "data_evm":float(data_error.sqrt()),"equalized_symbol_mse":float(data_error),
        "maximum_equalizer_gain":float(result["estimated_channel"].abs().clamp_min(1e-8).reciprocal().max())}
    values.update(jammer_power_diagnostics(batch,result["transmitted"]))
    tensors=[value for value in result.values() if isinstance(value,torch.Tensor)]
    values["finite"]=all(bool(torch.isfinite(value).all()) for value in tensors)
    return values


def _write_csv(path,rows):
    fields=sorted({key for row in rows for key in row})
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def _heatmaps(rows,out):
    try:
        import matplotlib.pyplot as plt
    except ImportError:return
    out.mkdir(parents=True,exist_ok=True)
    requested={"aggregate_normalized_mse":"aggregate_loss","layer7_relative_improvement_over_zero":"layer7_improvement",
        "layers1_to_7_relative_improvement_over_zero":"layers1_to_7_improvement","layers6_to_7_relative_improvement_over_zero":"layers6_to_7_improvement","post_equalization_sinr_db":"post_eq_sinr",
        "csi_nmse":"csi_nmse","maximum_equalizer_gain":"maximum_equalizer_gain"}
    unseen=[row for row in rows if row["group"]==GROUPS[-1]]
    snrs=sorted({row["snr_db"] for row in unseen});jsrs=sorted({row["jsr_db"] for row in unseen})
    for metric,name in requested.items():
        matrix=[[next((r.get(metric+"_mean",float("nan")) for r in unseen if r["snr_db"]==snr and r["jsr_db"]==jsr),float("nan")) for jsr in jsrs] for snr in snrs]
        fig,axis=plt.subplots();image=axis.imshow(matrix,aspect="auto",origin="lower");axis.set_xticks(range(len(jsrs)),jsrs);axis.set_yticks(range(len(snrs)),snrs);axis.set_xlabel("requested JSR dB");axis.set_ylabel("SNR dB");fig.colorbar(image,ax=axis);fig.tight_layout();fig.savefig(out/f"{name}.png");plt.close(fig)


def main():
    args=parse_args();snrs=parse_values(args.snr_values);jsrs=parse_values(args.jsr_values);grid=sweep_grid(snr_values=snrs,jsr_values=jsrs,realizations=args.realizations)
    if args.dry_run:
        print(json.dumps({"dry_run":True,"stage":"j2_boundary_sweep","grid_points":len(grid),"snr_values":snrs,"jsr_values":jsrs,"realizations":args.realizations,"output_dir":args.output_dir},indent=2));return
    if len(grid)>5 and not args.allow_long_run:raise SystemExit("boundary sweeps longer than five evaluations require --allow-long-run")
    out=Path(args.output_dir)
    if out.exists() and not args.overwrite:raise SystemExit(f"refusing existing output directory: {out}")
    out.mkdir(parents=True,exist_ok=True)
    accepted=verify_j1_artifact(args.j1_summary,args.j1_checkpoint)
    config=load_config(args.config);config["device"]=args.device;device=resolve_device(args.device)
    codec,model=build_components(config,device);codec.eval()
    for parameter in codec.parameters():parameter.requires_grad_(False)
    checkpoint=torch.load(args.j1_checkpoint,map_location="cpu",weights_only=False)
    if checkpoint.get("stage_metadata",{}).get("diagnostic_stage")!="j1_weak_random_barrage":raise SystemExit("checkpoint is not accepted J1")
    model.load_state_dict(checkpoint["model"],strict=True);model.eval()
    sources={"train":RepresentationSource(config,codec,device,"train"),"val":RepresentationSource(config,codec,device,"val")}
    j1_suite=json.loads((Path(args.j1_summary).parent/"validation_suite.json").read_text())
    ids={group:sorted({row["utterance_id"] for row in j1_suite["scenarios"] if row["group"]==group}) for group in GROUPS}
    rows=[]
    with torch.no_grad():
        for point in grid:
            for group in GROUPS:
                identifier=ids[group][point["realization"]%len(ids[group])];target,waveform=_find(_source(sources,identifier),identifier)
                seed=derive_sweep_seed(args.seed,point["snr_db"],point["jsr_db"],point["realization"],group)
                batch=_make_batch(codec,model,config,target=target,waveform=waveform,snr_db=point["snr_db"],jsr_db=point["jsr_db"],jammer_type="barrage",seed=seed,device=device)
                result=forward_integration_path("g3_random_clean",codec,model,target,config,batch=batch)
                latent=exposure_metric_summary(result["reconstruction"],target,group=group);flat=_flatten_metrics(latent);channel=_channel_metrics(result,batch)
                rows.append({**point,"group":group,"utterance_id":identifier,"sample_id":f"{group}:{point['snr_db']}:{point['jsr_db']}:{point['realization']}","seed":seed,
                    "legitimate_channel_hash":stable_tensor_hash(batch.signal_fading),"jammer_channel_hash":stable_tensor_hash(batch.jammer_fading),"jammer_waveform_hash":stable_tensor_hash(batch.jammer),"awgn_hash":stable_tensor_hash(batch.noise),**flat,**channel})
    aggregated=aggregate_realizations(rows,METRICS);unseen=[row for row in aggregated if row["group"]==GROUPS[-1]];selection=select_training_range(unseen)
    _write_csv(out/"realizations.csv",rows);_write_csv(out/"aggregated.csv",aggregated);_heatmaps(aggregated,out/"heatmaps")
    worst=sorted(rows,key=lambda row:row["aggregate_normalized_mse"],reverse=True)[:max(1,len(rows)//10)]
    summary={"diagnostic_version":J2_VERSION,"accepted_j1":accepted,"grid":{"snr_db":snrs,"jsr_db":jsrs,"realizations_per_group_point":args.realizations},
        "validation_groups":ids,"row_count":len(rows),"fraction_nonfinite":sum(not row["finite"] for row in rows)/len(rows),
        "worst_case_samples":[{key:row[key] for key in ("sample_id","utterance_id","seed","snr_db","jsr_db","aggregate_normalized_mse","legitimate_channel_hash","jammer_channel_hash","jammer_waveform_hash","awgn_hash")} for row in worst],"selected_training_range":selection}
    (out/"summary.json").write_text(json.dumps(summary,indent=2));(out/"selected_training_range.json").write_text(json.dumps(selection,indent=2));(out/"resolved_config.yaml").write_text(yaml.safe_dump(config,sort_keys=True))
    if out.name=="boundary_sweep": (out.parent/"selected_training_range.json").write_text(json.dumps(selection,indent=2))
    after=verify_j1_artifact(args.j1_summary,args.j1_checkpoint,expected=accepted)
    (out/"j1_artifact_verification.json").write_text(json.dumps(after,indent=2));print(json.dumps({"rows":len(rows),"selected_training_range":selection},indent=2))


if __name__=="__main__":main()
