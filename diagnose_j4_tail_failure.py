from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from evaluation.paired import run_mode_on_paired_batch
from speech_jscc.config import load_config, resolve_device
from speech_jscc.diagnostics.content_generalization import build_content_subsets
from speech_jscc.diagnostics.g0_exposure import exposure_metric_summary
from speech_jscc.diagnostics.j2_barrage import file_sha256, summarize_layer_groups
from speech_jscc.diagnostics.j4_tail import (
    J4_TAIL_VERSION, ROOT_CAUSE_THRESHOLDS, classify_root_cause, data_only_batch,
    distribution, no_jammer_batch, paired_burst_barrage_batches, select_strongest_condition, summarize_failure_rates,
)
from speech_jscc.diagnostics.o5_root_cause import stable_tensor_hash
from speech_jscc.experiment import build_components
from train_latent_jscc import RepresentationSource
from train_stage1_fixed_tx import _make_batch
from diagnose_j2_barrage_boundary import _find


def parse_args():
    p=argparse.ArgumentParser(description="J4 Layer-7 failure-only paired diagnostic")
    p.add_argument("--config",default="configs/conv_conformer_j4_random_burst.yaml")
    p.add_argument("--diagnostic-config",default="configs/j4_failure_diagnostic.yaml")
    p.add_argument("--j3-checkpoint",required=True);p.add_argument("--j4-checkpoint",required=True)
    p.add_argument("--j4-best-checkpoint");p.add_argument("--j4-summary",required=True)
    p.add_argument("--selected-distribution",required=True);p.add_argument("--output-dir",required=True)
    p.add_argument("--device",default="cuda");p.add_argument("--seed",type=int,default=23)
    p.add_argument("--unseen-utterances",type=int,default=64);p.add_argument("--realizations-per-utterance",type=int,default=2)
    p.add_argument("--gain-caps",default="10,20,40");p.add_argument("--allow-long-run",action="store_true")
    p.add_argument("--overwrite",action="store_true");p.add_argument("--dry-run",action="store_true");p.add_argument("--smoke-test",action="store_true")
    return p.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields=sorted({key for row in rows for key in row})
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def _hashes(batch) -> dict[str,str]:
    return {"latent":stable_tensor_hash(batch.representation),"burst_mask":stable_tensor_hash(batch.jammer_mask),
            "legitimate_channel":stable_tensor_hash(batch.signal_fading),"jammer_channel":stable_tensor_hash(batch.jammer_fading),
            "jammer_waveform":stable_tensor_hash(batch.jammer),"awgn":stable_tensor_hash(batch.noise),
            "pilot_mask":stable_tensor_hash(batch.pilot_mask)}


def _state(model,target):
    state=torch.zeros(target.shape[0],model.encoder.channel_state_dim,device=target.device,dtype=target.dtype)
    gates=torch.ones(target.shape[0],model.encoder.num_layers,device=target.device,dtype=target.dtype)
    return state,gates


def _evaluate(model,codec,batch,target,config,*,equalizer="estimated",cap=None):
    state,gates=_state(model,target)
    result=run_mode_on_paired_batch(codec,model,batch,state,gates,equalizer=equalizer,
        fading="multipath_block",channel_estimator="dft_tap_ls",
        estimator_num_taps=config["channel"].get("estimator_num_taps",6),
        estimator_ridge_lambda=config["channel"].get("estimator_ridge_lambda",1e-6),
        allocation_mode="uniform",resource_reliability=torch.ones_like(batch.noise.real),
        receiver_state_mode="observable_v1",decode_waveform=False,equalizer_gain_cap=cap)
    metric=exposure_metric_summary(result["reconstruction"],target,group="unseen_speaker_tail")
    group=summarize_layer_groups(metric); gain=result["estimated_channel"].abs().clamp_min(1e-8).reciprocal().flatten()
    signal=result["faded_signal"].abs().square().mean();jammer=result["faded_jammer"].abs().square().mean();noise=result["noise"].abs().square().mean()
    row={}
    for prefix in ("aggregate","layers1_to_7","layers6_to_7","layer7"):
        for key,value in group[prefix].items():
            if isinstance(value,(int,float)):row[f"{prefix}_{key}"]=value
    row.update({"csi_nmse":float(result["csi_nmse"].mean()),"pilot_evm":float(result["pilot_evm"].mean()),
        "post_equalization_sinr_db":float(10*torch.log10(result["post_equalization_sinr"].clamp_min(1e-12)).mean()),
        "equalizer_gain_mean":float(gain.mean()),"equalizer_gain_p95":float(torch.quantile(gain,.95)),
        "equalizer_gain_p99":float(torch.quantile(gain,.99)),"maximum_equalizer_gain":float(gain.max()),
        "realized_received_snr_db":float(10*torch.log10(signal/noise.clamp_min(1e-12))),
        "realized_received_global_jsr_db":float(10*torch.log10(jammer/signal.clamp_min(1e-12))),
        "finite":all(bool(torch.isfinite(x).all()) for x in result.values() if isinstance(x,torch.Tensor))})
    return row


def _aggregate(rows):
    output=[]
    for (checkpoint,mode),members in sorted(_group(rows,"checkpoint","mode").items()):
        item={"checkpoint":checkpoint,"mode":mode,"realizations":len(members),"utterances":len({r['utterance_id'] for r in members})}
        numeric=[k for k,v in members[0].items() if isinstance(v,(int,float)) and k not in {"seed","realization"}]
        for key in numeric:
            values=[float(r[key]) for r in members if math.isfinite(float(r[key]))]
            if not values: continue
            stats=distribution(values)
            for stat,value in stats.items():item[f"{key}_{stat}"]=value
        item["failure_rates"]=summarize_failure_rates([{"utterance_id":r["utterance_id"],"layer7_improvement":r["layer7_relative_improvement_over_zero"]} for r in members])
        output.append(item)
    return output


def _group(rows,*keys):
    result={}
    for row in rows:result.setdefault(tuple(row[k] for k in keys),[]).append(row)
    return result


def _tail(aggregate,checkpoint,mode):
    row=next(x for x in aggregate if x["checkpoint"]==checkpoint and x["mode"]==mode)
    return {"negative_rate":row["layer7_relative_improvement_over_zero_negative_rate"],
            "p10":row["layer7_relative_improvement_over_zero_p10"],
            "mean":row["layer7_relative_improvement_over_zero_mean"]}


def _plots(rows,out):
    try: import matplotlib.pyplot as plt
    except ImportError:return
    out.mkdir(parents=True,exist_ok=True);final=[r for r in rows if r['checkpoint']=='j4_final']
    specs=[('mode','layer7_relative_improvement_over_zero','layer7_distributions'),('maximum_equalizer_gain','layer7_relative_improvement_over_zero','equalizer_gain_tail'),('csi_nmse','layer7_relative_improvement_over_zero','csi_nmse_vs_layer7')]
    for x,y,name in specs:
        fig,ax=plt.subplots()
        if x=='mode':
            groups=_group(final,'mode');labels=[k[0] for k in groups];ax.boxplot([[r[y] for r in groups[(label,)]] for label in labels],tick_labels=labels);ax.tick_params(axis='x',rotation=70)
        else:ax.scatter([r[x] for r in final],[r[y] for r in final],s=8,alpha=.5)
        ax.set_xlabel(x);ax.set_ylabel(y);fig.tight_layout();fig.savefig(out/f'{name}.png');plt.close(fig)


def main():
    a=parse_args();caps=[float(x) for x in a.gain_caps.split(',') if x];distribution_cfg=json.loads(Path(a.selected_distribution).read_text());condition=select_strongest_condition(distribution_cfg)
    evaluations=a.unseen_utterances*a.realizations_per_utterance*(6+2*len(caps)+2)*2
    if a.dry_run:
        print(json.dumps({"dry_run":True,"version":J4_TAIL_VERSION,"condition":condition,"minimum_content_samples":a.unseen_utterances,
            "realizations_per_content":a.realizations_per_utterance,"gain_caps":caps,"approximate_evaluations":evaluations,"output_dir":a.output_dir},indent=2));return
    if evaluations>20 and not a.allow_long_run:raise SystemExit("full J4 tail diagnostic requires --allow-long-run")
    out=Path(a.output_dir)
    if out.exists() and not a.overwrite:raise SystemExit(f"refusing existing output directory: {out}")
    out.mkdir(parents=True,exist_ok=True);original_summary=Path(a.j4_summary);original_hash=file_sha256(original_summary)
    config=load_config(a.config);config['device']=a.device;device=resolve_device(a.device);codec,_=build_components(config,device);codec.eval();[p.requires_grad_(False) for p in codec.parameters()]
    checkpoint_paths={"accepted_j3":Path(a.j3_checkpoint),"j4_final":Path(a.j4_checkpoint)}
    if a.j4_best_checkpoint and Path(a.j4_best_checkpoint).is_file():checkpoint_paths["j4_best"]=Path(a.j4_best_checkpoint)
    models={}
    for label,path in checkpoint_paths.items():
        _,model=build_components(config,device);cp=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(cp['model'],strict=True);model.eval();models[label]=model
    subsets=build_content_subsets(Path(config['data']['train_manifest']),Path(config['data']['valid_manifest']),Path(config['data']['latent_cache_dir']),seed=a.seed,validation_items_per_group=max(128,a.unseen_utterances))
    ids=subsets['subsets']['256']['unseen_speaker_ids'][:a.unseen_utterances]
    if len(ids)<64 and not a.smoke_test:raise SystemExit(f"at least 64 unseen-speaker utterances required, found {len(ids)}")
    source=RepresentationSource(config,codec,device,'val');rows=[];failed=[]
    for content_index,identifier in enumerate(ids):
      target,waveform=_find(source,identifier)
      for realization in range(a.realizations_per_utterance):
        seed=int(hashlib.sha256(f"j4_tail|{a.seed}|{identifier}|{realization}".encode()).hexdigest()[:8],16)
        cfg=json.loads(json.dumps(config));cfg['channel']['jammed_fraction']=condition['burst_fraction']
        burst=_make_batch(codec,next(iter(models.values())),cfg,target=target,waveform=waveform,snr_db=condition['snr_db'],jsr_db=condition['jsr_db'],jammer_type='burst',seed=seed,device=device)
        active_jsr=condition['jsr_db']-10*math.log10(condition['burst_fraction'])
        burst,barrage_equal,barrage_active,canonical_jammer_hash=paired_burst_barrage_batches(burst,seed=seed+99173)
        batches={"estimated_normal":(burst,"estimated",None),"oracle_normal":(burst,"oracle",None),
                 "data_only":(data_only_batch(burst),"estimated",None),"no_jammer":(no_jammer_batch(burst),"estimated",None),
                 "barrage_equal_global":(barrage_equal,"estimated",None),"barrage_equal_active":(barrage_active,"estimated",None)}
        for cap in caps:batches[f"estimated_clip_{cap:g}"]=(burst,"estimated",cap);batches[f"oracle_clip_{cap:g}"]=(burst,"oracle",cap)
        base_hashes=_hashes(burst)
        with torch.no_grad():
          for checkpoint,model in models.items():
            for mode,(batch,equalizer,cap) in batches.items():
              metric=_evaluate(model,codec,batch,target,cfg,equalizer=equalizer,cap=cap)
              row={"checkpoint":checkpoint,"mode":mode,"utterance_id":identifier,"content_index":content_index,"realization":realization,"seed":seed,
                   "snr_db":condition['snr_db'],"requested_global_jsr_db":condition['jsr_db'],"burst_fraction":condition['burst_fraction'],
                   "active_window_jsr_db":active_jsr,**{f"paired_{k}_hash":v for k,v in base_hashes.items()},**metric};rows.append(row)
              row["canonical_jammer_source_hash"]=canonical_jammer_hash
              if metric['layer7_relative_improvement_over_zero']<0:failed.append({k:row[k] for k in row if k.endswith('_hash') or k in {'checkpoint','mode','utterance_id','realization','seed','layer7_relative_improvement_over_zero','csi_nmse','maximum_equalizer_gain'}})
    aggregate=_aggregate(rows);final='j4_final';baseline=_tail(aggregate,final,'estimated_normal')
    classification=classify_root_cause(baseline,oracle=_tail(aggregate,final,'oracle_normal'),data_only=_tail(aggregate,final,'data_only'),
        clipped={int(c):_tail(aggregate,final,f'estimated_clip_{c:g}') for c in caps},oracle_clipped={int(c):_tail(aggregate,final,f'oracle_clip_{c:g}') for c in caps},
        equal_global_barrage=_tail(aggregate,final,'barrage_equal_global'),thresholds=ROOT_CAUSE_THRESHOLDS)
    _write_csv(out/'realizations.csv',rows);utterance=[]
    for (checkpoint,mode,identifier),members in _group(rows,'checkpoint','mode','utterance_id').items():
        utterance.append({'checkpoint':checkpoint,'mode':mode,'utterance_id':identifier,'realizations':len(members),
            'layer7_improvement_mean':sum(r['layer7_relative_improvement_over_zero'] for r in members)/len(members),
            'layer7_any_negative':any(r['layer7_relative_improvement_over_zero']<0 for r in members)})
    _write_csv(out/'utterances.csv',utterance);flat_aggregate=[]
    for item in aggregate:
        flat={k:v for k,v in item.items() if not isinstance(v,(dict,list))};flat_aggregate.append(flat)
    _write_csv(out/'aggregated.csv',flat_aggregate);_write_csv(out/'paired_comparison.csv',flat_aggregate)
    confidence={f"{x['checkpoint']}:{x['mode']}":x['failure_rates'] for x in aggregate}
    summary={"version":J4_TAIL_VERSION,"condition":condition,"content_samples":len(ids),"realizations_per_content":a.realizations_per_utterance,
        "realization_samples":len(ids)*a.realizations_per_utterance,"confidence_interval_method":"95% Wilson score interval",
        "checkpoints":{k:{'path':str(v),'sha256':file_sha256(v)} for k,v in checkpoint_paths.items()},"best_j4_available":'j4_best' in checkpoint_paths,
        "aggregate":aggregate,"root_cause":classification,"failed_sample_count":len(failed),"original_j4_summary_sha256":original_hash}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'confidence_intervals.json').write_text(json.dumps(confidence,indent=2));(out/'failed_samples.json').write_text(json.dumps(failed,indent=2))
    supplement={"supplement_version":J4_TAIL_VERSION,"original_summary_path":str(original_summary),"original_summary_sha256":original_hash,
        "original_preserved":file_sha256(original_summary)==original_hash,"corrected_strongest_condition":condition,"diagnostic_summary_path":str(out/'summary.json'),
        "classification":classification['classification']}
    (out/'corrected_strongest_condition_supplement.json').write_text(json.dumps(supplement,indent=2));_plots(rows,out/'plots')
    (out/'resolved_config.yaml').write_text(yaml.safe_dump({'stage_config':config,'diagnostic_config':yaml.safe_load(Path(a.diagnostic_config).read_text()),'condition':condition}))
    print(json.dumps({"classification":classification['classification'],"content_samples":len(ids),"realization_samples":len(ids)*a.realizations_per_utterance,"output_dir":str(out)},indent=2))


if __name__=='__main__':main()
