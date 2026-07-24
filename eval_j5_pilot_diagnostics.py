from __future__ import annotations
import argparse,csv,hashlib,json,math
from dataclasses import replace
from pathlib import Path
import torch
from evaluation.paired import run_mode_on_paired_batch
from speech_jscc.config import load_config,resolve_device
from speech_jscc.experiment import build_components
from speech_jscc.diagnostics.content_generalization import build_content_subsets
from speech_jscc.diagnostics.g0_exposure import exposure_metric_summary
from speech_jscc.diagnostics.j2_barrage import summarize_layer_groups,file_sha256
from speech_jscc.diagnostics.j4_tail import distribution,summarize_failure_rates
from speech_jscc.diagnostics.j5_pilot import normalize_pilot_local_batch,verify_j4_accepted
from train_latent_jscc import RepresentationSource
from train_stage1_fixed_tx import _make_batch
from diagnose_j2_barrage_boundary import _find

def parse():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--j4-manifest',required=True);p.add_argument('--j4-checkpoint',required=True);p.add_argument('--j5-checkpoint',required=True);p.add_argument('--selected-distribution',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=23);p.add_argument('--unseen-utterances',type=int,default=64);p.add_argument('--realizations-per-utterance',type=int,default=2);p.add_argument('--allow-long-run',action='store_true');p.add_argument('--overwrite',action='store_true');p.add_argument('--dry-run',action='store_true');return p.parse_args()
def state(model,target):return torch.zeros(target.shape[0],model.encoder.channel_state_dim,device=target.device),torch.ones(target.shape[0],8,device=target.device)
def data_only(batch):
 mask=(~batch.pilot_mask).expand_as(batch.jammer_mask);jam=torch.where(mask,torch.randn_like(batch.jammer),torch.zeros_like(batch.jammer));target=batch.jammer.abs().square().mean();jam=jam*(target/jam.abs().square().mean().clamp_min(1e-12)).sqrt();return replace(batch,jammer=jam,jammer_mask=mask)
def barrage(batch):
 mask=torch.ones_like(batch.jammer_mask);jam=torch.randn_like(batch.jammer);target=batch.jammer.abs().square().mean();jam=jam*(target/jam.abs().square().mean().clamp_min(1e-12)).sqrt();return replace(batch,jammer=jam,jammer_mask=mask,jammer_type='barrage')
def nojam(batch):return replace(batch,jammer=torch.zeros_like(batch.jammer),jammer_mask=torch.zeros_like(batch.jammer_mask))
def run(codec,model,batch,target,equalizer):
 s,g=state(model,target);r=run_mode_on_paired_batch(codec,model,batch,s,g,equalizer=equalizer,fading='multipath_block',channel_estimator='dft_tap_ls',estimator_num_taps=6,receiver_state_mode='observable_v1');m=summarize_layer_groups(exposure_metric_summary(r['reconstruction'],target,group='unseen_speaker'));o={}
 for p,v in m.items():
  for k,x in v.items():
   if isinstance(x,(int,float)):o[f'{p}_{k}']=x
 o.update(csi_nmse=float(r['csi_nmse'].mean()),pilot_evm=float(r['pilot_evm'].mean()),post_equalization_sinr_linear=float(r['post_equalization_sinr'].mean()),post_equalization_sinr_db=float(10*torch.log10(r['post_equalization_sinr'].clamp_min(1e-12)).mean()),**r['equalizer_gain_statistics']);return o
def main():
 a=parse();dist=json.loads(Path(a.selected_distribution).read_text());condition=dist['strongest_selected_condition'];n=a.unseen_utterances*a.realizations_per_utterance*6*2
 if a.dry_run:print(json.dumps({'dry_run':True,'evaluations':n,'condition':condition,'output_dir':a.output_dir},indent=2));return
 if n>20 and not a.allow_long_run:raise SystemExit('long J5 paired evaluation requires --allow-long-run')
 out=Path(a.output_dir)
 if out.exists() and not a.overwrite:raise SystemExit(f'refusing existing output directory: {out}')
 verify_j4_accepted(a.j4_manifest,a.j4_checkpoint);cfg=load_config(a.config);cfg['device']=a.device;dev=resolve_device(a.device);codec,_=build_components(cfg,dev);codec.eval();[p.requires_grad_(False) for p in codec.parameters()];models={}
 for label,path in {'j4':a.j4_checkpoint,'j5':a.j5_checkpoint}.items():
  _,model=build_components(cfg,dev);model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'],strict=True);model.eval();models[label]=model
 subsets=build_content_subsets(Path(cfg['data']['train_manifest']),Path(cfg['data']['valid_manifest']),Path(cfg['data']['latent_cache_dir']),seed=a.seed,validation_items_per_group=max(128,a.unseen_utterances))['subsets']['256'];ids=subsets['unseen_speaker_ids'][:a.unseen_utterances];source=RepresentationSource(cfg,codec,dev,'val');rows=[]
 with torch.no_grad():
  for identifier in ids:
   target,wave=_find(source,identifier)
   for realization in range(a.realizations_per_utterance):
    seed=int(hashlib.sha256(f'j5_final|{a.seed}|{identifier}|{realization}'.encode()).hexdigest()[:8],16);base=_make_batch(codec,next(iter(models.values())),cfg,target=target,waveform=wave,snr_db=condition['snr_db'],jsr_db=0,jammer_type='pilot',seed=seed,device=dev);pilot=normalize_pilot_local_batch(base,condition['pilot_jsr_db'],condition['coverage']);modes={'pilot_estimated':(pilot,'estimated'),'pilot_oracle':(pilot,'oracle'),'no_jammer':(nojam(pilot),'estimated'),'data_only_matched':(data_only(pilot),'estimated'),'barrage_matched':(barrage(pilot),'estimated')}
    for checkpoint,model in models.items():
     for mode,(batch,eq) in modes.items():rows.append({'checkpoint':checkpoint,'mode':mode,'utterance_id':identifier,'realization':realization,'seed':seed,**run(codec,model,batch,target,eq)})
 groups={}
 for row in rows:groups.setdefault((row['checkpoint'],row['mode']),[]).append(row)
 aggregate=[]
 for (cp,mode),members in groups.items():
  item={'checkpoint':cp,'mode':mode,'n':len(members),'utterances':len({x['utterance_id'] for x in members})};keys=[k for k,v in members[0].items() if isinstance(v,(int,float)) and k not in {'seed','realization'}]
  for key in keys:
   for stat,value in distribution([x[key] for x in members]).items():item[f'{key}_{stat}']=value
  item['layer7_failure_rates']=summarize_failure_rates([{'utterance_id':x['utterance_id'],'layer7_improvement':x['layer7_relative_improvement_over_zero']} for x in members]);aggregate.append(item)
 out.mkdir(parents=True);fields=sorted({k for x in rows for k in x});
 with (out/'realizations.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 (out/'summary.json').write_text(json.dumps({'version':'j5_final_paired_v1','condition':condition,'checkpoint_hashes':{'j4':file_sha256(a.j4_checkpoint),'j5':file_sha256(a.j5_checkpoint)},'aggregate':aggregate},indent=2));print(json.dumps({'output_dir':str(out),'realizations':len(rows)},indent=2))
if __name__=='__main__':main()
