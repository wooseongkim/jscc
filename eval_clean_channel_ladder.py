from __future__ import annotations
import argparse,csv,json
from dataclasses import replace
from pathlib import Path
import torch
from evaluation.paired import run_mode_on_paired_batch
from speech_jscc.config import load_config,resolve_device
from speech_jscc.experiment import build_components
from speech_jscc.diagnostics.content_generalization import build_content_subsets
from speech_jscc.diagnostics.architecture_screening import direct_bypass
from speech_jscc.diagnostics.g0_exposure import exposure_metric_summary
from src.evaluation.clean_end_to_end import clean_ladder_conditions,neutral_observable_state,relative_waveform_metrics
from src.evaluation.waveform_metrics import waveform_metrics
from train_latent_jscc import RepresentationSource
from train_stage1_fixed_tx import _make_batch
from diagnose_j2_barrage_boundary import _find

def parse():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--device',default='cuda');p.add_argument('--samples',type=int,default=64);p.add_argument('--seed',type=int,default=23);p.add_argument('--overwrite',action='store_true');p.add_argument('--allow-long-run',action='store_true');p.add_argument('--dry-run',action='store_true');return p.parse_args()
def main():
 a=parse();conditions=clean_ladder_conditions([30,20,15,10,5],seed=a.seed)
 if a.dry_run:print(json.dumps({'dry_run':True,'samples':a.samples,'conditions':conditions,'output_dir':a.output_dir},indent=2));return
 if a.samples>2 and not a.allow_long_run:raise SystemExit('full clean ladder requires --allow-long-run')
 out=Path(a.output_dir)
 if out.exists() and not a.overwrite:raise SystemExit(f'refusing existing output directory: {out}')
 cfg=load_config(a.config);cfg['device']=a.device;dev=resolve_device(a.device);codec,model=build_components(cfg,dev);codec.eval();model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model'],strict=True);model.eval();subset=build_content_subsets(Path(cfg['data']['train_manifest']),Path(cfg['data']['valid_manifest']),Path(cfg['data']['latent_cache_dir']),seed=a.seed,validation_items_per_group=max(64,a.samples))['subsets']['256'];source=RepresentationSource(cfg,codec,dev,'val');rows=[];sr=int(cfg['codec']['sample_rate'])
 with torch.no_grad():
  for identifier in subset['unseen_speaker_ids'][:a.samples]:
   target,wave=_find(source,identifier);clean=codec.decode_representation(target);clean_m=waveform_metrics(wave,clean,sr);state=neutral_observable_state(1,device=dev,dtype=target.dtype);gates=torch.ones(1,8,device=dev)
   for c in conditions:
    if c['stage']=='C0':recon=direct_bypass(model,target,state);channel={}
    else:
     snr=c.get('snr_db',120.);batch=_make_batch(codec,model,cfg,target=target,waveform=wave,snr_db=snr,jsr_db=-120,jammer_type='barrage',seed=c['seed'],device=dev);batch=replace(batch,jammer=torch.zeros_like(batch.jammer),jammer_mask=torch.zeros_like(batch.jammer_mask))
     if c['stage'] in ('C1','C2'):batch=replace(batch,signal_fading=torch.ones_like(batch.signal_fading))
     if c['stage']=='C1':batch=replace(batch,noise=torch.zeros_like(batch.noise))
     equalizer='oracle' if c['stage'] in ('C1','C2','C3') else 'estimated';result=run_mode_on_paired_batch(codec,model,batch,state,gates,equalizer=equalizer,fading='multipath_block',channel_estimator='dft_tap_ls',estimator_num_taps=6,receiver_state_mode='observable_v1');recon=result['reconstruction'];channel={'csi_nmse':float(result['csi_nmse'].mean()),'pilot_evm':float(result['pilot_evm'].mean()),'post_equalization_sinr_db':float(10*torch.log10(result['post_equalization_sinr'].clamp_min(1e-12)).mean())}
    latent=exposure_metric_summary(recon,target,group='unseen_speaker')['aggregate'];current=waveform_metrics(wave,codec.decode_representation(recon),sr);rows.append({'utterance_id':identifier,**c,**{f'latent_{k}':v for k,v in latent.items() if isinstance(v,(int,float))},**current,**relative_waveform_metrics(current,clean_m),**channel})
 out.mkdir(parents=True);fields=sorted({k for r in rows for k in r});
 with (out/'realizations.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 stages={}
 for stage in ('C0','C1','C2','C3','C4'):
  members=[r for r in rows if r['stage']==stage];stages[stage]={'n':len(members),'mean_si_sdr_db':sum(r['si_sdr_db'] for r in members)/len(members),'mean_delta_si_sdr_db':sum(r['delta_si_sdr_db'] for r in members)/len(members),'mean_latent_loss':sum(r['latent_normalized_mse'] for r in members)/len(members)}
 summary={'version':'clean_channel_ladder_v1','samples':a.samples,'stages':stages,'first_waveform_degradation_stage':next((k for k,v in stages.items() if v['mean_delta_si_sdr_db']<-3),None)};(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
