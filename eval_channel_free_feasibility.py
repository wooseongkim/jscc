from __future__ import annotations
import argparse,csv,json,shutil
from pathlib import Path
import torch
from scipy.io import wavfile
from speech_jscc.config import load_config,resolve_device
from speech_jscc.experiment import build_components
from speech_jscc.data import resolve_waveform_splits,load_waveform_segment
from speech_jscc.diagnostics.architecture_screening import direct_bypass
from speech_jscc.diagnostics.content_generalization import parse_speaker_id
from speech_jscc.diagnostics.g0_exposure import exposure_metric_summary
from src.evaluation.clean_end_to_end import summed_latent_metrics,relative_waveform_metrics
from src.evaluation.waveform_metrics import waveform_metrics
from speech_jscc.training.channel_free_feasibility import feasibility_classification,configure_bottleneck,select_unseen_speaker_paths

def parse():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--experiment-a-best-latent',required=True);p.add_argument('--experiment-a-best-waveform',required=True);p.add_argument('--experiment-b-best-waveform',required=True);p.add_argument('--experiment-c-best-waveform');p.add_argument('--output-dir',required=True);p.add_argument('--device',default='cuda');p.add_argument('--samples',type=int,default=64);p.add_argument('--seed',type=int,default=23);p.add_argument('--overwrite',action='store_true');p.add_argument('--allow-long-run',action='store_true');p.add_argument('--dry-run',action='store_true');return p.parse_args()

def prepare_output_directory(out: Path, overwrite: bool) -> Path:
 if out.exists():
  if not overwrite:raise SystemExit(f'refusing existing output directory: {out}')
  shutil.rmtree(out)
 out.mkdir(parents=True)
 audio=out/'waveform_examples';audio.mkdir()
 return audio

def main():
 a=parse();paths={'experiment_a_best_latent':a.experiment_a_best_latent,'experiment_a_best_waveform':a.experiment_a_best_waveform,'experiment_b_best_waveform':a.experiment_b_best_waveform};
 if a.experiment_c_best_waveform:paths['experiment_c_best_waveform']=a.experiment_c_best_waveform
 if a.dry_run:print(json.dumps({'dry_run':True,'samples':a.samples,'checkpoints':paths,'latent_cache':False,'output_dir':a.output_dir},indent=2));return
 if a.samples>2 and not a.allow_long_run:raise SystemExit('full feasibility evaluation requires --allow-long-run')
 out=Path(a.output_dir)
 cfg=load_config(a.config)
 if 'latent_cache_dir' in cfg.get('data',{}):raise SystemExit('latent cache is forbidden')
 cfg['device']=a.device;device=resolve_device(a.device);codec,_=build_components(cfg,device);codec.eval();train,val=resolve_waveform_splits(cfg['data'],a.seed);import random;random.Random(a.seed).shuffle(train);train=train[:256];validation=select_unseen_speaker_paths(train,val,limit=a.samples,seed=a.seed+1);models={}
 for label,path in paths.items():
  cp=torch.load(path,map_location='cpu',weights_only=False);uses=int(cp.get('channel_uses',1920));local=configure_bottleneck(cfg,uses);_,model=build_components(local,device);model.load_state_dict(cp['model'],strict=True);model.eval();models[label]=model
 audio=prepare_output_directory(out,a.overwrite);rows=[];latent_store={k:([],[]) for k in models};sr=int(cfg['codec']['sample_rate'])
 with torch.no_grad():
  for index,path in enumerate(validation):
   wave=load_waveform_segment(path,sr,int(cfg['codec']['waveform_samples'])).unsqueeze(0).to(device);target=codec.encode_waveform(wave);clean=codec.decode_representation(target);clean_m=waveform_metrics(wave,clean,sr);rows.append({'utterance_id':str(path),'mode':'clean_codec',**clean_m})
   if index<8:wavfile.write(audio/f'{index:03d}_clean_codec.wav',sr,clean.squeeze().cpu().numpy().astype('float32'))
   for label,model in models.items():
    state=torch.zeros(1,model.encoder.channel_state_dim,device=device);recon=direct_bypass(model,target,state);decoded=codec.decode_representation(recon);current=waveform_metrics(wave,decoded,sr);latent=exposure_metric_summary(recon,target,group='unseen_speaker');summed=summed_latent_metrics(recon,target);rows.append({'utterance_id':str(path),'mode':label,**current,**relative_waveform_metrics(current,clean_m),'latent_loss':latent['aggregate']['normalized_mse'],'layer7_loss':latent['per_layer'][7]['normalized_mse'],**{f'summed_{k}':v for k,v in summed.items()}});latent_store[label][0].append(recon);latent_store[label][1].append(target)
    if index<8:wavfile.write(audio/f'{index:03d}_{label}.wav',sr,decoded.squeeze().cpu().numpy().astype('float32'))
 fields=sorted({k for r in rows for k in r});
 with (out/'utterances.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 aggregate={}
 clean_rows=[r for r in rows if r['mode']=='clean_codec'];aggregate['clean_codec']={'si_sdr_db':sum(r['si_sdr_db'] for r in clean_rows)/len(clean_rows),'waveform_snr_db':sum(r['waveform_snr_db'] for r in clean_rows)/len(clean_rows),'stft_l1':sum(r['stft_l1'] for r in clean_rows)/len(clean_rows)}
 for label,(recons,targets) in latent_store.items():
  members=[r for r in rows if r['mode']==label];metric=exposure_metric_summary(torch.cat(recons),torch.cat(targets),group='unseen_speaker');dsi=sum(r['delta_si_sdr_db'] for r in members)/len(members);dsnr=sum(r['delta_waveform_snr_db'] for r in members)/len(members);ratio=sum(r['stft_ratio'] for r in members)/len(members);aggregate[label]={'si_sdr_db':sum(r['si_sdr_db'] for r in members)/len(members),'delta_si_sdr_db':dsi,'delta_waveform_snr_db':dsnr,'stft_ratio':ratio,'classification':feasibility_classification(delta_si_sdr=dsi,delta_waveform_snr=dsnr,stft_ratio=ratio),'aggregate_latent':metric['aggregate'],'per_layer':metric['per_layer'],'summed_latent':summed_latent_metrics(torch.cat(recons),torch.cat(targets))}
 preferred='experiment_b_best_waveform' if 'experiment_b_best_waveform' in aggregate else 'experiment_a_best_waveform';final=aggregate[preferred]['classification'];summary={'samples':len(validation),'latent_cache_used':False,'aggregate':aggregate,'final_classification':final,'design_recommendation':'keep' if final=='CHANNEL_FREE_FEASIBLE' else 'reconsider' if final=='CHANNEL_FREE_INFEASIBLE' else 'marginal'};(out/'summary.json').write_text(json.dumps(summary,indent=2));baseline=out.parent/'codec_baseline';baseline.mkdir(exist_ok=True);(baseline/'summary.json').write_text(json.dumps({'samples':len(validation),**aggregate['clean_codec']},indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
