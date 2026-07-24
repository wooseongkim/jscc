from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import torch
from scipy.io import wavfile
from evaluation.paired import run_mode_on_paired_batch
from speech_jscc.config import load_config,resolve_device
from speech_jscc.experiment import build_components
from speech_jscc.diagnostics.content_generalization import build_content_subsets
from speech_jscc.diagnostics.j5_pilot import normalize_pilot_local_batch
from speech_jscc.metrics.audio_quality import summarize_audio_metrics,align_waveforms
from train_latent_jscc import RepresentationSource
from train_stage1_fixed_tx import _make_batch
from diagnose_j2_barrage_boundary import _find

def parse():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--j5-checkpoint',required=True);p.add_argument('--selected-distribution',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--device',default='cuda');p.add_argument('--samples',type=int,default=8);p.add_argument('--seed',type=int,default=23);p.add_argument('--allow-long-run',action='store_true');p.add_argument('--overwrite',action='store_true');p.add_argument('--dry-run',action='store_true');return p.parse_args()
def metrics(ref,est,sr):
 ref,est=align_waveforms(ref,est);noise=(ref-est).square().mean(-1);snr=10*torch.log10(ref.square().mean(-1).clamp_min(1e-12)/noise.clamp_min(1e-12));win=torch.hann_window(512,device=ref.device);stft=(torch.stft(ref,512,128,window=win,return_complex=True).abs()-torch.stft(est,512,128,window=win,return_complex=True).abs()).abs().mean();return {'waveform_snr_db':float(snr.mean()),'stft_l1':float(stft),**summarize_audio_metrics(ref,est,sr)}
def main():
 a=parse();dist=json.loads(Path(a.selected_distribution).read_text());condition=dist['strongest_selected_condition']
 if a.dry_run:print(json.dumps({'dry_run':True,'samples':a.samples,'condition':condition,'output_dir':a.output_dir},indent=2));return
 if a.samples>2 and not a.allow_long_run:raise SystemExit('waveform bridge requires --allow-long-run')
 out=Path(a.output_dir)
 if out.exists() and not a.overwrite:raise SystemExit(f'refusing existing output directory: {out}')
 cfg=load_config(a.config);cfg['device']=a.device;dev=resolve_device(a.device);codec,model=build_components(cfg,dev);codec.eval();[p.requires_grad_(False) for p in codec.parameters()];model.load_state_dict(torch.load(a.j5_checkpoint,map_location='cpu',weights_only=False)['model'],strict=True);model.eval();subset=build_content_subsets(Path(cfg['data']['train_manifest']),Path(cfg['data']['valid_manifest']),Path(cfg['data']['latent_cache_dir']),seed=a.seed,validation_items_per_group=max(a.samples,8))['subsets']['256'];source=RepresentationSource(cfg,codec,dev,'val');rows=[];audio=out/'audio_examples';audio.mkdir(parents=True)
 with torch.no_grad():
  for index,identifier in enumerate(subset['unseen_speaker_ids'][:a.samples]):
   target,wave=_find(source,identifier);base=_make_batch(codec,model,cfg,target=target,waveform=wave,snr_db=condition['snr_db'],jsr_db=0,jammer_type='pilot',seed=a.seed+index,device=dev);batch=normalize_pilot_local_batch(base,condition['pilot_jsr_db'],condition['coverage']);state=torch.zeros(1,model.encoder.channel_state_dim,device=dev);gates=torch.ones(1,8,device=dev)
   for mode,eq,current in [('j5_estimated','estimated',batch),('j5_oracle','oracle',batch),('no_jammer','estimated',__import__('dataclasses').replace(batch,jammer=torch.zeros_like(batch.jammer),jammer_mask=torch.zeros_like(batch.jammer_mask)))]:
    result=run_mode_on_paired_batch(codec,model,current,state,gates,equalizer=eq,fading='multipath_block',channel_estimator='dft_tap_ls',estimator_num_taps=6,receiver_state_mode='observable_v1');decoded=codec.decode_representation(result['reconstruction']);row={'utterance_id':identifier,'mode':mode,**metrics(wave,decoded,int(cfg['codec']['sample_rate']))};rows.append(row);wavfile.write(audio/f'{index:03d}_{mode}.wav',int(cfg['codec']['sample_rate']),decoded.squeeze().clamp(-1,1).cpu().numpy().astype('float32'))
   clean=codec.decode_representation(target);rows.append({'utterance_id':identifier,'mode':'clean_codec',**metrics(wave,clean,int(cfg['codec']['sample_rate']))});wavfile.write(audio/f'{index:03d}_clean_codec.wav',int(cfg['codec']['sample_rate']),clean.squeeze().clamp(-1,1).cpu().numpy().astype('float32'))
 out.mkdir(parents=True,exist_ok=True);fields=sorted({k for r in rows for k in r});
 with (out/'waveform_quality.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 optional={'stoi_available':any(r.get('stoi_available') for r in rows),'pesq_available':False,'visqol_available':False,'speaker_embedding_available':False};(out/'summary.json').write_text(json.dumps({'version':'j5_waveform_bridge_v1','condition':condition,'rows':rows,'optional_metric_status':optional,'installation':'pip install pystoi pesq; install ViSQOL separately'},indent=2));print(json.dumps({'output_dir':str(out),'rows':len(rows),'optional_metric_status':optional},indent=2))
if __name__=='__main__':main()
