from __future__ import annotations
import argparse,csv,hashlib,json,subprocess
from pathlib import Path
import torch
from scipy.io import wavfile
from speech_jscc.config import load_config,resolve_device
from speech_jscc.experiment import build_components
from speech_jscc.diagnostics.content_generalization import build_content_subsets
from speech_jscc.diagnostics.g0_exposure import exposure_metric_summary
from speech_jscc.diagnostics.architecture_screening import direct_bypass
from speech_jscc.diagnostics.j2_barrage import file_sha256,summarize_layer_groups
from src.evaluation.clean_end_to_end import neutral_observable_state,ideal_ofdm_roundtrip,summed_latent_metrics,oracle_layer_replacements,classify_identity
from src.evaluation.waveform_metrics import waveform_metrics
from train_latent_jscc import RepresentationSource
from diagnose_j2_barrage_boundary import _find

def parse():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--j4-checkpoint',required=True);p.add_argument('--j5-checkpoint',required=True);p.add_argument('--output-root',required=True);p.add_argument('--device',default='cuda');p.add_argument('--samples',type=int,default=16);p.add_argument('--seed',type=int,default=23);p.add_argument('--overwrite',action='store_true');p.add_argument('--allow-long-run',action='store_true');p.add_argument('--dry-run',action='store_true');return p.parse_args()
def write_csv(path,rows):
 fields=sorted({k for r in rows for k in r});
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def flatten(prefix,value):
 out={}
 for p,row in summarize_layer_groups(value).items():
  for k,v in row.items():
   if isinstance(v,(int,float)):out[f'{prefix}_{p}_{k}']=v
 return out
def save(path,wave,sr):wavfile.write(path,sr,wave.squeeze().clamp(-1,1).detach().cpu().numpy().astype('float32'))
def main():
 a=parse()
 if a.dry_run:print(json.dumps({'dry_run':True,'phases':['codec_identity','existing_j5_direct_channel_free','existing_j5_ideal_ofdm'],'samples':a.samples,'output_root':a.output_root},indent=2));return
 if a.samples>2 and not a.allow_long_run:raise SystemExit('full clean end-to-end diagnostic requires --allow-long-run')
 root=Path(a.output_root)
 if root.exists() and any(root.iterdir()) and not a.overwrite:raise SystemExit(f'refusing existing output root: {root}')
 cfg=load_config(a.config);cfg['device']=a.device;dev=resolve_device(a.device);codec,model=build_components(cfg,dev);codec.eval();[p.requires_grad_(False) for p in codec.parameters()];checkpoint=torch.load(a.j5_checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['model'],strict=True);model.eval();sr=int(cfg['codec']['sample_rate'])
 root.mkdir(parents=True,exist_ok=True);status={'schema_version':'clean_end_to_end_pre_diagnostic_v1','provisional_latent_result':True,'end_to_end_speech_quality_validated':False,'final_j5_classification_deferred':True,'accepted_j4_checkpoint':str(Path(a.j4_checkpoint).resolve()),'accepted_j4_checkpoint_sha256':file_sha256(a.j4_checkpoint),'j5_final_checkpoint':str(Path(a.j5_checkpoint).resolve()),'j5_final_checkpoint_sha256':file_sha256(a.j5_checkpoint),'git_commit':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip(),'codec_metadata':cfg['codec'],'latent_shape':[8,50,1024],'normalization_configuration':{'preprocessing':'none','loss':cfg['train']['latent_normalization']},'ofdm_configuration':{'grid':[64,32],'pilots':128,'data_resources':1920,'resource_mapping_version':'pilot_reserved_v1'},'pause_reason':'No-jammer waveform SI-SDR remained approximately -18.78 dB; pilot jamming is not the primary waveform failure.'};(root/'pre_diagnostic_status.json').write_text(json.dumps(status,indent=2))
 subsets=build_content_subsets(Path(cfg['data']['train_manifest']),Path(cfg['data']['valid_manifest']),Path(cfg['data']['latent_cache_dir']),seed=a.seed,validation_items_per_group=max(a.samples,16))['subsets']['256'];ids=subsets['unseen_speaker_ids'][:a.samples];source=RepresentationSource(cfg,codec,dev,'val');identity_dir=root/'codec_identity';direct_dir=root/'existing_j5_direct_channel_free';ofdm_dir=root/'existing_j5_ideal_ofdm';wave_dir=root/'waveform_examples'
 for p in (identity_dir,direct_dir,ofdm_dir,wave_dir):p.mkdir(parents=True,exist_ok=True)
 identity=[];direct_rows=[];ofdm_rows=[]
 with torch.no_grad():
  for index,identifier in enumerate(ids):
   cached,wave=_find(source,identifier);encoded=codec.encode_waveform(wave);diff=(encoded-cached).abs();clean=codec.decode_representation(encoded);normalized_roundtrip=encoded.clone();roundtrip=codec.decode_representation(normalized_roundtrip);cached_wave=codec.decode_representation(cached);clean_metrics=waveform_metrics(wave,clean,sr);round_metrics=waveform_metrics(clean,roundtrip,sr);cache_metrics=waveform_metrics(clean,cached_wave,sr);identity.append({'utterance_id':identifier,'waveform_hash':hashlib.sha256(wave.cpu().numpy().tobytes()).hexdigest(),'cached_latent_hash':hashlib.sha256(cached.cpu().numpy().tobytes()).hexdigest(),'direct_latent_hash':hashlib.sha256(encoded.cpu().numpy().tobytes()).hexdigest(),'latent_shape':str(list(encoded.shape[1:])),'normalization_max_abs_error':0.0,'normalization_mean_abs_error':0.0,'cached_direct_max_abs_error':float(diff.max()),'cached_direct_mean_abs_error':float(diff.mean()),**{f'cached_direct_layer_{layer}_max_abs_error':float(diff[:,layer].max()) for layer in range(8)},**{f'clean_{k}':v for k,v in clean_metrics.items() if isinstance(v,(int,float))},**{f'roundtrip_vs_clean_{k}':v for k,v in round_metrics.items() if isinstance(v,(int,float))},**{f'cached_vs_clean_{k}':v for k,v in cache_metrics.items() if isinstance(v,(int,float))}})
   state=neutral_observable_state(1,device=dev,dtype=cached.dtype);recon=direct_bypass(model,cached,state);metric=exposure_metric_summary(recon,cached,group='unseen_speaker');decoded=codec.decode_representation(recon);direct_rows.append({'utterance_id':identifier,**flatten('latent',metric),**{f'summed_{k}':v for k,v in summed_latent_metrics(recon,cached).items()},**waveform_metrics(wave,decoded,sr)});symbols=model.encoder(cached,state);rt=ideal_ofdm_roundtrip(symbols);ofdm_recon=model.decoder(rt['recovered'],state);ofdm_metric=exposure_metric_summary(ofdm_recon,cached,group='unseen_speaker');ofdm_wave=codec.decode_representation(ofdm_recon);ofdm_rows.append({'utterance_id':identifier,'encoder_symbol_count':symbols.shape[1],'packed_grid_shape':str(list(rt['grid'].shape[1:])),'pilot_count':rt['pilot_count'],'data_count':rt['data_count'],'data_symbol_recovery_max_error':rt['max_recovery_error'],'pilot_leakage':rt['pilot_leakage'],**flatten('latent',ofdm_metric),**{f'summed_{k}':v for k,v in summed_latent_metrics(ofdm_recon,cached).items()},**waveform_metrics(wave,ofdm_wave,sr)});save(wave_dir/f'{index:03d}_source.wav',wave,sr);save(wave_dir/f'{index:03d}_clean_codec.wav',clean,sr);save(wave_dir/f'{index:03d}_j5_direct.wav',decoded,sr);save(wave_dir/f'{index:03d}_j5_ideal_ofdm.wav',ofdm_wave,sr)
 write_csv(identity_dir/'utterances.csv',identity);write_csv(direct_dir/'utterances.csv',direct_rows);write_csv(ofdm_dir/'utterances.csv',ofdm_rows)
 identity_result=classify_identity(normalization_max_error=max(r['normalization_max_abs_error'] for r in identity),cached_direct_max_error=max(r['cached_direct_max_abs_error'] for r in identity),codec_baseline_reproduced=0<=sum(r['clean_si_sdr_db'] for r in identity)/len(identity)<=10);identity_pass=identity_result['passed'] and all(r['latent_shape']=='[8, 50, 1024]' for r in identity);resource_pass=max(r['data_symbol_recovery_max_error'] for r in ofdm_rows)<=1e-5 and max(r['pilot_leakage'] for r in ofdm_rows)==0
 summary={'samples':len(ids),'utterance_ids':ids,'codec_identity_pass':identity_pass,'identity_classification':identity_result['classification'],'cache_mismatch_utterances':[r['utterance_id'] for r in identity if r['cached_direct_max_abs_error']>1e-5],'resource_identity_pass':resource_pass,'later_training_blocked':not identity_pass,'neutral_state':neutral_observable_state(1,device='cpu')[0].tolist(),'mean_clean_codec_si_sdr':sum(r['clean_si_sdr_db'] for r in identity)/len(identity),'mean_direct_si_sdr':sum(r['si_sdr_db'] for r in direct_rows)/len(direct_rows),'mean_ideal_ofdm_si_sdr':sum(r['si_sdr_db'] for r in ofdm_rows)/len(ofdm_rows),'mean_direct_latent_loss':sum(r['latent_aggregate_normalized_mse'] for r in direct_rows)/len(direct_rows),'mean_ideal_ofdm_latent_loss':sum(r['latent_aggregate_normalized_mse'] for r in ofdm_rows)/len(ofdm_rows),'classification':identity_result['classification'] if not identity_pass else 'EXISTING_CHECKPOINT_REPRESENTATION_FAILURE' if resource_pass else 'OFDM_RESOURCE_PATH_BUG'}
 for p in (identity_dir,direct_dir,ofdm_dir):(p/'summary.json').write_text(json.dumps(summary,indent=2))
 (root/'root_cause_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
