from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import torch,yaml
from speech_jscc.config import load_config,resolve_device
from speech_jscc.experiment import build_components
from speech_jscc.diagnostics.j5_pilot import verify_j4_accepted,normalize_pilot_local_batch,pilot_jammer_diagnostics
from speech_jscc.diagnostics.g0_exposure import exposure_metric_summary
from speech_jscc.diagnostics.j2_barrage import summarize_layer_groups,derive_sweep_seed
from speech_jscc.diagnostics.j4_tail import distribution
from speech_jscc.diagnostics.o5_root_cause import stable_tensor_hash
from speech_jscc.diagnostics.content_generalization import build_content_subsets
from evaluation.paired import run_mode_on_paired_batch
from train_latent_jscc import RepresentationSource
from train_stage1_fixed_tx import _make_batch
from diagnose_j2_barrage_boundary import _find

GROUPS=('seen_utterance_unseen_channel','same_speaker_unseen_utterance_unseen_channel','unseen_speaker_unseen_utterance_unseen_channel')
def vals(x):return [float(v) for v in x.split(',')]
def find_source(sources, identifier):
 for source in (sources['val'], sources['train']):
  try:return source,_find(source,identifier)
  except ValueError:pass
 raise ValueError(f'utterance not found in train/validation sources: {identifier}')
def parse():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--j4-manifest',required=True);p.add_argument('--j4-checkpoint',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=23);p.add_argument('--snr-values',default='5,10,15');p.add_argument('--pilot-jsr-values',default='-10,-5,0,5,10');p.add_argument('--coverages',default='.25,.5,1');p.add_argument('--realizations',type=int,default=16);p.add_argument('--allow-long-run',action='store_true');p.add_argument('--overwrite',action='store_true');p.add_argument('--dry-run',action='store_true');return p.parse_args()
def csvwrite(p,rows):
 fields=sorted({k for r in rows for k in r});f=p.open('w',newline='');w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows);f.close()
def flat(metric):
 g=summarize_layer_groups(metric);o={}
 for prefix,value in g.items():
  for k,v in value.items():
   if isinstance(v,(int,float)):o[f'{prefix}_{k}']=v
 return o
def plots(rows,out):
 try:import matplotlib.pyplot as plt
 except ImportError:return
 out.mkdir(parents=True,exist_ok=True);unseen=[r for r in rows if r['group']==GROUPS[-1]]
 for x,y,name in [('pilot_jsr_db','csi_nmse','csi_nmse_vs_pilot_jsr'),('pilot_jsr_db','pilot_evm','pilot_evm_vs_pilot_jsr'),('pilot_jsr_db','layer7_relative_improvement_over_zero','layer7_vs_pilot_jsr'),('coverage','aggregate_normalized_mse','coverage_vs_reconstruction')]:
  fig,ax=plt.subplots();ax.scatter([r[x] for r in unseen],[r[y] for r in unseen],s=8,alpha=.4);ax.set_xlabel(x);ax.set_ylabel(y);fig.tight_layout();fig.savefig(out/f'{name}.png');plt.close(fig)
def main():
 a=parse();snrs=vals(a.snr_values);jsrs=vals(a.pilot_jsr_values);covers=vals(a.coverages);count=len(snrs)*len(jsrs)*len(covers)*a.realizations*3
 if a.dry_run:print(json.dumps({'dry_run':True,'evaluations':count,'grid':{'snr':snrs,'pilot_jsr':jsrs,'coverage':covers},'output_dir':a.output_dir},indent=2));return
 if count>15 and not a.allow_long_run:raise SystemExit('long J5 boundary sweep requires --allow-long-run')
 out=Path(a.output_dir)
 if out.exists() and not a.overwrite:raise SystemExit(f'refusing existing output directory: {out}')
 out.mkdir(parents=True,exist_ok=True);accepted=verify_j4_accepted(a.j4_manifest,a.j4_checkpoint);config=load_config(a.config);config['device']=a.device;device=resolve_device(a.device);codec,model=build_components(config,device);codec.eval();[p.requires_grad_(False) for p in codec.parameters()];cp=torch.load(a.j4_checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(cp['model'],strict=True);model.eval()
 subsets=build_content_subsets(Path(config['data']['train_manifest']),Path(config['data']['valid_manifest']),Path(config['data']['latent_cache_dir']),seed=a.seed,validation_items_per_group=16)['subsets']['256'];sources={'train':RepresentationSource(config,codec,device,'train'),'val':RepresentationSource(config,codec,device,'val')};ids={'seen_utterance_unseen_channel':subsets['seen_utterance_ids'],'same_speaker_unseen_utterance_unseen_channel':subsets['same_speaker_unseen_ids'],'unseen_speaker_unseen_utterance_unseen_channel':subsets['unseen_speaker_ids']};rows=[]
 with torch.no_grad():
  for snr in snrs:
   for jsr in jsrs:
    for cover in covers:
     for realization in range(a.realizations):
      for group in GROUPS:
       identifier=ids[group][realization%len(ids[group])];source,(target,wave)=find_source(sources,identifier);seed=derive_sweep_seed(a.seed,snr,jsr,realization,f'{group}|coverage={cover}');base=_make_batch(codec,model,config,target=target,waveform=wave,snr_db=snr,jsr_db=0,jammer_type='pilot',seed=seed,device=device);batch=normalize_pilot_local_batch(base,jsr,cover);state=torch.zeros(1,model.encoder.channel_state_dim,device=device);gates=torch.ones(1,8,device=device);result=run_mode_on_paired_batch(codec,model,batch,state,gates,equalizer='estimated',fading='multipath_block',channel_estimator='dft_tap_ls',estimator_num_taps=6,receiver_state_mode='observable_v1',decode_waveform=False);diag=pilot_jammer_diagnostics(result['transmitted'],batch.jammer,batch.jammer_mask,batch.pilot_mask,requested_pilot_jsr_db=jsr,faded_signal=result['faded_signal'],faded_jammer=result['faded_jammer']);gain=result['equalizer_gain_statistics'];rows.append({'group':group,'utterance_id':identifier,'snr_db':snr,'pilot_jsr_db':jsr,'coverage':cover,'realization':realization,'seed':seed,'mask_hash':stable_tensor_hash(batch.jammer_mask),'channel_hash':stable_tensor_hash(batch.signal_fading),'jammer_channel_hash':stable_tensor_hash(batch.jammer_fading),'jammer_hash':stable_tensor_hash(batch.jammer),'noise_hash':stable_tensor_hash(batch.noise),**flat(exposure_metric_summary(result['reconstruction'],target,group=group)),**diag,**gain,'csi_nmse':float(result['csi_nmse'].mean()),'pilot_evm':float(result['pilot_evm'].mean()),'post_equalization_sinr_linear':float(result['post_equalization_sinr'].mean()),'post_equalization_sinr_db':float(10*torch.log10(result['post_equalization_sinr'].clamp_min(1e-12)).mean())})
 grouped={}
 for r in rows:grouped.setdefault((r['group'],r['snr_db'],r['pilot_jsr_db'],r['coverage']),[]).append(r)
 agg=[]
 for key,m in grouped.items():
  x={'group':key[0],'snr_db':key[1],'pilot_jsr_db':key[2],'coverage':key[3],'n':len(m)}
  for metric in ('aggregate_normalized_mse','aggregate_relative_improvement_over_zero','layers1_to_7_relative_improvement_over_zero','layers6_to_7_relative_improvement_over_zero','layer7_relative_improvement_over_zero','layer7_pearson_correlation','layer7_power_ratio','csi_nmse','pilot_evm','post_equalization_sinr_db'):
   s=distribution([r[metric] for r in m]);x.update({f'{metric}_{k}':v for k,v in s.items()})
  agg.append(x)
 unseen=[x for x in agg if x['group']==GROUPS[-1]];recover=[x for x in unseen if x['aggregate_relative_improvement_over_zero_mean']>0 and x['layers1_to_7_relative_improvement_over_zero_mean']>=.05]
 if recover:
  worst=min(recover,key=lambda x:x['layer7_relative_improvement_over_zero_mean']);selection={'defined':True,'selected_snr_range_db':[min(x['snr_db'] for x in recover),max(x['snr_db'] for x in recover)],'selected_pilot_jsr_range_db':[min(x['pilot_jsr_db'] for x in recover),max(x['pilot_jsr_db'] for x in recover)],'selected_pilot_coverages':sorted({x['coverage'] for x in recover}),'strongest_selected_condition':{'snr_db':worst['snr_db'],'pilot_jsr_db':worst['pilot_jsr_db'],'coverage':worst['coverage']},'boundary_evidence':worst}
 else:selection={'defined':False,'reason':'no recoverable pilot-jammer transition'}
 utterance=[]
 for key,members in {(r['group'],r['utterance_id']):[x for x in rows if x['group']==r['group'] and x['utterance_id']==r['utterance_id']] for r in rows}.items():utterance.append({'group':key[0],'utterance_id':key[1],'realizations':len(members),'aggregate_loss_mean':sum(x['aggregate_normalized_mse'] for x in members)/len(members),'layer7_improvement_mean':sum(x['layer7_relative_improvement_over_zero'] for x in members)/len(members)})
 stochastic={key:len({r[key] for r in rows}) for key in ('mask_hash','channel_hash','jammer_channel_hash','jammer_hash','noise_hash')}
 mask_report={'all_masks_subset_of_pilots':all(r['jammer_leakage_power_on_data_resources']<=1e-12 for r in rows),'coverage_values':sorted({r['attacked_pilot_fraction'] for r in rows}),'maximum_data_leakage_power':max(r['jammer_leakage_power_on_data_resources'] for r in rows)}
 csvwrite(out/'realizations.csv',rows);csvwrite(out/'utterances.csv',utterance);csvwrite(out/'aggregated.csv',agg);(out/'summary.json').write_text(json.dumps({'version':'j5_pilot_boundary_v1','accepted_j4':accepted,'grid':{'snr':snrs,'pilot_jsr':jsrs,'coverage':covers,'realizations':a.realizations},'rows':len(rows),'selection':selection,'stochastic_diversity':stochastic,'mask_validation':mask_report},indent=2));(out/'stochastic_hash_report.json').write_text(json.dumps(stochastic,indent=2));(out/'pilot_mask_validation_report.json').write_text(json.dumps(mask_report,indent=2));(out/'selected_training_distribution.json').write_text(json.dumps(selection,indent=2));(out.parent/'selected_training_distribution.json').write_text(json.dumps(selection,indent=2));(out/'resolved_config.yaml').write_text(yaml.safe_dump(config));plots(rows,out/'plots');print(json.dumps(selection,indent=2))
if __name__=='__main__':main()
