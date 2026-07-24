from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import torch,yaml
from speech_jscc.config import load_config,resolve_device
from speech_jscc.experiment import build_components
from speech_jscc.diagnostics.conv_conformer_integration import forward_integration_path
from speech_jscc.diagnostics.g0_exposure import exposure_metric_summary
from speech_jscc.diagnostics.j2_barrage import aggregate_realizations,derive_sweep_seed
from speech_jscc.diagnostics.j4_burst import J4_FRACTIONS,J4_JSR_GRID,J4_SNR_GRID,J4_VERSION,burst_diagnostics,tail_statistics,verify_j3_accepted
from speech_jscc.diagnostics.j3_narrowband import sinr_fields
from speech_jscc.diagnostics.o5_root_cause import stable_tensor_hash
from train_latent_jscc import RepresentationSource
from train_stage1_fixed_tx import _make_batch
from diagnose_j2_barrage_boundary import GROUPS,_find,_source,_flatten_metrics,_heatmaps,_write_csv

METRICS=("aggregate_normalized_mse","aggregate_relative_improvement_over_zero","aggregate_pearson_correlation","aggregate_power_ratio","layers1_to_7_relative_improvement_over_zero","layers6_to_7_relative_improvement_over_zero","layer7_relative_improvement_over_zero","layer7_pearson_correlation","layer7_power_ratio","post_equalization_sinr_linear","post_equalization_sinr_db","csi_nmse","pilot_evm","maximum_equalizer_gain","pilot_overlap_fraction","data_overlap_fraction","realized_received_global_jsr_db","realized_received_active_window_jsr_db")
TAIL=("aggregate_relative_improvement_over_zero","layers1_to_7_relative_improvement_over_zero","layers6_to_7_relative_improvement_over_zero","layer7_relative_improvement_over_zero","layer7_pearson_correlation","layer7_power_ratio")
def values(text):return [float(x) for x in text.split(',')]
def parse_args():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--j3-manifest',required=True);p.add_argument('--j3-checkpoint',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=23);p.add_argument('--snr-values',default=','.join(map(str,J4_SNR_GRID)));p.add_argument('--jsr-values',default=','.join(map(str,J4_JSR_GRID)));p.add_argument('--fractions',default=','.join(map(str,J4_FRACTIONS)));p.add_argument('--realizations',type=int,default=16);p.add_argument('--allow-long-run',action='store_true');p.add_argument('--overwrite',action='store_true');p.add_argument('--dry-run',action='store_true');return p.parse_args()
def channel(result,batch,fraction,jsr):
 d=burst_diagnostics(result['transmitted'],batch.jammer,batch.jammer_mask,batch.pilot_mask,requested_fraction=fraction,requested_global_jsr_db=jsr,faded_signal=result['faded_signal'],faded_jammer=result['faded_jammer']);d.update(sinr_fields(result['post_equalization_sinr']))
 d.update({'requested_snr_db':float(batch.snr_db.detach().mean()),'realized_received_snr_db':float((10*torch.log10(result['faded_signal'].abs().square().mean()/result['noise'].abs().square().mean().clamp_min(1e-12))).detach()),'csi_nmse':float(result['csi_nmse'].detach().mean()),'pilot_evm':float(result['pilot_evm'].detach().mean()),'maximum_equalizer_gain':float(result['estimated_channel'].detach().abs().clamp_min(1e-8).reciprocal().max()),'legitimate_received_power':float(result['faded_signal'].detach().abs().square().mean()),'jammer_received_power':float(result['faded_jammer'].detach().abs().square().mean()),'awgn_power':float(result['noise'].detach().abs().square().mean()),'finite':all(bool(torch.isfinite(v).all()) for v in result.values() if isinstance(v,torch.Tensor))});return d
def aggregate(rows,fractions):
 output=[]
 for fraction in fractions:
  part=[r for r in rows if r['requested_fraction']==fraction];items=aggregate_realizations(part,METRICS)
  for item in items:
   members=[r for r in part if r['group']==item['group'] and r['snr_db']==item['snr_db'] and r['jsr_db']==item['jsr_db']];item['requested_fraction']=fraction
   for metric in TAIL:
    stats=tail_statistics([r[metric] for r in members]);item.update({f'{metric}_{k}':v for k,v in stats.items()})
  output+=items
 return output
def select_distribution(rows):
 unseen=[r for r in rows if r['group']==GROUPS[-1]];passing=[r for r in unseen if float(r['aggregate_relative_improvement_over_zero_mean'])>=.1 and float(r['layers1_to_7_relative_improvement_over_zero_mean'])>=.075];near=[r for r in unseen if float(r['aggregate_relative_improvement_over_zero_mean'])>0 and .05<=float(r['layers1_to_7_relative_improvement_over_zero_mean'])<.075];eligible=passing+near
 if not passing:return {'defined':False,'reason':'no clearly recoverable burst condition'}
 fractions=sorted({float(r['requested_fraction']) for r in eligible})
 if len(fractions)<2:return {'defined':False,'reason':'fewer than two recoverable burst durations'}
 weakest=min(eligible,key=lambda r:float(r['layer7_relative_improvement_over_zero_mean']));snrs=sorted({float(r['snr_db']) for r in eligible});jsrs=sorted({float(r['jsr_db']) for r in eligible})
 all_pass=len(eligible)==len(unseen)
 return {'defined':True,'selected_snr_range_db':[snrs[0],snrs[-1]],'selected_global_jsr_range_db':[jsrs[0],jsrs[-1]],'selected_burst_fractions':fractions,'active_window_jsr_ranges_db':{str(f):[jsrs[0]-10*torch.log10(torch.tensor(f)).item(),jsrs[-1]-10*torch.log10(torch.tensor(f)).item()] for f in fractions},'boundary_evidence':{'passing_conditions':len(passing),'near_boundary_conditions':len(near),'all_conditions_passed':all_pass,'statement':'No failure boundary was identified within the evaluated J4 range.' if all_pass else 'A recoverable-to-near-boundary transition was identified.'},'weakest_metric':'layer7_improvement','weakest_layer':7,'worst_tail_condition':weakest,'pilot_overlap_influence':'see pilot_overlap_vs_loss.csv','mapping_overlap_influence':'see burst_position_vs_per_layer_loss.csv'}
def plots(rows,out):
 import matplotlib.pyplot as plt
 out.mkdir(parents=True,exist_ok=True);unseen=[r for r in rows if r['group']==GROUPS[-1]]
 specs=[('requested_fraction','aggregate_normalized_mse','burst_fraction_vs_loss'),('burst_start_symbol','aggregate_normalized_mse','burst_position_vs_loss'),('pilot_overlap_fraction','aggregate_normalized_mse','pilot_overlap_vs_loss'),('requested_fraction','layers6_to_7_relative_improvement_over_zero','layers6_to_7_improvement'),('requested_fraction','post_equalization_sinr_db','post_eq_sinr'),('requested_fraction','csi_nmse','csi_nmse')]
 for x,y,name in specs:
  fig,ax=plt.subplots();ax.scatter([r[x] for r in unseen],[r[y] for r in unseen],s=8,alpha=.6);ax.set_xlabel(x);ax.set_ylabel(y);fig.tight_layout();fig.savefig(out/f'{name}.png');plt.close(fig)
 values=sorted(float(r['layer7_relative_improvement_over_zero']) for r in unseen);fig,ax=plt.subplots();ax.hist(values,bins=min(20,max(5,len(values)//4)));ax.set_xlabel('Layer 7 improvement');fig.tight_layout();fig.savefig(out/'layer7_improvement_distribution.png');plt.close(fig)
 fig,ax=plt.subplots();ax.plot(range(len(values)),values);ax.axhline(0,color='red');ax.set_ylabel('Layer 7 improvement');fig.tight_layout();fig.savefig(out/'layer7_tail_failure.png');plt.close(fig)
def main():
 a=parse_args();snrs=values(a.snr_values);jsrs=values(a.jsr_values);fractions=values(a.fractions);count=len(snrs)*len(jsrs)*len(fractions)*a.realizations*3
 if a.dry_run:print(json.dumps({'dry_run':True,'evaluations':count,'output_dir':a.output_dir},indent=2));return
 if count>15 and not a.allow_long_run:raise SystemExit('long boundary sweep requires --allow-long-run')
 out=Path(a.output_dir)
 if out.exists() and not a.overwrite:raise SystemExit(f'refusing existing output directory: {out}')
 out.mkdir(parents=True,exist_ok=True);accepted=verify_j3_accepted(a.j3_manifest,a.j3_checkpoint);config=load_config(a.config);config['device']=a.device;device=resolve_device(a.device);codec,model=build_components(config,device);codec.eval();[p.requires_grad_(False) for p in codec.parameters()]
 cp=torch.load(a.j3_checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(cp['model'],strict=True);model.eval();sources={'train':RepresentationSource(config,codec,device,'train'),'val':RepresentationSource(config,codec,device,'val')};suite=json.load(open(Path(accepted['summary_path']).parent/'validation_suite.json'));ids={g:sorted({r['utterance_id'] for r in suite['scenarios'] if r['group']==g}) for g in GROUPS};rows=[]
 with torch.no_grad():
  for snr in snrs:
   for jsr in jsrs:
    for fraction in fractions:
     for realization in range(a.realizations):
      for group in GROUPS:
       identifier=ids[group][realization%len(ids[group])];target,wave=_find(_source(sources,identifier),identifier);seed=derive_sweep_seed(a.seed,snr,jsr,realization,group);cfg=json.loads(json.dumps(config));cfg['channel']['jammed_fraction']=fraction;batch=_make_batch(codec,model,cfg,target=target,waveform=wave,snr_db=snr,jsr_db=jsr,jammer_type='burst',seed=seed,device=device);result=forward_integration_path('g3_random_clean',codec,model,target,cfg,batch=batch);flat=_flatten_metrics(exposure_metric_summary(result['reconstruction'],target,group=group));diag=channel(result,batch,fraction,jsr);rows.append({'snr_db':snr,'jsr_db':jsr,'requested_fraction':fraction,'realization':realization,'group':group,'utterance_id':identifier,'sample_id':f'{group}:snr={snr}:jsr={jsr}:f={fraction}:r={realization}','seed':seed,'mask_hash':stable_tensor_hash(batch.jammer_mask),'legitimate_channel_hash':stable_tensor_hash(batch.signal_fading),'jammer_channel_hash':stable_tensor_hash(batch.jammer_fading),'jammer_waveform_hash':stable_tensor_hash(batch.jammer),'awgn_hash':stable_tensor_hash(batch.noise),**flat,**diag})
 aggregated=aggregate(rows,fractions);selection=select_distribution(aggregated);_write_csv(out/'realizations.csv',rows);_write_csv(out/'aggregated.csv',aggregated);_write_csv(out/'global_jsr_vs_active_window_jsr.csv',[{'global_jsr_db':j,'burst_fraction':f,'active_window_jsr_db':j-10*torch.log10(torch.tensor(f)).item()} for j in jsrs for f in fractions]);_write_csv(out/'pilot_overlap_vs_loss.csv',[{'sample_id':r['sample_id'],'pilot_overlap_fraction':r['pilot_overlap_fraction'],'loss':r['aggregate_normalized_mse']} for r in rows]);_write_csv(out/'burst_position_vs_per_layer_loss.csv',[{'sample_id':r['sample_id'],'burst_start_symbol':r['burst_start_symbol'],**{f'layer{i}_loss':r[f'layer{i}_normalized_mse'] for i in range(8)}} for r in rows]);plots(rows,out/'plots')
 for f in fractions:_heatmaps([r for r in aggregated if r['requested_fraction']==f],out/'plots'/f'fraction_{f:g}')
 worst=sorted(rows,key=lambda r:r['layer7_relative_improvement_over_zero'])[:max(1,len(rows)//20)];hashes={k:len({r[k] for r in rows}) for k in ('mask_hash','legitimate_channel_hash','jammer_channel_hash','jammer_waveform_hash','awgn_hash')};mask_report={'all_contiguous':all(r['contiguous_burst_verified'] for r in rows),'all_full_band':all(r['full_band_inside_burst_verified'] for r in rows),'no_wraparound':all(not r['wraparound_used'] for r in rows),'maximum_leakage':max(r['leakage_power_outside_burst'] for r in rows),'unique_mask_hashes':hashes['mask_hash']};summary={'version':J4_VERSION,'accepted_j3':accepted,'grid':{'snr_db':snrs,'global_jsr_db':jsrs,'burst_fractions':fractions,'realizations':a.realizations},'row_count':len(rows),'fraction_nonfinite':sum(not r['finite'] for r in rows)/len(rows),'unique_hashes':hashes,'mask_validation':mask_report,'worst_realizations':[{k:r[k] for k in ('sample_id','utterance_id','seed','layer7_relative_improvement_over_zero','mask_hash','legitimate_channel_hash','jammer_channel_hash','jammer_waveform_hash','awgn_hash')} for r in worst],'selected_training_distribution':selection};(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'stochastic_hash_report.json').write_text(json.dumps(hashes,indent=2));(out/'mask_validation_report.json').write_text(json.dumps(mask_report,indent=2));(out/'worst_realization_report.json').write_text(json.dumps(summary['worst_realizations'],indent=2));(out/'selected_training_distribution.json').write_text(json.dumps(selection,indent=2));(out.parent/'selected_training_distribution.json').write_text(json.dumps(selection,indent=2));(out/'resolved_config.yaml').write_text(yaml.safe_dump(config));print(json.dumps({'rows':len(rows),'selection':selection},indent=2))
if __name__=='__main__':main()
