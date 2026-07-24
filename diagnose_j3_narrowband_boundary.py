from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import torch,yaml
from speech_jscc.config import load_config,resolve_device
from speech_jscc.experiment import build_components
from speech_jscc.diagnostics.conv_conformer_integration import forward_integration_path
from speech_jscc.diagnostics.g0_exposure import exposure_metric_summary
from speech_jscc.diagnostics.j2_barrage import aggregate_realizations,derive_sweep_seed,summarize_layer_groups
from speech_jscc.diagnostics.j3_narrowband import J3_FRACTIONS,J3_JSR_GRID,J3_SNR_GRID,J3_VERSION,narrowband_diagnostics,sinr_fields,verify_j2_artifact
from speech_jscc.diagnostics.o5_root_cause import stable_tensor_hash
from train_latent_jscc import RepresentationSource
from train_stage1_fixed_tx import _make_batch
from diagnose_j2_barrage_boundary import GROUPS,_find,_source,_flatten_metrics,_heatmaps,_write_csv

METRICS=("aggregate_normalized_mse","aggregate_raw_mse","aggregate_relative_improvement_over_zero","aggregate_cosine_similarity","aggregate_pearson_correlation","aggregate_power_ratio","aggregate_optimal_scalar_rescaled_normalized_loss","layers1_to_7_relative_improvement_over_zero","layers6_to_7_relative_improvement_over_zero","layer7_relative_improvement_over_zero","post_equalization_sinr_linear","post_equalization_sinr_db","csi_nmse","pilot_evm","maximum_equalizer_gain","pilot_resource_overlap_ratio","data_resource_overlap_ratio","realized_received_global_jsr_db","realized_received_inband_jsr_db","jammed_data_symbol_mse","unjammed_data_symbol_mse")+tuple(f"layer{layer}_{metric}" for layer in range(8) for metric in ("normalized_mse","relative_improvement_over_zero","pearson_correlation","cosine_similarity","power_ratio"))
def values(text):return [float(x) for x in text.split(',')]
def args():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--j2-summary',required=True);p.add_argument('--j2-checkpoint',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=23);p.add_argument('--snr-values',default=','.join(map(str,J3_SNR_GRID)));p.add_argument('--jsr-values',default=','.join(map(str,J3_JSR_GRID)));p.add_argument('--fractions',default=','.join(map(str,J3_FRACTIONS)));p.add_argument('--realizations',type=int,default=16);p.add_argument('--allow-long-run',action='store_true');p.add_argument('--overwrite',action='store_true');p.add_argument('--dry-run',action='store_true');return p.parse_args()
def channel(result,batch,fraction,jsr):
 d=narrowband_diagnostics(result['transmitted'],batch.jammer,batch.jammer_mask,batch.pilot_mask,requested_fraction=fraction,requested_global_jsr_db=jsr,faded_signal=result['faded_signal'],faded_jammer=result['faded_jammer']);d.update(sinr_fields(result['post_equalization_sinr']))
 data=~batch.pilot_mask;jammed=data&batch.jammer_mask;clean=data&~batch.jammer_mask;error=(result['equalized_estimated']-result['transmitted']).abs().square()
 d.update({'requested_snr_db':float(batch.snr_db.detach().mean()),'realized_received_snr_db':float((10*torch.log10(result['faded_signal'].abs().square().mean()/result['noise'].abs().square().mean().clamp_min(1e-12))).detach()), 'csi_nmse':float(result['csi_nmse'].detach().mean()),'pilot_evm':float(result['pilot_evm'].detach().mean()),'maximum_equalizer_gain':float(result['estimated_channel'].detach().abs().clamp_min(1e-8).reciprocal().max()),'legitimate_received_power':float(result['faded_signal'].detach().abs().square().mean()),'jammer_received_power':float(result['faded_jammer'].detach().abs().square().mean()),'awgn_power':float(result['noise'].detach().abs().square().mean()),'jammed_data_symbol_mse':float(error[jammed].detach().mean()),'unjammed_data_symbol_mse':float(error[clean].detach().mean()),'finite':all(bool(torch.isfinite(value).all()) for value in result.values() if isinstance(value,torch.Tensor))});return d
def select_distribution(rows):
 unseen=[r for r in rows if r['group']==GROUPS[-1]]
 passing=[r for r in unseen if float(r['aggregate_relative_improvement_over_zero_mean'])>=.1 and float(r['layers1_to_7_relative_improvement_over_zero_mean'])>=.075]
 near=[r for r in unseen if float(r['aggregate_relative_improvement_over_zero_mean'])>0 and .05<=float(r['layers1_to_7_relative_improvement_over_zero_mean'])<.075]
 eligible=passing+near
 if not passing:return {'defined':False,'reason':'no swept condition clearly passes the primary aggregate and enhancement gates'}
 fractions=sorted({float(r['requested_fraction']) for r in eligible})
 if len(fractions)<2:return {'defined':False,'reason':'boundary evidence supports fewer than two jammer bandwidths','supported_fractions':fractions}
 vulnerable=min(eligible,key=lambda r:float(r['layer7_relative_improvement_over_zero_mean']))
 snrs=sorted({float(r['snr_db']) for r in eligible});jsrs=sorted({float(r['jsr_db']) for r in eligible})
 return {'defined':True,'selected_snr_range_db':[snrs[0],snrs[-1]],'selected_global_jsr_range_db':[jsrs[0],jsrs[-1]],'selected_jammed_subcarrier_fractions':fractions, 'evidence':{'passing_conditions':len(passing),'near_boundary_conditions':len(near),'excluded_unrecoverable_conditions':len(unseen)-len(eligible),'most_vulnerable_selected_condition':vulnerable},'first_vulnerable_metric':'layer7_improvement','first_vulnerable_layer':7,'pilot_overlap_influence':'quantified in pilot_overlap_vs_loss.csv; causal attribution requires paired overlap diagnostics','global_local_jsr_interpretation':'local_inband_jsr_db = global_jsr_db - 10*log10(fraction)'}
def main():
 a=args();snrs=values(a.snr_values);jsrs=values(a.jsr_values);fractions=values(a.fractions);count=len(snrs)*len(jsrs)*len(fractions)*a.realizations*3
 if a.dry_run:print(json.dumps({'dry_run':True,'stage':'j3_random_narrowband_boundary','evaluations':count,'snr':snrs,'global_jsr':jsrs,'fractions':fractions,'output_dir':a.output_dir},indent=2));return
 if count>15 and not a.allow_long_run:raise SystemExit('long boundary sweep requires --allow-long-run')
 out=Path(a.output_dir); 
 if out.exists() and not a.overwrite:raise SystemExit(f'refusing existing output directory: {out}')
 out.mkdir(parents=True,exist_ok=True);accepted=verify_j2_artifact(a.j2_summary,a.j2_checkpoint)
 config=load_config(a.config);config['device']=a.device;device=resolve_device(a.device);codec,model=build_components(config,device);codec.eval();[p.requires_grad_(False) for p in codec.parameters()]
 cp=torch.load(a.j2_checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(cp['model'],strict=True);model.eval();sources={'train':RepresentationSource(config,codec,device,'train'),'val':RepresentationSource(config,codec,device,'val')};suite=json.load(open(Path(a.j2_summary).parent/'validation_suite.json'));ids={g:sorted({r['utterance_id'] for r in suite['scenarios'] if r['group']==g}) for g in GROUPS};rows=[]
 with torch.no_grad():
  for snr in snrs:
   for jsr in jsrs:
    for fraction in fractions:
     for realization in range(a.realizations):
      for group in GROUPS:
       identifier=ids[group][realization%len(ids[group])];target,wave=_find(_source(sources,identifier),identifier);seed=derive_sweep_seed(a.seed,snr,jsr,realization,group);cfg=json.loads(json.dumps(config));cfg['channel']['jammed_fraction']=fraction;batch=_make_batch(codec,model,cfg,target=target,waveform=wave,snr_db=snr,jsr_db=jsr,jammer_type='narrowband',seed=seed,device=device);result=forward_integration_path('g3_random_clean',codec,model,target,cfg,batch=batch);flat=_flatten_metrics(exposure_metric_summary(result['reconstruction'],target,group=group));diag=channel(result,batch,fraction,jsr);rows.append({'snr_db':snr,'jsr_db':jsr,'requested_fraction':fraction,'realization':realization,'group':group,'utterance_id':identifier,'sample_id':f'{group}:snr={snr:g}:jsr={jsr:g}:f={fraction:g}:r={realization}','seed':seed,'mask_hash':stable_tensor_hash(batch.jammer_mask),'legitimate_channel_hash':stable_tensor_hash(batch.signal_fading),'jammer_channel_hash':stable_tensor_hash(batch.jammer_fading),'jammer_waveform_hash':stable_tensor_hash(batch.jammer),'awgn_hash':stable_tensor_hash(batch.noise),**flat,**diag})
 aggregated=[]
 for f in fractions:
  part=[{**r,'group':r['group']} for r in rows if r['requested_fraction']==f];items=aggregate_realizations(part,METRICS)
  for item in items:item['requested_fraction']=f
  aggregated+=items
 selection=select_distribution(aggregated);_write_csv(out/'realizations.csv',rows);_write_csv(out/'aggregated.csv',aggregated)
 for fraction in fractions:_heatmaps([row for row in aggregated if row['requested_fraction']==fraction],out/'plots'/f'fraction_{fraction:g}')
 _write_csv(out/'global_jsr_vs_local_jsr.csv',[{'requested_global_jsr_db':jsr,'jammed_subcarrier_fraction':fraction,'approximate_local_inband_jsr_db':jsr-10*torch.log10(torch.tensor(fraction)).item()} for jsr in jsrs for fraction in fractions])
 _write_csv(out/'pilot_overlap_vs_loss.csv',[{'sample_id':row['sample_id'],'pilot_resource_overlap_ratio':row['pilot_resource_overlap_ratio'],'aggregate_normalized_mse':row['aggregate_normalized_mse']} for row in rows])
 _write_csv(out/'jammer_location_vs_per_layer_loss.csv',[{'sample_id':row['sample_id'],'narrowband_start_index':row['narrowband_start_index'],**{f'layer{i}_normalized_mse':row[f'layer{i}_normalized_mse'] for i in range(8)}} for row in rows])
 worst=sorted(rows,key=lambda row:row['aggregate_normalized_mse'],reverse=True)[:max(1,len(rows)//10)]
 summary={'version':J3_VERSION,'accepted_j2':accepted,'row_count':len(rows),'groups':ids,'grid':{'snr_db':snrs,'requested_global_jsr_db':jsrs,'fractions':fractions,'realizations':a.realizations},'fraction_nonfinite':sum(not row['finite'] for row in rows)/len(rows),'worst_case_samples':[{key:row[key] for key in ('sample_id','utterance_id','seed','snr_db','jsr_db','requested_fraction','aggregate_normalized_mse','layer7_relative_improvement_over_zero','mask_hash','legitimate_channel_hash','jammer_channel_hash','jammer_waveform_hash','awgn_hash')} for row in worst],'selected_training_distribution':selection}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'selected_training_distribution.json').write_text(json.dumps(selection,indent=2));(out.parent/'selected_training_distribution.json').write_text(json.dumps(selection,indent=2));(out/'resolved_config.yaml').write_text(yaml.safe_dump(config));verify_j2_artifact(a.j2_summary,a.j2_checkpoint,expected=accepted);print(json.dumps({'rows':len(rows),'selection':selection},indent=2))
if __name__=='__main__':main()
