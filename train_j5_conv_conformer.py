from __future__ import annotations
import argparse,json,sys,platform,subprocess
from pathlib import Path
import math
import torch,yaml
from speech_jscc.config import load_config,resolve_device
from speech_jscc.experiment import build_components
from speech_jscc.diagnostics.j5_pilot import J5_THRESHOLDS,classify_j5,j5_gate,j5_policy,normalize_pilot_local_batch,pilot_jammer_diagnostics,verify_j4_accepted
from speech_jscc.diagnostics.content_generalization import build_content_subsets,build_content_validation_suite
from speech_jscc.diagnostics.conv_conformer_integration import build_j5_validation_suite,forward_integration_path
from speech_jscc.diagnostics.g0_exposure import EpochSampler,exposure_metric_summary,gradient_norms
from speech_jscc.diagnostics.j2_barrage import file_sha256,summarize_layer_groups
from speech_jscc.diagnostics.j4_burst import tail_statistics
from speech_jscc.diagnostics.o5_root_cause import stable_tensor_hash,linear_slope
from train_latent_jscc import RepresentationSource,layer_weighted_latent_mse
from train_stage1_fixed_tx import _make_batch

def parse():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--selected-distribution',required=True);p.add_argument('--j4-manifest',required=True);p.add_argument('--parent-checkpoint',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=23);p.add_argument('--subset-size',default='256');p.add_argument('--steps',type=int,default=4096);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--validation-every',type=int,default=100);p.add_argument('--checkpoint-every',type=int,default=500);p.add_argument('--resume');p.add_argument('--overwrite',action='store_true');p.add_argument('--allow-long-run',action='store_true');p.add_argument('--dry-run',action='store_true');return p.parse_args()
def one(source,identifier):
 i=next(i for i,p in enumerate(source.dataset.paths) if p.as_posix().endswith(identifier));z,w=source.dataset[i];return z.unsqueeze(0),w.unsqueeze(0)
def evaluate(codec,model,config,source,scenarios,device):
 grouped={}
 with torch.no_grad():
  for s in scenarios:
   target,wave=one(source,s['utterance_id']);base=_make_batch(codec,model,config,target=target,waveform=wave,snr_db=s['snr_db'],jsr_db=0,jammer_type='pilot',seed=s['channel_seed'],device=device);batch=normalize_pilot_local_batch(base,s['pilot_jsr_db'],s['coverage']);result=forward_integration_path('g3_random_clean',codec,model,target,config,batch=batch);grouped.setdefault(s['group'],[]).append((result['reconstruction'],target,result,batch))
 out={}
 for group,rows in grouped.items():
  metric=exposure_metric_summary(torch.cat([x[0] for x in rows]),torch.cat([x[1] for x in rows]),group=group);sample=[summarize_layer_groups(exposure_metric_summary(x[0],x[1],group=group)) for x in rows];metric['tail_statistics']={'layer7_improvement':tail_statistics([x['layer7']['relative_improvement_over_zero'] for x in sample])};metric['layers']=summarize_layer_groups(metric);metric['channel_metrics']={'csi_nmse':sum(float(x[2]['csi_nmse'].mean()) for x in rows)/len(rows),'pilot_evm':sum(float(x[2]['pilot_evm'].mean()) for x in rows)/len(rows)};out[group]=metric
 return out
def main():
 a=parse();dist=json.loads(Path(a.selected_distribution).read_text())
 if a.dry_run:print(json.dumps({'dry_run':True,'stage':'j5_pilot_targeted','steps':a.steps,'distribution':dist,'output_dir':a.output_dir},indent=2));return
 if a.steps>5 and not a.allow_long_run:raise SystemExit('long J5 training requires --allow-long-run')
 out=Path(a.output_dir)
 if out.exists() and not (a.overwrite or a.resume):raise SystemExit(f'refusing existing output directory: {out}')
 accepted=verify_j4_accepted(a.j4_manifest,a.parent_checkpoint);config=load_config(a.config);config['device']=a.device;device=resolve_device(a.device);codec,model=build_components(config,device);codec.eval();[p.requires_grad_(False) for p in codec.parameters()];parent=torch.load(a.parent_checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(parent['model'],strict=True);sources={'train':RepresentationSource(config,codec,device,'train'),'val':RepresentationSource(config,codec,device,'val')};subsets=build_content_subsets(Path(config['data']['train_manifest']),Path(config['data']['valid_manifest']),Path(config['data']['latent_cache_dir']),seed=a.seed);subset=subsets['subsets'][a.subset_size];base=build_content_validation_suite(subset,a.seed);suite=build_j5_validation_suite(base,a.seed,dist);optimizer=torch.optim.Adam(model.parameters(),lr=float(config['train']['learning_rate']));weights=torch.ones(8,device=device);start=0;history=[]
 provenance={'diagnostic_stage':'j5_pilot_targeted','stage_definition_version':'j5_pilot_targeted_v1','initialization_mode':'j4_transfer','initialization_source_stage':'j4_random_burst','initial_weights_loaded':True,'parent_checkpoint_path':str(Path(a.parent_checkpoint).resolve()),'parent_checkpoint_sha256':file_sha256(a.parent_checkpoint),'parent_accepted_manifest':str(Path(a.j4_manifest).resolve()),'accepted_j4':accepted,'architecture_version':'conv_conformer_v1','representation_shape':[8,50,1024],'selected_distribution_hash':file_sha256(a.selected_distribution),'validation_suite_hash':suite['validation_suite_hash'],'stage_local_steps':a.steps}
 if a.resume:
  cp=torch.load(a.resume,map_location='cpu',weights_only=False)
  for k in ('diagnostic_stage','parent_checkpoint_sha256','selected_distribution_hash','validation_suite_hash'):
   if cp['provenance'].get(k)!=provenance[k]:raise SystemExit(f'resume mismatch: {k}')
  model.load_state_dict(cp['model'],strict=True);optimizer.load_state_dict(cp['optimizer']);start=cp['step'];history=cp['history']
 out.mkdir(parents=True,exist_ok=True);(out/'resolved_config.yaml').write_text(yaml.safe_dump({'config':config,'distribution':dist,'provenance':provenance}));(out/'validation_suite.json').write_text(json.dumps(suite,indent=2));(out/'command.txt').write_text(' '.join(sys.argv)+'\n');sampler=EpochSampler(subset['train_ids'],batch_size=a.batch_size,seed=a.seed,subset_key='j5_256');hashes={k:set() for k in ('channel','jammer_channel','jammer','noise','mask')};snrs=[];jsrs=[];covers=[]
 with (out/'metrics.jsonl').open('a' if a.resume else 'w') as f:
  for step in range(start+1,a.steps+1):
   steps_per_epoch=math.ceil(len(subset['train_ids'])/a.batch_size);epoch=(step-1)//steps_per_epoch+1;batch_index=(step-1)%steps_per_epoch;ordered=sampler.permutation(epoch);ids=ordered[batch_index*a.batch_size:(batch_index+1)*a.batch_size];pairs=[one(sources['train'],x) for x in ids];target=torch.cat([x[0] for x in pairs]);wave=torch.cat([x[1] for x in pairs]);p=j5_policy(a.seed,step,dist['selected_snr_range_db'],dist['selected_pilot_jsr_range_db'],dist['selected_pilot_coverages']);base_batch=_make_batch(codec,model,config,target=target,waveform=wave,snr_db=p['snr_db'],jsr_db=0,jammer_type='pilot',seed=p['seed'],device=device);batch=normalize_pilot_local_batch(base_batch,p['pilot_jsr_db'],p['coverage']);optimizer.zero_grad(set_to_none=True);result=forward_integration_path('g3_random_clean',codec,model,target,config,batch=batch);loss,_=layer_weighted_latent_mse(result['reconstruction'],target,weights,config['train']['latent_normalization']);loss.backward();grads=gradient_norms(model);torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();metric=exposure_metric_summary(result['reconstruction'].detach(),target,group='train');diag=pilot_jammer_diagnostics(result['transmitted'],batch.jammer,batch.jammer_mask,batch.pilot_mask,requested_pilot_jsr_db=p['pilot_jsr_db'],faded_signal=result['faded_signal'],faded_jammer=result['faded_jammer']);record={'step':step,'loss':float(loss),'aggregate':metric['aggregate'],'per_layer':metric['per_layer'],'gradient_norms':grads,'channel_metrics':{'csi_nmse':float(result['csi_nmse'].mean()),'pilot_evm':float(result['pilot_evm'].mean()),**result['equalizer_gain_statistics']},'jammer_metrics':diag};snrs.append(p['snr_db']);jsrs.append(p['pilot_jsr_db']);covers.append(p['coverage'])
   for k,v in [('channel',batch.signal_fading),('jammer_channel',batch.jammer_fading),('jammer',batch.jammer),('noise',batch.noise),('mask',batch.jammer_mask)]:hashes[k].add(stable_tensor_hash(v))
   if step==1 or step%a.validation_every==0 or step==a.steps:record['validation']=evaluate(codec,model,config,sources['val'],[x for x in suite['scenarios'] if x['group'].startswith('unseen_speaker') or x['group']=='j5_strongest_selected_condition'],device)
   history.append(record);f.write(json.dumps(record)+'\n');f.flush()
   if step%a.checkpoint_every==0 or step==a.steps:torch.save({'diagnostic_type':'j5_pilot_targeted','provenance':provenance,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'history':history},out/'diagnostic_last.pt')
 final=next(x['validation'] for x in reversed(history) if x.get('validation'));unseen=final['unseen_speaker_unseen_utterance_unseen_channel']['layers'];strong=final['j5_strongest_selected_condition']['layers'];tails=final['unseen_speaker_unseen_utterance_unseen_channel']['tail_statistics']['layer7_improvement'];infra={'finite':all(x['aggregate']['finite'] for x in history),'mask':all(x['jammer_metrics']['jammer_leakage_power_on_data_resources']<=1e-12 for x in history),'coverage':set(dist['selected_pilot_coverages']).issubset(set(covers)),'diversity':min(map(len,hashes.values()))>=min(a.steps,2),'provenance':True,'gain_logging':True};gate=j5_gate(unseen,strong,infrastructure=infra,tail={'p10':tails['p10'],'negative_rate':tails['negative_rate']});curve=[x['validation']['unseen_speaker_unseen_utterance_unseen_channel']['aggregate']['normalized_mse'] for x in history if x.get('validation')];slope=linear_slope(curve[-max(2,len(curve)//5):]);classification=classify_j5(gate,curve[-1]<=min(curve),slope);summary={'classification':classification,'gate':gate,'steps':a.steps,'provenance':provenance,'validation':final,'stochastic_diversity':{k:len(v) for k,v in hashes.items()},'snr_range':[min(snrs),max(snrs)],'pilot_jsr_range':[min(jsrs),max(jsrs)],'coverage_values':sorted(set(covers))};(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({'classification':classification,'output_dir':str(out)},indent=2))
if __name__=='__main__':main()
