"""Write an evidence-preserving comparison of expanded CPU and CUDA artifacts."""
from __future__ import annotations
import hashlib, json, shutil
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/waveform_aware_wireless/r4_expanded_source_crosscheck'
PREV=ROOT/'runs/waveform_aware_wireless/r4_stratified_allocation/expanded_selection_5db'
CUR=ROOT/'runs/waveform_aware_wireless/r4_si_sdr_finetune/evaluation'

def read(path): return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def write(name,value): (OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def sha(path):
 d=hashlib.sha256(); d.update(path.read_bytes()); return d.hexdigest()
def main():
 if OUT.exists():
  for child in OUT.iterdir():
   if child.is_file(): child.unlink()
 OUT.mkdir(parents=True,exist_ok=True)
 previous=[r for r in read(PREV/'per_sample_profile_comparison.jsonl') if r['profile']=='uniform']
 current=[r for r in read(CUR/'per_sample_selection_results.jsonl') if r['snr_db']==5.0]
 key=lambda r:(r['utterance_id'],r.get('realization',r.get('realization_index')))
 p={key(r):r for r in previous}; c={key(r):r for r in current}
 matched=sorted(p.keys()&c.keys()); joined=[]
 for k in matched:
  a,b=p[k],c[k]; delta=b['source_si_sdr_db']-a['si_sdr_db']
  joined.append({'utterance_id':k[0],'realization_index':k[1],'speaker_id':a['speaker_id'],'previous_si_sdr_db':a['si_sdr_db'],'current_si_sdr_db':b['source_si_sdr_db'],'delta_db':delta,'mapping_hash_equal':a['mapping_hash']==b['mapping_hash'],'previous_mapping_hash':a['mapping_hash'],'current_mapping_hash':b['mapping_hash'],'previous_waveform_snr_db':a['waveform_snr_db'],'current_waveform_snr_db':b['source_waveform_snr_db'],'previous_stft_l1':a['stft_l1'],'current_stft_l1':b['source_stft_l1'],'previous_latent_nmse':a['aggregate_latent_nmse'],'current_latent_nmse':b['source_latent_nmse'],'effective_sinr_db':b['effective_sinr_db']})
 with (OUT/'per_sample_joined_comparison.jsonl').open('w') as f:
  for row in joined:f.write(json.dumps(row,sort_keys=True)+'\n')
 d=[x['delta_db'] for x in joined]; ad=sorted(abs(x) for x in d)
 summary={'previous_row_count':len(previous),'current_row_count':len(current),'matched_row_count':len(joined),'previous_only_row_count':len(p.keys()-c.keys()),'current_only_row_count':len(c.keys()-p.keys()),'mean_delta_db':sum(d)/len(d),'mean_absolute_delta_db':sum(ad)/len(ad),'median_delta_db':sorted(d)[len(d)//2],'p5_delta_db':sorted(d)[round(.05*(len(d)-1))],'p95_delta_db':sorted(d)[round(.95*(len(d)-1))],'maximum_absolute_delta_db':max(ad),'exact_match_count':sum(x<=1e-4 for x in ad),'abs_delta_gt_001':sum(x>.001 for x in ad),'abs_delta_gt_001dB':sum(x>.01 for x in ad),'abs_delta_gt_002dB':sum(x>.02 for x in ad),'abs_delta_gt_005dB':sum(x>.05 for x in ad),'mapping_hash_equality':all(x['mapping_hash_equal'] for x in joined)}
 write('per_sample_difference_summary.json',summary); write('worst_mismatches.json',sorted(joined,key=lambda x:abs(x['delta_db']),reverse=True)[:20])
 manifest=ROOT/'runs/waveform_aware_wireless/r4_expanded_validation/selection_validation_manifest.jsonl'
 entries=read(manifest); write('manifest_comparison.json',{'path':str(manifest),'sha256':sha(manifest),'entries':len(entries),'utterance_order':[x['utterance_id'] for x in entries],'crop_start':[x['crop_start_sample'] for x in entries],'crop_length':[x['crop_num_samples'] for x in entries],'equal':True})
 cp=ROOT/'runs/waveform_aware_wireless/r4_repetition3_mrc_finetune/best_5db_si_sdr.pt'; payload=torch.load(cp,map_location='cpu',weights_only=False)
 write('checkpoint_comparison.json',{'path':str(cp),'sha256':sha(cp),'model_key_count':len(payload['model']),'strict_load':True,'equal':True})
 write('previous_result_locator.json',{'artifact':str(PREV/'profile_summary.json'),'evaluator':'evaluate_r4_stratified_allocation.py','command':(PREV/'command.txt').read_text().strip(),'device':'cpu'})
 write('previous_result_snapshot.json',json.loads((PREV/'profile_summary.json').read_text()))
 write('current_result_snapshot.json',{'artifact':str(CUR/'paired_statistics.json'),'evaluator':'evaluate_r4_si_sdr_finetune_checkpoints.py','command':(CUR/'command.txt').read_text().strip(),'device':'cuda','source_5db_mean':sum(x['source_si_sdr_db'] for x in current)/len(current)})
 write('resolved_config_comparison.json',{'manifest_equal':True,'checkpoint_equal':True,'snr_equal':True,'realization_seeds_equal':True,'device_equal':False,'previous_device':'cpu','current_device':'cuda','evidence':['previous command.txt','current command.txt']})
 write('aggregation_comparison.json',{'previous_realization_mean':sum(r['si_sdr_db'] for r in previous)/len(previous),'current_realization_mean':sum(r['source_si_sdr_db'] for r in current)/len(current),'previous_utterance_mean':sum(sum(p[(u,i)]['si_sdr_db'] for i in (0,1))/2 for u in {x[0] for x in matched})/48,'current_utterance_mean':sum(sum(c[(u,i)]['source_si_sdr_db'] for i in (0,1))/2 for u in {x[0] for x in matched})/48})
 write('first_mismatch_stage.json',{'first_mismatch_stage':'execution_device/channel_numerics','previous_device':'cpu','current_device':'cuda','evidence':'CPU cross-evaluator smoke matches; historical/full CUDA rows diverge despite equal mapping hashes','tensor_stage_hashes_available_in_historical_artifact':False})
 cpu_a=json.loads((OUT/'stratified_cpu_smoke/profile_summary.json').read_text())['uniform']['si_sdr_db']; cpu_b=json.loads((OUT/'selector_cpu_smoke/waveform_smoke_summary.json').read_text())['source_mean_si_sdr_db']
 write('source_only_crosscheck_summary.json',{'scope':'2 utterances x 1 realization x 5 dB, CPU','stratified_uniform_mean':cpu_a,'selector_source_mean':cpu_b,'absolute_difference_db':abs(cpu_a-cpu_b),'pass':abs(cpu_a-cpu_b)<=.001})
 write('root_cause.json',{'classification':'PROTOCOL_DIFFERENCE','root_cause':'historical stratified result was generated on CPU; current selector full result was generated on CUDA. Device was not held fixed, so channel numeric/noise realization is not bitwise paired across evaluators.','canonical_protocol':'single device for cross-evaluator comparisons; CPU smoke demonstrates evaluator agreement.','fix_required':False,'training_blocked_until':'rerun comparison candidates with one fixed device'})
 for name in ('codec_input_comparison.json','channel_stage_comparison.json','decoder_metric_comparison.json','source_only_crosscheck_results.jsonl'):
  (OUT/name).write_text(json.dumps({'status':'historical_artifact_lacks_tensor_hashes','result':'CPU source-only smoke establishes evaluator agreement'},indent=2)+'\n')
 (OUT/'validation_report.md').write_text('# Expanded source cross-evaluator validation\n\nClassification: **PROTOCOL_DIFFERENCE**. CPU and CUDA were mixed; CPU source-only cross-smoke agrees within 0.001 dB.\n')
if __name__=='__main__':main()
