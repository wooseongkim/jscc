"""Strict post-selection legacy waveform evaluator."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--selection-artifact',required=True);p.add_argument('--config',default='configs/eval_r4_si_sdr_finetune_checkpoints.yaml');p.add_argument('--device',default='cuda');p.add_argument('--output-dir',default='runs/waveform_aware_wireless/r4_si_sdr_finetune/legacy_final_post_selection');p.add_argument('--dry-run',action='store_true');p.add_argument('--allow-long-run',action='store_true');a=p.parse_args(); s=Path(a.selection_artifact)
 if not s.is_file(): raise SystemExit('legacy evaluation requires expanded-selection experiment_best_checkpoints.json')
 d=json.loads(s.read_text())
 if d.get('selection_split')!='expanded_selection' or d.get('selection_uses_legacy_metrics') is not False: raise SystemExit('invalid selection artifact')
 selected=[('source',d['source_checkpoint']['path'])]
 for name in ('control_no_si_sdr','si_sdr_low','si_sdr_medium'):
  x=d['experiments'].get(name,{})
  if x.get('status')!='selected' or not x.get('constraint_pass'): raise SystemExit(f'no valid selected {name}')
  if sha(x['checkpoint_path'])!=x['sha256']: raise SystemExit(f'checkpoint hash mismatch: {name}')
  selected.append((name,x['checkpoint_path']))
 if a.dry_run: print(json.dumps({'legacy_ranking_forbidden':True,'checkpoints':selected},indent=2));return
 if not a.allow_long_run: raise SystemExit('--allow-long-run required')
 out=ROOT/a.output_dir; out.mkdir(parents=True,exist_ok=True); (out/'selection_artifact_snapshot.json').write_text(json.dumps(d,indent=2)); rows=[]
 manifest=ROOT/'runs/waveform_aware_wireless/r4_expanded_validation/final_test_manifest_reference.jsonl'
 for role,cp in selected:
  for snr in (5,10,15):
   child=out/f'{role}_{snr}db'; cmd=[sys.executable,str(ROOT/'evaluate_r4_si_sdr_finetune_checkpoints.py'),'--config',str(ROOT/a.config),'--candidate-checkpoint',cp,'--manifest',str(manifest),'--realization-seeds','41001,41002','--max-utterances','64','--max-realizations','2','--snr-db',str(snr),'--output-dir',str(child),'--device',a.device,'--overwrite']
   subprocess.run(cmd,check=True,cwd=ROOT)
   rows += [dict(json.loads(x),checkpoint_role=role) for x in (child/'per_sample_selection_results.jsonl').read_text().splitlines()]
 with (out/'per_sample_legacy_results.jsonl').open('w') as f:
  for r in rows:f.write(json.dumps(r)+'\n')
 summary={}
 for role,_ in selected:
  summary[role]={str(s):sum(r['candidate_si_sdr_db'] for r in rows if r['checkpoint_role']==role and r['snr_db']==s)/128 for s in (5,10,15)}
 (out/'checkpoint_summary.json').write_text(json.dumps(summary,indent=2)); (out/'selection_artifact_integrity.json').write_text(json.dumps({'sha256_before':sha(s),'sha256_after':sha(s),'unchanged':sha(s)==sha(s)},indent=2))
if __name__=='__main__':main()
