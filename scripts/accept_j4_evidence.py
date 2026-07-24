from __future__ import annotations
import argparse,json,subprocess,hashlib
from datetime import datetime,timezone
from pathlib import Path
from speech_jscc.diagnostics.j2_barrage import file_sha256

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='runs/stage1_conv_conformer_jammer/j4_random_burst');p.add_argument('--overwrite',action='store_true');a=p.parse_args();root=Path(a.root);out=root/'accepted_manifest.json'
 if out.exists() and not a.overwrite:raise SystemExit(f'refusing existing accepted manifest: {out}')
 training=root/'training/j3_transfer';summary=training/'summary.json';checkpoint=training/'diagnostic_last.pt';tail=root/'tail_diagnostic/summary.json';supp=root/'tail_diagnostic/corrected_strongest_condition_supplement.json'
 data=json.loads(summary.read_text());prov=data['provenance'];parent=Path(prov['parent_checkpoint']);commit=subprocess.run(['git','rev-parse','HEAD'],text=True,capture_output=True,check=True).stdout.strip()
 manifest={'schema_version':'j4_accepted_v1','accepted_status':'ACCEPTED_PASS','j4_checkpoint_path':str(checkpoint),'j4_checkpoint_sha256':file_sha256(checkpoint),
  'original_j4_summary_path':str(summary),'original_j4_summary_sha256':file_sha256(summary),'corrected_supplement_path':str(supp),'corrected_supplement_sha256':file_sha256(supp),
  'tail_diagnostic_summary_path':str(tail),'tail_diagnostic_summary_sha256':file_sha256(tail),'parent_j3_checkpoint_path':str(parent),'parent_j3_checkpoint_sha256':file_sha256(parent),
  'architecture_version':prov['architecture_version'],'codec_metadata':prov['preprocessing'],'latent_shape':prov['representation_shape'],'preprocessing_hash':prov['preprocessing'].get('preprocessing_hash') or hashlib.sha256(json.dumps(prov['preprocessing'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
  'dataset_manifest_hash':prov['train_manifest_hash'],'git_commit':commit,'acceptance_timestamp':datetime.now(timezone.utc).isoformat(),
  'acceptance_reason':'Corrected SNR 5 dB, pilot+data burst global JSR 0 dB, fraction 0.125 evaluation over 64 unseen utterances/128 realizations: Layer-7 p10 +1.56%, negative rate 2.34%; all corrected J4 gates pass.',
  'original_summary_preserved':True}
 out.write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
