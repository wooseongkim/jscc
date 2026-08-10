"""Fixed UEP repetition/power profiles under a deterministic broadband jammer."""
from __future__ import annotations
import argparse, copy, csv, json, math, shutil, sys
from pathlib import Path
import torch, yaml
from channels.r4_uep_allocator import UEPProfile, UEP_PROFILES
from evaluate_r4_expanded_validation import _cache_codec_inputs, _conditions, _make_engine, _tensor_hash
from evaluate_r4_si_sdr_finetune_checkpoints import _checkpoint_payload, _repository_path, _source_rows
from speech_jscc.config import resolve_device
from speech_jscc.evaluation.expanded_validation import ManifestEntry, file_sha256
from speech_jscc.evaluation.r4_jammer_baseline import build_r4_jammer, tensor_hash
from speech_jscc.evaluation.r4_si_sdr_checkpoint_selection import paired_statistics
from speech_jscc.experiment import build_components
from speech_jscc.training.channel_free_revalidation import per_layer_nmse
from speech_jscc.training.r4_waveform_finetune import freeze_codec_for_input_gradient
from src.evaluation.waveform_metrics import waveform_metrics

ROOT=Path(__file__).resolve().parent
def _json(p,v): p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def _jsonl(p,rows): p.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows))
def _jsr(values): return [None if x=='no_jammer' else float(x) for x in values]
def _profiles_from_json(path: Path | None):
 if path is None: return dict(UEP_PROFILES)
 payload=json.loads(path.read_text()); items=payload.get('profiles',payload)
 profiles={}
 for name,value in items.items():
  # Search artifacts were serialized from float32 tensors.  Their simplex
  # sum can differ from one by about 1e-8 even though the optimizer enforced
  # it exactly.  Canonicalize only this serialization roundoff; materially
  # invalid profile JSON remains an error in ``UEPProfile``.
  shares=tuple(float(x) for x in value['power_share']); total=sum(shares)
  if abs(total-1.0)<=1e-6: shares=tuple(x/total for x in shares)
  profiles[name]=UEPProfile(name,tuple(value['repetition']),power_share=shares)
 return profiles
def _selection_jammer_seed(utterance_id, realization_index, snr_db, jsr_db):
 # Stable condition-derived seed; independent of repetition/power candidate.
 text=f'expanded_selection|{utterance_id}|{realization_index}|{snr_db}|{jsr_db}'
 return int(__import__('hashlib').sha256(text.encode()).hexdigest()[:8],16) % (2**31-1)
def args():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',default='configs/eval_r4_broadband_uep_profiles.yaml'); p.add_argument('--checkpoint'); p.add_argument('--split',default='legacy_final',choices=['legacy_final','expanded_selection']); p.add_argument('--profiles',nargs='+',default=None); p.add_argument('--profile-json'); p.add_argument('--jammer-type',default='broadband_awgn',choices=['broadband_awgn']); p.add_argument('--jsr-db',nargs='+'); p.add_argument('--snr-db',nargs='+',type=float); p.add_argument('--max-utterances',type=int); p.add_argument('--max-realizations',type=int); p.add_argument('--device',default='cpu'); p.add_argument('--output-dir'); p.add_argument('--allow-long-run',action='store_true'); p.add_argument('--overwrite',action='store_true'); p.add_argument('--dry-run',action='store_true'); return p.parse_args()
def main():
 a=args(); c=yaml.safe_load(_repository_path(a.config).read_text()); profile_defs=_profiles_from_json(_repository_path(a.profile_json) if a.profile_json else None); names=a.profiles or list(profile_defs)
 if any(x not in profile_defs for x in names): raise SystemExit('unknown UEP profile')
 if 'U0' not in names: raise SystemExit('U0 is required for paired UEP comparison')
 checkpoint=_repository_path(a.checkpoint or c['medium_checkpoint'])
 if a.dry_run: print(json.dumps({'checkpoint':str(checkpoint),'profiles':names,'jsr_db':a.jsr_db or c['jsr_db']},indent=2)); return
 n=a.max_utterances or c['evaluation']['utterances']; realiz=a.max_realizations or c['evaluation']['realizations']; snrs=a.snr_db or c['evaluation']['snr_db']; jsrs=_jsr(a.jsr_db or c['jsr_db'])
 if n*realiz*len(snrs)*len(jsrs)*len(names)>c['evaluation']['smoke_max_rows'] and not a.allow_long_run: raise SystemExit('full UEP evaluation requires --allow-long-run')
 out=_repository_path(a.output_dir or c['output_root'])
 if out.exists():
  if not a.overwrite: raise SystemExit(f'refusing existing output directory: {out}')
  shutil.rmtree(out)
 out.mkdir(parents=True)
 payload=_checkpoint_payload(checkpoint); mc=copy.deepcopy(payload['config']); mc['device']=a.device
 for k in ('config_path','checkpoint_path'):
  p=Path(mc['codec'][k]); mc['codec'][k]=str(_repository_path(p)) if not p.is_absolute() else str(p)
 dev=resolve_device(a.device); codec,model=build_components(mc,dev); model.load_state_dict(payload['model'],strict=True); freeze_codec_for_input_gradient(codec); model.eval().requires_grad_(False)
 model._expanded_validation_config=mc
 manifest_key='legacy_final_manifest' if a.split=='legacy_final' else 'expanded_selection_manifest'
 entries=[ManifestEntry(**json.loads(x)) for x in _repository_path(c[manifest_key]).read_text().splitlines() if x.strip()][:n]; cache=_cache_codec_inputs(codec,entries,mc,dev)
 canonical_rows=[json.loads(x) for x in _repository_path(c['canonical_broadband_baseline_rows']).read_text().splitlines() if x.strip()] if a.split=='legacy_final' else []
 canonical_seed={(r['utterance_id'],int(r['realization_index']),float(r['snr_db']),r['target_jsr_db']):int(r['jammer_seed']) for r in canonical_rows if r['jammer_type']=='broadband_awgn'}
 spec={'physical_profile_config':str(_repository_path(c['physical_profile_config'])),'physical':c['physical'],'pairing':{'csi_report_tolerance':1e-5}}
 engines={name:_make_engine(codec,model,spec,uep_profile_name=name if name in UEP_PROFILES else 'U0',uep_profile=profile_defs[name]) for name in names}; u0_engine=_make_engine(codec,model,spec); source=copy.deepcopy(model); source.load_state_dict(_checkpoint_payload(_repository_path(c['csi_reference_checkpoint']))['model'],strict=True); source._expanded_validation_config=mc; source.eval().requires_grad_(False); csi_engine=_make_engine(codec,source,spec)
 rows=[]
 with torch.no_grad():
  for ri,seed in enumerate(c['realization_seeds'][:realiz]):
   for snr in snrs:
    conditions=_conditions(spec,csi_engine,snr=snr,seed=seed,count=len(cache)); csi_rows=_source_rows(codec=codec,model=source,cache=cache,conditions=conditions,specification=spec,device=dev)
    for cached,condition,csirow in zip(cache,conditions,csi_rows,strict=True):
     wave,target=cached['waveform'].to(dev),cached['target'].to(dev)
     # U0 provides the nominal whole-grid signal calibration; this jammer tensor is reused verbatim by every profile.
     u0clean=u0_engine.forward(target,wave,condition,csirow['input_report'],training=False)
     for jsr in jsrs:
      if jsr is None:
       seed_j=0
      else:
       key=(cached['entry'].utterance_id,ri,float(snr),float(jsr))
       if a.split=='legacy_final':
        if key not in canonical_seed: raise RuntimeError(f'missing canonical broadband seed: {key}')
        seed_j=canonical_seed[key]
       else: seed_j=_selection_jammer_seed(cached['entry'].utterance_id,ri,float(snr),float(jsr))
      jammer=build_r4_jammer(u0clean.allocation.place(u0clean.tx_symbols), u0clean.allocation.profile and __import__('channels.physical_ofdm',fromlist=['active_grid_masks']).active_grid_masks(u0clean.allocation.profile,device=dev).candidate_data, jammer_type='no_jammer' if jsr is None else 'broadband_awgn',jsr_db=jsr,seed=seed_j)
      for name,engine in engines.items():
       result=engine.forward(target,wave,condition,csirow['input_report'],training=False,jammer_type='no_jammer' if jsr is None else 'broadband_awgn',jammer_jsr_db=jsr,jammer_seed=seed_j,jammer_override=jammer)
       m=waveform_metrics(wave,result.decoded_waveform,int(mc['codec']['sample_rate'])); layer=per_layer_nmse(result.reconstruction,target); alloc=result.allocation
       row={'split':a.split,'utterance_id':cached['entry'].utterance_id,'speaker_id':cached['entry'].speaker_id,'realization_index':ri,'realization_seed':seed,'snr_db':float(snr),'target_jsr_db':jsr,'profile':name,'repetition':list(profile_defs[name].repetition),'power_share':[float(x) for x in profile_defs[name].total_power_share()],'per_re_power':[float(x) for x in profile_defs[name].per_re_layer_power()],'normalized_power':[float(x) for x in profile_defs[name].normalized_power()],'mapping_hash':_tensor_hash(result.mapping_indices),'jammer_mask_hash':jammer.statistics['jammer_mask_hash'],'jammer_tensor_hash':jammer.statistics['jammer_tensor_hash'],'jammer_seed':seed_j,'codec_input_hash':cached['codec_baseline_hash'],'target_waveform_hash':_tensor_hash(wave),'si_sdr_db':float(m['si_sdr_db']),'waveform_snr_db':float(m['waveform_snr_db']),'stft_l1':float(m['stft_l1']),'aggregate_latent_nmse':float(layer.mean()),'per_layer_nmse':[float(x) for x in layer],'effective_sinr_db':float(10*torch.log10(result.effective_sinr)),'total_re':int(alloc.selected_candidate_indices.ge(0).sum()) if hasattr(alloc,'selected_re_count') else int(alloc.selected_candidate_indices.numel()),'total_power':float(alloc.power_source_order.sum()),'finite':bool(torch.isfinite(result.reconstruction).all())}
       if not row['finite'] or row['total_re']!=5760 or abs(row['total_power']-5760)>1e-2: raise FloatingPointError('UEP invariant failure')
       rows.append(row)
 base={(r['utterance_id'],r['realization_index'],r['snr_db'],r['target_jsr_db']):r for r in rows if r['profile']=='U0'}
 for r in rows:
  b=base[(r['utterance_id'],r['realization_index'],r['snr_db'],r['target_jsr_db'])]; r['delta_si_sdr_vs_u0_db']=r['si_sdr_db']-b['si_sdr_db']
 summary=[]; stats={}
 for name in names:
  for snr in snrs:
   for jsr in jsrs:
    rr=[r for r in rows if r['profile']==name and r['snr_db']==snr and r['target_jsr_db']==jsr]; values={}
    for r in rr: values.setdefault(r['utterance_id'],[]).append(r['delta_si_sdr_vs_u0_db'])
    st=paired_statistics([sum(v)/len(v) for v in values.values()],samples=int(c['bootstrap_samples']),seed=int(c['bootstrap_seed'])); stats[f'{name}|{snr}|{jsr}']=st; summary.append({'profile':name,'snr_db':snr,'target_jsr_db':jsr,'rows':len(rr),'mean_si_sdr_db':sum(r['si_sdr_db'] for r in rr)/len(rr),'mean_delta_si_sdr_vs_u0_db':st['mean'],'mean_waveform_snr_db':sum(r['waveform_snr_db'] for r in rr)/len(rr),'mean_stft_l1':sum(r['stft_l1'] for r in rr)/len(rr),'mean_latent_nmse':sum(r['aggregate_latent_nmse'] for r in rr)/len(rr)})
 full_scope=n==c['evaluation']['utterances'] and realiz==c['evaluation']['realizations'] and all(x in snrs for x in [5.,10.,15.])
 anchor={'status':'NOT_APPLICABLE_SUBSET','pass':None,'u0_rows':len([r for r in rows if r['profile']=='U0'])}
 if full_scope and a.split=='legacy_final':
  observed={str(x):next(s['mean_si_sdr_db'] for s in summary if s['profile']=='U0' and s['snr_db']==5. and s['target_jsr_db']==x) for x in jsrs}
  expected=c['u0_legacy_5db_reference']; errors={str(x):float(observed[str(x)])-float(expected['no_jammer' if x is None else int(x)]) for x in jsrs}
  passed=all(abs(x)<=float(c['u0_anchor_tolerance_db']) for x in errors.values())
  anchor={'status':'PASS' if passed else 'FAIL','pass':passed,'observed':observed,'errors_db':errors,'tolerance_db':c['u0_anchor_tolerance_db']}
  if not passed: raise RuntimeError(f'U0 anchor failed: {anchor}')
 _jsonl(out/'per_sample_uep_results.jsonl',rows); _json(out/'paired_statistics.json',stats); _json(out/'profile_manifest.json',{n:{'repetition':profile_defs[n].repetition,'power_share':[float(x) for x in profile_defs[n].total_power_share()],'per_re_power':[float(x) for x in profile_defs[n].per_re_layer_power()]} for n in names}); _json(out/'profile_power_normalization.json',{n:[float(x) for x in profile_defs[n].normalized_power()] for n in names}); _json(out/'variable_repetition_mapping_summary.json',{'profiles':names,'total_re':5760,'total_power':5760});
 with (out/'per_condition_summary.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)
 _json(out/'u0_anchor_check.json',anchor); _json(out/'clean_cost_summary.json',{k:v for k,v in stats.items() if '|5.0|None' in k}); _json(out/'layer_uep_damage_summary.json',{}); _json(out/'final_uep_decision.json',{'status':'descriptive_only','selection':'no profile selected'}); (out/'resolved_config.yaml').write_text(yaml.safe_dump(c)); (out/'command.txt').write_text(' '.join(sys.argv)+'\n'); _json(out/'checkpoint_hashes.json',{'checkpoint_sha256':file_sha256(checkpoint)}); print(json.dumps({'output':str(out),'rows':len(rows)},indent=2))
if __name__=='__main__': main()
