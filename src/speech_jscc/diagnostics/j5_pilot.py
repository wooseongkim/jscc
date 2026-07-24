from __future__ import annotations
import math,random,json
from pathlib import Path
from dataclasses import replace
from typing import Any
import torch
from evaluation.paired import PairedEvaluationBatch
from speech_jscc.diagnostics.j2_barrage import file_sha256
from speech_jscc.diagnostics.random_distribution import SeedDeriver

J5_VERSION='j5_pilot_targeted_v1'
J5_THRESHOLDS={'aggregate_improvement':.10,'aggregate_correlation':.30,'aggregate_power_ratio':.12,'enhancement_improvement':.075,'enhancement_correlation':.25,'enhancement_power_ratio':.08,'deep_improvement':.05,'layer7_improvement':.03,'layer7_power_ratio':.04,'worst_aggregate':.075,'worst_enhancement':.05,'worst_layer7':.02,'tail_p10':0.,'tail_negative_rate':.10}

def make_pilot_local_jammer(reference,pilot_mask,pilot_jsr_db,coverage,*,generator):
    from channels.jammer import make_jammer_mask,_complex_normal_like
    mask=make_jammer_mask(reference,'pilot',coverage,pilot_mask=pilot_mask,generator=generator)
    raw=_complex_normal_like(reference,generator)*mask
    jammer=torch.zeros_like(raw)
    for i in range(reference.shape[0]):
        signal=reference[i][mask[i]].abs().square().mean();power=raw[i][mask[i]].abs().square().mean().clamp_min(1e-12)
        jammer[i]=raw[i]*torch.sqrt(signal*(10**(float(pilot_jsr_db)/10))/power)
    return jammer,mask

def normalize_pilot_local_batch(batch:PairedEvaluationBatch,pilot_jsr_db:float,coverage:float):
    generator=torch.Generator(device=batch.jammer.device).manual_seed(batch.seed+55109)
    reference=torch.full_like(batch.jammer,complex(math.sqrt(batch.target_power),0))
    jammer,mask=make_pilot_local_jammer(reference,batch.pilot_mask,pilot_jsr_db,coverage,generator=generator)
    return replace(batch,jammer_type='pilot',jammer=jammer,jammer_mask=mask,jsr_db=torch.full_like(batch.jsr_db,float(pilot_jsr_db)),metadata={**batch.metadata,'jsr_semantics':'pilot_local','pilot_coverage':float(coverage)})

def pilot_jammer_diagnostics(signal,jammer,mask,pilot_mask,*,requested_pilot_jsr_db,faded_signal=None,faded_jammer=None):
    total=int(pilot_mask[0].sum());attacked=int(mask[0].sum());grid=pilot_mask[0].numel();fs=signal if faded_signal is None else faded_signal;fj=jammer if faded_jammer is None else faded_jammer
    local=[]
    for i in range(signal.shape[0]):local.append(fj[i][mask[i]].abs().square().mean()/fs[i][mask[i]].abs().square().mean().clamp_min(1e-12))
    global_ratio=fj.abs().square().mean((1,2))/fs.abs().square().mean((1,2)).clamp_min(1e-12);data=~pilot_mask
    return {'requested_pilot_jsr_db':float(requested_pilot_jsr_db),'realized_received_pilot_jsr_db':float(10*torch.log10(torch.stack(local)).mean()),
      'requested_global_equivalent_jsr_db':float(requested_pilot_jsr_db)+10*math.log10(attacked/grid),'realized_received_global_equivalent_jsr_db':float(10*torch.log10(global_ratio).mean()),
      'total_pilot_count':total,'attacked_pilot_count':attacked,'attacked_pilot_fraction':attacked/total,'pilot_fraction_of_full_grid':total/grid,
      'pilot_jammer_power':float(jammer[mask].abs().square().mean()),'legitimate_pilot_power':float(signal[mask].abs().square().mean()),
      'jammer_leakage_power_on_data_resources':float(jammer[data].abs().square().mean())}

def j5_policy(seed,step,snr_range,jsr_range,coverages):
    rng=random.Random(SeedDeriver(seed).seed('j5_distribution',step));return {'seed':SeedDeriver(seed).seed('j5_channel_jammer_noise',step),'snr_db':rng.uniform(*snr_range),'pilot_jsr_db':rng.uniform(*jsr_range),'coverage':float(rng.choice(list(coverages)))}

def j5_gate(unseen,strongest,*,infrastructure,tail,thresholds=J5_THRESHOLDS):
    a=unseen['aggregate'];e=unseen['layers1_to_7'];d=unseen['layers6_to_7'];l=unseen['layer7'];wa=strongest['aggregate'];we=strongest['layers1_to_7'];wl=strongest['layer7']
    g={'aggregate_pass':a['relative_improvement_over_zero']>=thresholds['aggregate_improvement'] and a['pearson_correlation']>=thresholds['aggregate_correlation'] and a['power_ratio']>=thresholds['aggregate_power_ratio'],
      'enhancement_pass':e['relative_improvement_over_zero']>=thresholds['enhancement_improvement'] and e['pearson_correlation']>=thresholds['enhancement_correlation'] and e['power_ratio']>=thresholds['enhancement_power_ratio'],
      'deep_pass':d['relative_improvement_over_zero']>=thresholds['deep_improvement'],'layer7_pass':l['relative_improvement_over_zero']>=thresholds['layer7_improvement'] and l['pearson_correlation']>0 and l['power_ratio']>=thresholds['layer7_power_ratio'],
      'strongest_pass':wa['relative_improvement_over_zero']>=thresholds['worst_aggregate'] and we['relative_improvement_over_zero']>=thresholds['worst_enhancement'] and wl['relative_improvement_over_zero']>=thresholds['worst_layer7'] and min(wa['pearson_correlation'],we['pearson_correlation'],wl['pearson_correlation'])>0,
      'tail_pass':tail['p10']>=thresholds['tail_p10'] and tail['negative_rate']<=thresholds['tail_negative_rate'],**{f'infrastructure_{k}':bool(v) for k,v in infrastructure.items()},'thresholds':dict(thresholds)}
    g['mean_passed']=all(v for k,v in g.items() if k not in {'thresholds','tail_pass'});g['passed']=g['mean_passed'] and g['tail_pass'];return g

def classify_j5(g,final_is_best,slope):
    if g['passed']:return 'PASS'
    if not g.get('infrastructure_finite',True):return 'FAIL_NONFINITE'
    if not g.get('infrastructure_mask',True):return 'FAIL_MASK_IMPLEMENTATION'
    if g.get('mean_passed') and not g['tail_pass']:return 'MARGINAL_TAIL'
    if final_is_best and slope<0:return 'INCONCLUSIVE_NOT_CONVERGED'
    if not g.get('layer7_pass',True):return 'FAIL_LAYER7'
    if not g.get('enhancement_pass',True) or not g.get('deep_pass',True):return 'FAIL_ENHANCEMENT'
    return 'FAIL_PILOT_CSI' if not g.get('strongest_pass',True) else 'FAIL_AGGREGATE'

def verify_j4_accepted(manifest_path,checkpoint_path):
    m=json.loads(Path(manifest_path).read_text());
    if m.get('accepted_status')!='ACCEPTED_PASS' or m.get('j4_checkpoint_sha256')!=file_sha256(checkpoint_path):raise ValueError('accepted J4 artifact hash mismatch')
    for pkey,hkey in [('original_j4_summary_path','original_j4_summary_sha256'),('corrected_supplement_path','corrected_supplement_sha256'),('tail_diagnostic_summary_path','tail_diagnostic_summary_sha256')]:
        if file_sha256(m[pkey])!=m[hkey]:raise ValueError(f'accepted J4 artifact hash mismatch: {hkey}')
    return m
