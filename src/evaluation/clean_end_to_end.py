from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import Any
import torch
from channels.pilot import make_pilot_mask,insert_data_and_pilots,extract_data_resources
from models.observable_channel_state import build_observable_receiver_state_v1

def normalization_roundtrip_metrics(value,mean,std,epsilon):
    mean=mean.to(value)[None];std=std.to(value)[None].clamp_min(epsilon);restored=((value-mean)/std)*std+mean;diff=(restored-value).abs()
    return {'restored':restored,'max_abs_error':float(diff.max()),'mean_abs_error':float(diff.mean()),'per_layer_max_abs_error':[float(x) for x in diff.flatten(2).amax((0,2))]}

def neutral_observable_state(batch_size,*,device,dtype=torch.float32):
    shape=(batch_size,64,32);received=torch.ones(shape,device=device,dtype=torch.complex64);pilots=torch.ones_like(received);mask=make_pilot_mask(shape,4,time_spacing=4,device=device);estimated=torch.ones_like(received)
    state=build_observable_receiver_state_v1(received,pilots,mask,estimated)
    return state.to(dtype=dtype)

def ideal_ofdm_roundtrip(symbols,*,subcarriers=64,ofdm_symbols=32,pilot_spacing=4):
    mask=make_pilot_mask((symbols.shape[0],subcarriers,ofdm_symbols),pilot_spacing,time_spacing=pilot_spacing,device=symbols.device);grid,pilots=insert_data_and_pilots(symbols,mask);recovered=extract_data_resources(grid,mask);error=(recovered-symbols).abs()
    return {'recovered':recovered,'grid':grid,'pilots':pilots,'pilot_mask':mask,'pilot_count':int(mask[0].sum()),'data_count':int((~mask[0]).sum()),'max_recovery_error':float(error.max()),'pilot_leakage':int(mask[0][~mask[0]].sum()),'resource_order_hash':hashlib.sha256(recovered.detach().cpu().numpy().tobytes()).hexdigest()}

def _flat_metrics(reconstruction,target,epsilon=1e-8):
    x=reconstruction.float().flatten(1);y=target.float().flatten(1);mse=(x-y).square().mean();power_y=y.square().mean();power_x=x.square().mean();cos=(x*y).sum(1)/(x.norm(dim=1)*y.norm(dim=1)).clamp_min(epsilon);xc=x-x.mean(1,keepdim=True);yc=y-y.mean(1,keepdim=True);corr=(xc*yc).sum(1)/(xc.norm(dim=1)*yc.norm(dim=1)).clamp_min(epsilon)
    return {'raw_mse':float(mse),'normalized_mse':float(mse/power_y.clamp_min(epsilon)),'cosine_similarity':float(cos.mean()),'pearson_correlation':float(corr.mean()),'power_ratio':float(power_x/power_y.clamp_min(epsilon))}

def summed_latent_metrics(reconstruction,target):return _flat_metrics(reconstruction.sum(1),target.sum(1))

def oracle_layer_replacements(reconstruction,target):
    layer0=reconstruction.clone();layer0[:,0]=target[:,0];layer7=reconstruction.clone();layer7[:,7]=target[:,7];zero7=reconstruction.clone();zero7[:,7]=0
    return {'all_reconstructed':reconstruction,'oracle_layer0':layer0,'oracle_layer7':layer7,'zero_layer7':zero7,'all_oracle':target}

def relative_waveform_metrics(current,clean):
    out={'delta_waveform_snr_db':current['waveform_snr_db']-clean['waveform_snr_db'],'delta_si_sdr_db':current['si_sdr_db']-clean['si_sdr_db'],'stft_ratio':current['stft_l1']/max(clean['stft_l1'],1e-12)}
    for key in ('stoi','pesq','visqol','speaker_similarity'):
        out[f'delta_{key}']=None if current.get(key) is None or clean.get(key) is None else current[key]-clean[key]
    return out

def classify_identity(*,normalization_max_error,cached_direct_max_error,codec_baseline_reproduced,tolerance=1e-5):
    if normalization_max_error>tolerance:return {'passed':False,'classification':'NORMALIZATION_ROUNDTRIP_BUG'}
    if cached_direct_max_error>tolerance:return {'passed':False,'classification':'LATENT_CACHE_MISMATCH'}
    if not codec_baseline_reproduced:return {'passed':False,'classification':'CODEC_BASELINE_NOT_REPRODUCIBLE'}
    return {'passed':True,'classification':'PASS'}

@dataclass
class CheckpointSelector:
    best_latent:dict[str,Any]|None=None
    best_waveform:dict[str,Any]|None=None
    def update(self,*,step,latent_loss,delta_si_sdr,path):
        row={'step':int(step),'latent_loss':float(latent_loss),'delta_si_sdr':float(delta_si_sdr),'path':str(path)}
        if self.best_latent is None or latent_loss<self.best_latent['latent_loss']:self.best_latent=row.copy()
        if self.best_waveform is None or delta_si_sdr>self.best_waveform['delta_si_sdr']:self.best_waveform=row.copy()

def clean_ladder_conditions(snrs,*,seed):
    rows=[{'stage':'C0','seed':seed},{'stage':'C1','seed':seed}]
    for index,snr in enumerate(snrs):
        paired_seed=int(hashlib.sha256(f'clean_ladder_v1|{seed}|{snr}'.encode()).hexdigest()[:8],16)
        rows.extend({'stage':stage,'snr_db':float(snr),'seed':paired_seed} for stage in ('C2','C3','C4'))
    return rows
