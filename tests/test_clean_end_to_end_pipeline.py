import torch
from src.evaluation.clean_end_to_end import (neutral_observable_state,normalization_roundtrip_metrics,
    ideal_ofdm_roundtrip,summed_latent_metrics,oracle_layer_replacements,relative_waveform_metrics,classify_identity)

def test_normalization_roundtrip_is_float32_identity():
    value=torch.randn(2,8,5,7);mean=torch.randn(8,1,7);std=torch.rand(8,1,7)+.1
    result=normalization_roundtrip_metrics(value,mean,std,1e-6)
    assert result['max_abs_error']<=1e-5 and result['restored'].shape==value.shape

def test_neutral_observable_state_uses_existing_schema():
    state=neutral_observable_state(2,device='cpu')
    assert state.shape==(2,8);assert torch.isfinite(state).all();assert torch.allclose(state[0],torch.tensor([0.,0.,0.,0.,0.,0.,-3.,0.]))

def test_ideal_ofdm_roundtrip_recovers_all_data_without_pilots():
    symbols=torch.randn(2,1920,dtype=torch.complex64)
    result=ideal_ofdm_roundtrip(symbols,subcarriers=64,ofdm_symbols=32,pilot_spacing=4)
    assert result['pilot_count']==128 and result['data_count']==1920
    assert result['max_recovery_error']<=1e-6 and result['pilot_leakage']==0

def test_summed_latent_and_oracle_replacements_preserve_order():
    target=torch.randn(2,8,5,7);recon=target*.5
    metrics=summed_latent_metrics(recon,target)
    assert abs(metrics['power_ratio']-.25)<1e-5
    variants=oracle_layer_replacements(recon,target)
    assert torch.equal(variants['oracle_layer0'][:,0],target[:,0]);assert torch.equal(variants['oracle_layer7'][:,7],target[:,7])

def test_relative_waveform_metrics_compute_deltas_and_ratio():
    current={'waveform_snr_db':1.,'si_sdr_db':2.,'stft_l1':.4,'stoi':None}
    clean={'waveform_snr_db':3.,'si_sdr_db':5.,'stft_l1':.2,'stoi':None}
    out=relative_waveform_metrics(current,clean)
    assert out['delta_waveform_snr_db']==-2 and out['delta_si_sdr_db']==-3 and out['stft_ratio']==2

def test_cached_direct_latent_mismatch_blocks_later_training():
    result=classify_identity(normalization_max_error=0.,cached_direct_max_error=4.0,codec_baseline_reproduced=True)
    assert result=={'passed':False,'classification':'LATENT_CACHE_MISMATCH'}
