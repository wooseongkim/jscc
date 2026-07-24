from __future__ import annotations
import torch
from speech_jscc.metrics.audio_quality import align_waveforms,summarize_audio_metrics

def waveform_metrics(reference,estimate,sample_rate):
    reference,estimate=align_waveforms(reference,estimate);noise=(reference-estimate).square().mean(-1);snr=10*torch.log10(reference.square().mean(-1).clamp_min(1e-12)/noise.clamp_min(1e-12));distances=[]
    for size in (256,512,1024):
        window=torch.hann_window(size,device=reference.device,dtype=reference.dtype);a=torch.stft(reference,size,size//4,window=window,return_complex=True).abs();b=torch.stft(estimate,size,size//4,window=window,return_complex=True).abs();distances.append((a-b).abs().mean())
    audio=summarize_audio_metrics(reference,estimate,sample_rate)
    return {'waveform_snr_db':float(snr.mean()),'si_sdr_db':audio['si_sdr_db'],'stft_l1':float(distances[1]),'multi_resolution_stft_distance':float(torch.stack(distances).mean()),'waveform_correlation':float(torch.corrcoef(torch.stack((reference.flatten(),estimate.flatten())))[0,1]),'output_rms':float(estimate.square().mean().sqrt()),'output_peak':float(estimate.abs().max()),'output_dc_mean':float(estimate.mean()),'stoi':audio['stoi'],'stoi_available':audio['stoi_available'],'stoi_error':audio['stoi_error'],'pesq':None,'visqol':None,'speaker_similarity':None}
