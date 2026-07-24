from __future__ import annotations
import copy
import random
from collections import defaultdict
from pathlib import Path
import torch
from torch.nn import functional as F
from speech_jscc.diagnostics.content_generalization import parse_speaker_id

def multi_resolution_stft_loss(reconstruction,target,*,fft_sizes=(256,512,1024)):
    losses=[]
    for size in fft_sizes:
        hop=size//4;window=torch.hann_window(size,device=target.device,dtype=target.dtype)
        a=torch.stft(reconstruction,n_fft=size,hop_length=hop,window=window,return_complex=True).abs();b=torch.stft(target,n_fft=size,hop_length=hop,window=window,return_complex=True).abs();losses.append((a-b).abs().mean())
    return torch.stack(losses).mean()

def negative_si_sdr_loss(reconstruction,target,epsilon=1e-8):
    target=target-target.mean(-1,keepdim=True);reconstruction=reconstruction-reconstruction.mean(-1,keepdim=True);scale=(reconstruction*target).sum(-1,keepdim=True)/target.square().sum(-1,keepdim=True).clamp_min(epsilon);signal=scale*target;noise=reconstruction-signal;return -(10*torch.log10((signal.square().sum(-1)+epsilon)/(noise.square().sum(-1)+epsilon))).mean()

def summed_latent_loss(reconstruction,target,epsilon=1e-6):
    a=reconstruction.sum(1);b=target.sum(1);return (a-b).square().mean()/b.square().mean().clamp_min(epsilon)

def feasibility_classification(*,delta_si_sdr,delta_waveform_snr,stft_ratio):
    if delta_si_sdr>=-1 and delta_waveform_snr>=-1 and stft_ratio<=1.2:return 'CHANNEL_FREE_FEASIBLE'
    if -3<=delta_si_sdr<-1:return 'MARGINAL'
    return 'CHANNEL_FREE_INFEASIBLE'

def configure_bottleneck(config,channel_uses):
    output=copy.deepcopy(config);per_layer=int(channel_uses)//8;frames=int(output['model']['symbol_frames'])
    if channel_uses%8 or per_layer%frames:raise ValueError('channel uses must divide into 8 layers and symbol frames')
    output['model']['channel_uses']=int(channel_uses);output['model']['complex_channels_per_symbol_frame']=per_layer//frames;return output

def select_unseen_speaker_paths(train_paths,validation_paths,*,limit,seed):
    speakers={parse_speaker_id(path) for path in train_paths};groups=defaultdict(list)
    for path in validation_paths:
        speaker=parse_speaker_id(path)
        if speaker not in speakers:groups[speaker].append(Path(path))
    rng=random.Random(seed)
    for values in groups.values():rng.shuffle(values)
    keys=sorted(groups);rng.shuffle(keys);result=[]
    while len(result)<limit and any(groups.values()):
        for speaker in keys:
            if groups[speaker]:result.append(groups[speaker].pop())
            if len(result)>=limit:break
    return result

def enable_frozen_rnn_backward(module):
    """Enable cuDNN RNN backward while keeping every frozen weight unchanged."""
    count=0
    for child in module.modules():
        if isinstance(child,torch.nn.RNNBase):child.train(True);count+=1
    if any(parameter.requires_grad for parameter in module.parameters()):raise ValueError('codec module must be frozen before enabling RNN backward')
    return count

def decode_frozen_representation_with_gradient(codec,representation):
    """Decode continuous SpeechTokenizer embeddings without its eval reset."""
    model=getattr(codec,'model',None)
    if model is None:return codec.decode_representation(representation)
    if representation.ndim!=4 or tuple(representation.shape[1:])!=tuple(codec.representation_shape):raise ValueError('representation shape mismatch')
    model.eval();enable_frozen_rnn_backward(model.decoder);quantized=representation.permute(0,1,3,2).sum(dim=1);waveform=model.decoder(quantized).squeeze(1);length=int(codec.waveform_samples)
    if waveform.shape[-1]>length:waveform=waveform[...,:length]
    elif waveform.shape[-1]<length:waveform=F.pad(waveform,(0,length-waveform.shape[-1]))
    return waveform
