from __future__ import annotations
import hashlib
from dataclasses import dataclass
import torch

def _hash(value): return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

@dataclass
class LatentNormalizer:
    mean: torch.Tensor
    std: torch.Tensor
    epsilon: float
    metadata: dict
    def normalize(self,value): return (value-self.mean.to(value)[None])/self.std.to(value)[None].clamp_min(self.epsilon)
    def denormalize(self,value): return value*self.std.to(value)[None].clamp_min(self.epsilon)+self.mean.to(value)[None]

def fit_latent_normalizer(values,*,mode,epsilon,split,manifest_hash,cache_hash):
    if split != "train": raise ValueError("latent normalization statistics must use train data only")
    total=None; square=None; samples=0; frames=0; scalar_count=0
    for value in values:
        value=value.detach().float().cpu(); samples+=1; frames+=int(value.shape[1])
        if mode=="per_layer_per_dimension": current=value.sum(1); current_square=value.square().sum(1); scalar_count+=value.shape[1]
        elif mode=="per_layer_scalar": current=value.sum((1,2)); current_square=value.square().sum((1,2)); scalar_count+=value.shape[1]*value.shape[2]
        else: raise ValueError("normalization mode must be per_layer_scalar or per_layer_per_dimension")
        total=current.clone() if total is None else total+current; square=current_square.clone() if square is None else square+current_square
    if total is None: raise ValueError("normalization requires train samples")
    mean=total/scalar_count; variance=square/scalar_count-mean.square(); std=variance.clamp_min(0).sqrt()
    if mode=="per_layer_per_dimension": mean=mean[:,None,:]; std=std[:,None,:]
    else: mean=mean[:,None,None]; std=std[:,None,None]
    metadata={"mode":mode,"epsilon":float(epsilon),"mean_tensor_hash":_hash(mean),"std_tensor_hash":_hash(std),
        "train_manifest_hash":manifest_hash,"manifest_hash":manifest_hash,"latent_cache_hash":cache_hash,"cache_hash":cache_hash,
        "sample_count":samples,"frame_count":frames}
    metadata["normalization_stats_hash"]=hashlib.sha256((metadata["mean_tensor_hash"]+metadata["std_tensor_hash"]+mode+str(epsilon)).encode()).hexdigest()
    return LatentNormalizer(mean,std,float(epsilon),metadata)
