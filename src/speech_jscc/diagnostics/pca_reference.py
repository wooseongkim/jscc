from __future__ import annotations
import hashlib
import torch

def _hash(value): return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

class PerLayerPCAReference:
    diagnostic_type="offline_linear_reference"
    def __init__(self,components=480,seed=23): self.requested_components=int(components); self.seed=int(seed); self.metadata={}
    def fit(self,values,*,split="train"):
        if split != "train": raise ValueError("PCA must be fit on train data only")
        matrix=values.detach().float().cpu() if torch.is_tensor(values) else torch.stack([x.detach().float().cpu() for x in values])
        if matrix.ndim != 4: raise ValueError("PCA input must be [N,L,T,D]")
        n,layers=matrix.shape[:2]
        if n < self.requested_components: raise ValueError(f"PCA requires at least {self.requested_components} training samples")
        flattened=matrix.flatten(2); means=[]; components=[]; singular=[]; explained=[]
        torch.manual_seed(self.seed)
        for layer in range(layers):
            current=flattened[:,layer]; mean=current.mean(0); centered=current-mean
            rank=min(self.requested_components,centered.shape[0],centered.shape[1])
            _,s,v=torch.pca_lowrank(centered,q=rank,center=False,niter=4)
            means.append(mean); components.append(v.T); singular.append(s)
            variance=s.square()/max(n-1,1); total=centered.square().sum()/max(n-1,1); explained.append(variance/total.clamp_min(1e-12))
        max_rank=max(x.shape[0] for x in components)
        if any(x.shape[0] != max_rank for x in components): raise ValueError("all PCA layers must have equal fitted rank")
        self.mean=torch.stack(means); self.components=torch.stack(components); self.singular_values=torch.stack(singular); self.explained_variance_ratio=torch.stack(explained)
        self.original_shape=tuple(matrix.shape[2:]); self.metadata={"diagnostic_type":self.diagnostic_type,"method":"randomized_low_rank_svd",
            "requested_components":self.requested_components,"fitted_components":max_rank,"fitted_sample_count":n,
            "training_mean_hash":_hash(self.mean),"pca_component_hash":_hash(self.components)}
        return self
    def reconstruct(self,values):
        matrix=values.detach().float(); flat=matrix.flatten(2); mean=self.mean.to(matrix); comp=self.components.to(matrix)
        centered=flat-mean[None]; coefficients=torch.einsum("nld,lrd->nlr",centered,comp)
        result=torch.einsum("nlr,lrd->nld",coefficients,comp)+mean[None]
        return result.reshape_as(matrix)
