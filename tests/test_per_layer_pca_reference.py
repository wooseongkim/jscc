import torch
from speech_jscc.diagnostics.pca_reference import PerLayerPCAReference

def test_pca_analytic_projection_and_component_budget():
    base=torch.tensor([[1.,0.,0.,0.],[-1.,0.,0.,0.],[2.,0.,0.,0.],[-2.,0.,0.,0.]])
    values=base[:,None,None,:].repeat(1,2,1,1)
    pca=PerLayerPCAReference(components=1).fit(values,split="train")
    reconstructed=pca.reconstruct(values)
    assert torch.allclose(reconstructed,values,atol=1e-5)
    assert pca.components.shape==(2,1,4)
    assert pca.metadata["method"]=="randomized_low_rank_svd"

def test_pca_configured_480_components_without_covariance(monkeypatch):
    pca=PerLayerPCAReference(components=480)
    assert pca.requested_components==480
    assert not hasattr(pca,"covariance")
