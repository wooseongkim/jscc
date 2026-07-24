import torch
import pytest
from speech_jscc.diagnostics.latent_normalization import LatentNormalizer, fit_latent_normalizer

def test_normalization_round_trip_and_provenance():
    values=[torch.randn(8,3,5) for _ in range(4)]
    normalizer=fit_latent_normalizer(values,mode="per_layer_per_dimension",epsilon=1e-6,
        split="train",manifest_hash="m",cache_hash="c")
    value=values[0].unsqueeze(0)
    assert torch.allclose(normalizer.denormalize(normalizer.normalize(value)),value,atol=1e-5)
    assert normalizer.metadata["sample_count"]==4 and normalizer.metadata["frame_count"]==12
    assert normalizer.metadata["manifest_hash"]=="m"

def test_normalization_rejects_validation_and_test_statistics():
    for split in ("valid","validation","test"):
        with pytest.raises(ValueError,match="train"):
            fit_latent_normalizer([torch.randn(2,3,4)],mode="per_layer_scalar",epsilon=1e-6,split=split,manifest_hash="m",cache_hash="c")
