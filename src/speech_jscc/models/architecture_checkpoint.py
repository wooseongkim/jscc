from __future__ import annotations
from pathlib import Path
import torch

FIELDS=("model_architecture","architecture_version","normalization_mode","normalization_stats_hash","d_model",
 "encoder_conformer_blocks","decoder_conformer_blocks","num_attention_heads","ffn_expansion","convolution_kernel_size",
 "layer_mixer_blocks","symbol_frames","complex_channels_per_symbol_frame","representation_shape","total_data_channel_uses",
 "per_layer_channel_uses","resource_mapping_version","train_manifest_hash","latent_cache_hash")

def architecture_metadata(model,preprocessing,train_manifest_hash,latent_cache_hash):
    config=getattr(model,"model_config",{})
    return {"model_architecture":getattr(model,"architecture",None),"architecture_version":getattr(model,"architecture_version",None),
        "normalization_mode":preprocessing.get("normalization_mode","none"),"normalization_stats_hash":preprocessing.get("normalization_stats_hash","none"),
        **{key:config.get(key) for key in FIELDS if key in config},
        "representation_shape":list(model.encoder.representation_shape),"total_data_channel_uses":model.encoder.total_channel_uses,
        "per_layer_channel_uses":list(model.encoder.layer_channel_uses),"resource_mapping_version":"pilot_reserved_v1",
        "train_manifest_hash":train_manifest_hash,"latent_cache_hash":latent_cache_hash}

def validate_architecture_metadata(actual,expected):
    if not isinstance(actual,dict): raise ValueError("checkpoint architecture metadata is required")
    for key,value in expected.items():
        if actual.get(key) != value:
            label="normalization metadata" if key.startswith("normalization") else f"architecture metadata {key}"
            raise ValueError(f"incompatible {label}")

def load_architecture_checkpoint(path,model,*,expected_metadata,strict=True):
    if strict is not True: raise ValueError("partial or strict=False checkpoint loading is forbidden")
    payload=torch.load(Path(path),map_location="cpu",weights_only=False)
    validate_architecture_metadata(payload.get("architecture_metadata"),expected_metadata)
    model.load_state_dict(payload["model"],strict=True); return payload
