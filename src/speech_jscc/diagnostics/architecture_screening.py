from __future__ import annotations
import hashlib
import torch

def direct_bypass(model,representation,state,*,layer_gates=None,layer_power_allocation=None,normalizer=None):
    source=normalizer.normalize(representation) if normalizer is not None else representation
    symbols=model.encoder(source,state,layer_gates=layer_gates,layer_power_allocation=layer_power_allocation)
    reconstruction=model.decoder(symbols,state,layer_gates=layer_gates)
    return normalizer.denormalize(reconstruction) if normalizer is not None else reconstruction

def _passes(row):
    zero=float(row["baselines"]["zero"]); aggregate=row["aggregate"]
    return ((zero-float(aggregate["normalized_mse"]))/max(zero,1e-12)>=.05 and
        float(aggregate["pearson_correlation"])>.05 and float(aggregate["cosine_similarity"])>0 and
        float(aggregate["power_ratio"])>=.01 and bool(aggregate.get("finite",True)))

def _enhancement_pass(row):
    value=row["layers1_to_7_summary"]
    zero_rows=row.get("baseline_per_layer",{}).get("zero")
    zero=sum(x["normalized_mse"] for x in zero_rows[1:])/7 if zero_rows else 1.0
    return ((zero-float(value["normalized_mse"]))/max(zero,1e-12)>=.05 and
        float(value["pearson_correlation"])>.05 and float(value["cosine_similarity"])>0 and float(value["power_ratio"])>=.01)

def revised_g0_gate(validation,*,same_group,unseen_group):
    rows=[validation[same_group],validation[unseen_group]]
    layer0=all(((r["baselines"]["zero"]-r["per_layer"][0]["normalized_mse"])/max(r["baselines"]["zero"],1e-12)>=.05 and
        r["per_layer"][0]["pearson_correlation"]>0 and r["per_layer"][0]["cosine_similarity"]>0 and r["per_layer"][0]["power_ratio"]>=.01) for r in rows)
    enhancement=all(_enhancement_pass(r) for r in rows); same=_passes(rows[0]); unseen=_passes(rows[1])
    beats_global=all(r["aggregate"]["normalized_mse"]<r["baselines"]["global_mean"] for r in rows)
    beats_layerwise=all(r["aggregate"]["normalized_mse"]<r["baselines"]["layerwise_mean"] for r in rows)
    aggregate=all(_passes(r) for r in rows); finite=all(r["aggregate"].get("finite",True) for r in rows)
    return {"layer0_generalization_pass":layer0,"enhancement_layers_generalization_pass":enhancement,
        "aggregate_generalization_pass":aggregate,"beats_global_mean_predictor":beats_global,
        "beats_layerwise_mean_predictor":beats_layerwise,"same_speaker_generalization_pass":same,
        "unseen_speaker_generalization_pass":unseen,"architecture_screening_pass":layer0 and enhancement and same and unseen and beats_global and finite}

def parameter_report(model,codec=None):
    def count(module): return sum(p.numel() for p in module.parameters() if p.requires_grad)
    encoder=count(model.encoder); decoder=count(model.decoder); total=encoder+decoder
    largest=sorted(({"name":name,"shape":list(value.shape),"parameters":value.numel()} for name,value in model.named_parameters()),key=lambda x:x["parameters"],reverse=True)[:10]
    is_conformer=getattr(model,"architecture",None)=="conv_conformer_v1"
    if is_conformer and any(isinstance(module,torch.nn.Linear) and module.out_features==8*50*1024 for module in model.modules()):
        raise ValueError("giant flatten-output parameter tensor detected")
    if is_conformer and total>=30_000_000: raise ValueError("Conv-Conformer exceeds 30 million trainable parameters")
    return {"encoder_trainable_parameters":encoder,"decoder_trainable_parameters":decoder,"total_trainable_parameters":total,
        "speech_tokenizer_trainable_parameters":0 if codec is None else count(codec),"estimated_parameter_memory_bytes_fp32":total*4,"largest_parameter_tensors":largest}
