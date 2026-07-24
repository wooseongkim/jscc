from speech_jscc.diagnostics.architecture_screening import revised_g0_gate

def metric(loss=.8,corr=.2,cos=.2,power=.1,zero=1.,global_mean=.9,finite=True):
    return {"aggregate":{"normalized_mse":loss,"pearson_correlation":corr,"cosine_similarity":cos,"power_ratio":power,"finite":finite},
        "layers1_to_7_summary":{"normalized_mse":loss,"pearson_correlation":corr,"cosine_similarity":cos,"power_ratio":power},
        "per_layer":[{"normalized_mse":loss,"pearson_correlation":corr,"cosine_similarity":cos,"power_ratio":power}],
        "baselines":{"zero":zero,"global_mean":global_mean,"layerwise_mean":.95}}

def test_gate_fails_when_only_layer_zero_passes():
    value=metric(); value["layers1_to_7_summary"].update(normalized_mse=.99,pearson_correlation=0.,cosine_similarity=0.,power_ratio=.001)
    result=revised_g0_gate({"same":value,"unseen":value},same_group="same",unseen_group="unseen")
    assert result["layer0_generalization_pass"] and not result["enhancement_layers_generalization_pass"] and not result["architecture_screening_pass"]

def test_gate_requires_all_content_and_mean_conditions():
    result=revised_g0_gate({"same":metric(),"unseen":metric()},same_group="same",unseen_group="unseen")
    assert all(result.values())
