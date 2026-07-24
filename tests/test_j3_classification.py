import pytest

from speech_jscc.diagnostics.j3_narrowband import (
    aggregate_channel_diagnostics, classify_j3, j3_gate,
    validate_j3_resume_metadata, write_training_curves,
)


def m(imp=.2,corr=.4,power=.2): return {"relative_improvement_over_zero":imp,"pearson_correlation":corr,"cosine_similarity":corr,"power_ratio":power,"finite":True}
def result(): return {"aggregate":m(),"layers1_to_7":m(.12,.3,.12),"layers6_to_7":m(.08,.2,.08),"layer7":m(.05,.1,.05),"channel":{"finite":True,"csi_nmse":.05,"maximum_equalizer_gain":10}}

def test_j3_pass_and_layer7_classification():
    gate=j3_gate(result(),result(),infrastructure={"finite":True,"diversity":True,"coverage":True,"no_leakage":True,"contiguous":True,"metadata":True})
    assert classify_j3(gate,False,0)=="PASS"
    bad=result();bad["layer7"]["relative_improvement_over_zero"]=.01
    gate=j3_gate(bad,bad,infrastructure={"finite":True,"diversity":True,"coverage":True,"no_leakage":True,"contiguous":True,"metadata":True})
    assert classify_j3(gate,False,0)=="FAIL_LAYER7"

def test_mask_and_nonfinite_failures_take_precedence():
    infra={"finite":True,"diversity":True,"coverage":True,"no_leakage":False,"contiguous":True,"metadata":True}
    gate=j3_gate(result(),result(),infrastructure=infra)
    assert classify_j3(gate,False,0)=="FAIL_MASK_IMPLEMENTATION"
    infra={**infra,"finite":False,"no_leakage":True}
    assert classify_j3(j3_gate(result(),result(),infrastructure=infra),False,0)=="FAIL_NONFINITE"

def test_marginal_layer7_and_pilot_overlap_classifications():
    marginal = result()
    marginal["layer7"]["relative_improvement_over_zero"] = .025
    gate = j3_gate(marginal, marginal, infrastructure={"finite":True,"diversity":True,"coverage":True,"no_leakage":True,"contiguous":True,"metadata":True})
    assert classify_j3(gate,False,0) == "MARGINAL_LAYER7"
    gate["pilot_overlap_dominant"] = True
    assert classify_j3(gate,False,0) == "FAIL_PILOT_OVERLAP"

def test_sinr_schema_contains_linear_and_db():
    from speech_jscc.diagnostics.j3_narrowband import sinr_fields
    value=sinr_fields(0.1)
    assert value["post_equalization_sinr_linear"]==.1
    assert value["post_equalization_sinr_db"]==-10.

def test_channel_diagnostic_aggregation_preserves_index_lists():
    rows = [
        {"csi_nmse": 0.1, "contiguous_band_verified": True,
         "narrowband_subcarrier_indices": [2, 3]},
        {"csi_nmse": 0.3, "contiguous_band_verified": True,
         "narrowband_subcarrier_indices": [4, 5]},
    ]
    value = aggregate_channel_diagnostics(rows)
    assert value["csi_nmse"] == pytest.approx(0.2)
    assert value["contiguous_band_verified"] is True
    assert value["narrowband_subcarrier_indices"] == [[2, 3], [4, 5]]

def test_j3_resume_rejects_distribution_or_parent_mismatch():
    saved = {"stage":"j3_random_narrowband", "selected_distribution_hash":"a", "parent_checkpoint_hash":"b"}
    validate_j3_resume_metadata(saved, dict(saved))
    with pytest.raises(ValueError, match="selected_distribution_hash"):
        validate_j3_resume_metadata(saved, {**saved, "selected_distribution_hash":"changed"})

def test_training_curve_writer_creates_reconstruction_plots(tmp_path):
    pytest.importorskip("matplotlib")
    history = [
        {"step":1,"loss":.9,"aggregate":{"power_ratio":.1,"pearson_correlation":.2}},
        {"step":2,"loss":.8,"aggregate":{"power_ratio":.2,"pearson_correlation":.3}},
    ]
    write_training_curves(history, tmp_path)
    assert {path.name for path in tmp_path.glob("*.png")} == {
        "loss_vs_step.png", "power_ratio_vs_step.png", "correlation_vs_step.png"
    }
