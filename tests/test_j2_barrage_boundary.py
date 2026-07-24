import json
from pathlib import Path

import pytest

from speech_jscc.diagnostics.j2_barrage import (
    J2_JSR_GRID,
    J2_SNR_GRID,
    aggregate_realizations,
    classify_j2,
    derive_sweep_seed,
    j2_gate,
    select_training_range,
    select_initialization,
    sweep_grid,
    validate_j2_resume_metadata,
    verify_j1_artifact,
)


def metric(improvement=.2, correlation=.4, power=.2):
    return {"relative_improvement_over_zero": improvement,
            "pearson_correlation": correlation, "cosine_similarity": correlation,
            "power_ratio": power, "finite": True}


def validation():
    return {"aggregate": metric(), "layer0": metric(), "layers1_to_7": metric(.12,.3,.1),
            "layers6_to_7": metric(.08,.2,.08), "layer7": metric(.05,.1,.05),
            "channel": {"finite": True, "csi_nmse": .1, "maximum_equalizer_gain": 10.}}


def test_sweep_grid_matches_dense_spec_and_seed_is_deterministic():
    rows=sweep_grid(realizations=3)
    assert len(rows)==len(J2_SNR_GRID)*len(J2_JSR_GRID)*3
    assert {r["snr_db"] for r in rows}==set(J2_SNR_GRID)
    assert {r["jsr_db"] for r in rows}==set(J2_JSR_GRID)
    assert derive_sweep_seed(23,5,-7.5,2)==derive_sweep_seed(23,5,-7.5,2)
    assert derive_sweep_seed(23,5,-7.5,2)!=derive_sweep_seed(23,5,-7.5,3)


def test_aggregation_reports_mean_std_worst_decile_and_worst_sample():
    rows=[{"snr_db":5.,"jsr_db":-5.,"aggregate_normalized_mse":x,"sample_id":str(i)} for i,x in enumerate([.5,.7,.9,1.1])]
    value=aggregate_realizations(rows,["aggregate_normalized_mse"])[0]
    assert value["aggregate_normalized_mse_mean"]==pytest.approx(.8)
    assert value["aggregate_normalized_mse_std"]>0
    assert value["aggregate_normalized_mse_worst_decile"]>=value["aggregate_normalized_mse_mean"]
    assert value["aggregate_normalized_mse_worst_sample"]==1.1


def test_range_selection_begins_at_j1_boundary_and_stops_at_first_enhancement_failure():
    rows=[]
    for jsr,enh,agg in [(-10,.08,.15),(-7.5,.06,.12),(-5,.03,.08),(0,-.02,-.01)]:
        rows.append({"snr_db":5.,"jsr_db":jsr,"aggregate_relative_improvement_over_zero_mean":agg,
                     "layers1_to_7_relative_improvement_over_zero_mean":enh,
                     "layer7_relative_improvement_over_zero_mean":enh/2,
                     "csi_nmse_mean":.02,"maximum_equalizer_gain_mean":8.})
    result=select_training_range(rows)
    assert result["defined"] and result["selected_jsr_range_db"]==[-10.0,-5.0]
    assert result["first_failing_metric"]=="enhancement_layers_improvement"


def test_range_selection_stops_when_no_transition_exists():
    rows=[{"snr_db":5.,"jsr_db":x,"aggregate_relative_improvement_over_zero_mean":.2,
           "layers1_to_7_relative_improvement_over_zero_mean":.1,
           "layer7_relative_improvement_over_zero_mean":.05} for x in J2_JSR_GRID]
    assert not select_training_range(rows)["defined"]


def test_j2_gate_checks_deep_layers_and_classification():
    value=validation(); gate=j2_gate(value,value,randomness_pass=True,coverage_pass=True,parameters_finite=True)
    assert gate["passed"] and classify_j2(gate,final_is_best=False,loss_slope=0)=="PASS"
    value["layer7"]["relative_improvement_over_zero"]=.01
    gate=j2_gate(value,value,randomness_pass=True,coverage_pass=True,parameters_finite=True)
    assert not gate["passed"]
    assert classify_j2(gate,final_is_best=False,loss_slope=0)=="FAIL_ENHANCEMENT_COLLAPSE"


def test_nonfinite_and_not_converged_classifications_take_precedence():
    value=validation(); gate=j2_gate(value,value,randomness_pass=True,coverage_pass=True,parameters_finite=False)
    assert classify_j2(gate,final_is_best=True,loss_slope=-1e-3)=="FAIL_NONFINITE"
    value=validation(); value["aggregate"]["relative_improvement_over_zero"]=.09
    gate=j2_gate(value,value,randomness_pass=True,coverage_pass=True,parameters_finite=True)
    assert classify_j2(gate,final_is_best=True,loss_slope=-1e-3)=="INCONCLUSIVE_NOT_CONVERGED"


def test_j1_verification_rejects_mutation_and_wrong_stage(tmp_path):
    summary=tmp_path/"summary.json"; checkpoint=tmp_path/"diagnostic_last.pt"
    summary.write_text(json.dumps({"gate":{"stage_pass":True},"provenance":{"stage_name":"j1_weak_random_barrage","model_architecture":"conv_conformer_v1","subset_size":"256"}}))
    checkpoint.write_bytes(b"checkpoint")
    verified=verify_j1_artifact(summary,checkpoint)
    assert verified["summary_sha256"] and verified["checkpoint_sha256"]
    summary.write_text(summary.read_text()+"\n")
    with pytest.raises(ValueError,match="hash"):
        verify_j1_artifact(summary,checkpoint,expected=verified)


def test_initialization_selection_uses_unseen_and_deep_layer_evidence():
    fresh={"unseen_loss":.8,"layers1_to_7_improvement":.08,"layers6_to_7_improvement":.05,"layer7_improvement":.03,"gradient_finite_ratio":1.,"output_power_ratio":.15}
    transfer={**fresh,"unseen_loss":.7,"layers1_to_7_improvement":.12,"layers6_to_7_improvement":.08,"layer7_improvement":.05}
    assert select_initialization(fresh,transfer)["selected_initialization"]=="j1_transfer"


def test_resume_rejects_range_or_initialization_mismatch():
    saved={"stage":"j2_strong_barrage","initialization_mode":"fresh","selected_range_hash":"abc"}
    validate_j2_resume_metadata(saved,dict(saved))
    with pytest.raises(ValueError,match="initialization_mode"):
        validate_j2_resume_metadata(saved,{**saved,"initialization_mode":"j1_transfer"})
