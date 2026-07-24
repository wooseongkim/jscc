from speech_jscc.diagnostics.j4_burst import classify_j4, j4_gate, tail_statistics, validate_j4_resume_metadata
import pytest

def m(imp=.2,corr=.4,power=.2):return {"relative_improvement_over_zero":imp,"pearson_correlation":corr,"cosine_similarity":corr,"power_ratio":power,"finite":True}
def result():return {"aggregate":m(),"layers1_to_7":m(.12,.3,.12),"layers6_to_7":m(.08,.2,.08),"layer7":m(.05,.1,.05),"channel":{"csi_nmse":.05,"maximum_equalizer_gain":10}}

def test_tail_statistics_and_marginal_tail():
    stats=tail_statistics([.2,.1,.04,-.01])
    assert stats["negative_rate"]==.25 and stats["below_5_percent_rate"]==.5
    gate=j4_gate(result(),result(),infrastructure={"finite":True,"diversity":True,"coverage":True,"no_leakage":True,"contiguous":True,"full_band":True,"metadata":True},tail={"layer7_improvement_p10":-.01,"layer7_negative_rate":.05})
    assert classify_j4(gate,False,0)=="MARGINAL_TAIL"

def test_j4_pass_and_layer7_failure():
    tail={"layer7_improvement_p10":.01,"layer7_negative_rate":0.}
    infra={"finite":True,"diversity":True,"coverage":True,"no_leakage":True,"contiguous":True,"full_band":True,"metadata":True}
    gate=j4_gate(result(),result(),infrastructure=infra,tail=tail);assert classify_j4(gate,False,0)=="PASS"
    bad=result();bad["layer7"]["relative_improvement_over_zero"]=.01
    assert classify_j4(j4_gate(bad,bad,infrastructure=infra,tail=tail),False,0)=="FAIL_LAYER7"

def test_j4_resume_rejects_parent_or_distribution_change():
    saved={"stage":"j4_random_burst","selected_distribution_hash":"a","parent_checkpoint_hash":"b","accepted_manifest_hash":"c"}
    validate_j4_resume_metadata(saved,dict(saved))
    with pytest.raises(ValueError,match="parent_checkpoint_hash"):validate_j4_resume_metadata(saved,{**saved,"parent_checkpoint_hash":"x"})
