from speech_jscc.diagnostics.conv_conformer_integration import next_stage
def test_progression_stops_without_pass_or_explicit_continue():
    assert next_stage("g1_mapping_train",False,True)=="stop"
    assert next_stage("g1_mapping_train",True,False)=="stop"
    assert next_stage("g1_mapping_train",True,True)=="g2_fixed_clean"
    assert next_stage("g2_fixed_clean",True,True)=="g3_random_clean"
