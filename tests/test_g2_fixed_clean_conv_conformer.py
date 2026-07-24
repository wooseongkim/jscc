from speech_jscc.diagnostics.conv_conformer_integration import realization_policy
def test_g2_is_fixed_estimated_and_jammer_free():
    a=realization_policy("g2_fixed_clean",23,1); b=realization_policy("g2_fixed_clean",23,99)
    assert a==b and a["estimator"]=="dft_tap_ls" and a["equalizer"]=="estimated_zf" and a["jammer"]=="none"
    assert not a["oracle_neural_input"]
