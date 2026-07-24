from speech_jscc.diagnostics.conv_conformer_integration import realization_policy
def test_g3_realizations_and_snr_vary_reproducibly():
    rows=[realization_policy("g3_random_clean",23,x) for x in range(1,5)]
    assert len({x["seed"] for x in rows})==4 and len({x["snr_db"] for x in rows})>1
    assert all(5<=x["snr_db"]<=15 and x["jammer"]=="none" for x in rows)
