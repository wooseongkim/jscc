import pytest
from speech_jscc.diagnostics.conv_conformer_integration import validate_stage_metadata
def test_stage_metadata_rejects_architecture_stage_and_legacy_mapping():
    expected={"diagnostic_stage":"g2_fixed_clean","model_architecture":"conv_conformer_v1","resource_mapping_version":"pilot_reserved_v1"}
    validate_stage_metadata(expected,expected)
    for key,value in (("diagnostic_stage","g1_mapping_train"),("model_architecture","flat_mlp"),("resource_mapping_version","legacy_zero_fill_v0")):
        with pytest.raises(ValueError): validate_stage_metadata({**expected,key:value},expected)
