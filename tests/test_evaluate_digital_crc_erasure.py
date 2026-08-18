from __future__ import annotations

import pytest

from speech_jscc.evaluation.evaluate_digital_crc_erasure import (
    paired_condition_key,
    resolve_r4_fixed_profile,
    validate_csi_only_allocation,
)


def test_paired_condition_key_excludes_method_but_keeps_all_channel_seeds() -> None:
    base = {
        "sample_id": "a.wav", "crop_offset": 0, "snr_db": 5.0, "jsr_db": 10.0,
        "jammer_type": "broadband", "realization_index": 1,
        "channel_seed": 1, "noise_seed": 2, "jammer_seed": 3,
    }
    assert paired_condition_key({**base, "method": "proposed_jscc"}) == paired_condition_key({**base, "method": "digital_crc_erasure"})


def test_csi_only_config_rejects_any_risk_or_interference_input() -> None:
    validate_csi_only_allocation({"enabled": True, "mode": "csi_only", "delay_ttis": 1})
    with pytest.raises(ValueError, match="CSI-only"):
        validate_csi_only_allocation({"enabled": True, "mode": "delayed_rx_interference", "delay_ttis": 1})


def test_fixed_profile_is_loaded_from_artifact_not_defaulted() -> None:
    profile = resolve_r4_fixed_profile(
        "runs/waveform_aware_wireless/r4_broadband_uep_optimization/stage1_screen/selected_profiles.json"
    )
    assert profile.repetition == (3, 4, 3, 1, 5, 1, 4, 3)
    assert profile.power_share is not None
