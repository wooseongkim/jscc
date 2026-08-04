from __future__ import annotations

import pytest

from speech_jscc.training.r4_si_sdr_finetune import base_loss_weights, si_sdr_weight


def test_si_sdr_local_ramp() -> None:
    assert si_sdr_weight(0, target=0.02) == 0.0
    assert si_sdr_weight(249, target=0.02) == 0.0
    assert si_sdr_weight(250, target=0.02) == 0.0
    assert si_sdr_weight(1000, target=0.02) == pytest.approx(0.02)
    assert si_sdr_weight(3000, target=0.0) == 0.0


def test_source_global_ramp_is_not_reset_at_local_zero() -> None:
    config = {"latent_mse_weight": 1.0, "multires_stft_weight": 1.0, "waveform_l1_weight": .1, "channel_free_weight": .2}
    weights = base_loss_weights(5750, config)
    assert weights["stft"] > 0 and weights["waveform"] > 0
    assert weights["si_sdr"] == 0.0
