from __future__ import annotations

import torch
import pytest

from speech_jscc.models import SpeechJSCC
from speech_jscc.training.stage1 import (
    build_stage1_checkpoint_payload,
    stage1_metadata,
    validate_stage1_checkpoint_resources,
)


def test_stage1_checkpoint_metadata_identifies_fixed_tx_policy() -> None:
    config = {
        "codec": {"type": "mock", "sample_rate": 16000, "waveform_samples": 16000, "n_q": 8},
        "model": {"layers": 8, "frames": 50, "latent_dim": 1024},
        "channel": {
            "fading": "multipath_block",
            "num_taps": 6,
            "pdp": "exponential",
            "pdp_decay": 0.7,
            "channel_estimator": "dft_tap_ls",
            "estimator_num_taps": 6,
            "estimator_ridge_lambda": 1e-6,
        },
    }

    metadata = stage1_metadata(
        config,
        representation_shape=(8, 50, 1024),
        layer_weights=[1.0] * 8,
        representation_source="mock",
    )

    assert metadata["training_stage"]["label"] == "fixed_tx_channel_aware_rx_jammer_agnostic"
    assert metadata["receiver_policy"]["state_mode"] == "observable_v1"
    assert metadata["receiver_policy"]["uses_true_channel"] is False
    assert metadata["receiver_policy"]["uses_true_jammer_type"] is False
    assert metadata["channel_estimator"]["name"] == "dft_tap_ls"


def test_stage1_checkpoint_payload_excludes_disabled_modules() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    payload = build_stage1_checkpoint_payload(
        model,
        optimizer,
        step=3,
        best_metric=1.25,
        config={"seed": 1},
        metadata={"training_stage": {"name": "stage1_fixed_tx"}},
    )

    assert "learned_gate" not in payload
    assert "latent_refiner" not in payload
    assert "jammer_estimator" not in payload
    assert payload["step"] == 3
    assert "rng_state" in payload


def test_legacy_2048_symbol_checkpoint_is_strictly_rejected() -> None:
    model = SpeechJSCC((8, 50, 1024), 1920, channel_state_dim=8, hidden_dim=2)
    legacy = {"metadata": {"representation_shape": [8, 50, 1024]}}
    config = {"model": {"channel_uses": 1920, "grid_shape": [64, 32]}}

    with pytest.raises(ValueError, match="pilot_reserved_v1"):
        validate_stage1_checkpoint_resources(legacy, model, config)


def test_pilot_reserved_metadata_records_exact_resource_mapping() -> None:
    config = {
        "codec": {"type": "mock"},
        "model": {"layers": 8, "grid_shape": [64, 32], "channel_uses": 1920},
        "channel": {"pilot_spacing": 4, "pilot_time_spacing": 4},
    }
    metadata = stage1_metadata(
        config,
        representation_shape=(8, 50, 1024),
        layer_weights=[1.0] * 8,
        representation_source="test",
    )
    mapping = metadata["resource_mapping"]
    assert mapping == {
        "version": "pilot_reserved_v1",
        "grid_shape": [64, 32],
        "grid_total_resources": 2048,
        "pilot_resources": 128,
        "data_channel_uses": 1920,
        "per_layer_channel_uses": [240] * 8,
        "packing_order": "row_major_nonpilot",
        "pilot_overwrite_count": 0,
    }
