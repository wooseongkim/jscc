from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.summarize_o5_root_cause import build_report_data, record_at_step
from speech_jscc.diagnostics.o5_root_cause import optimal_scale_diagnostics


def test_layerwise_stage1_rescaled_loss_is_not_global_power_weighted_nmse() -> None:
    target = torch.tensor([[[[1.0, 1.0]], [[10.0, 10.0]]]])
    reconstruction = torch.tensor([[[[0.5, 0.5]], [[20.0, 20.0]]]])

    result = optimal_scale_diagnostics(
        reconstruction, target, epsilon=1e-8, layer_weights=torch.ones(2)
    )

    assert result["stage1_layerwise_rescaled_loss"] == pytest.approx(0.0, abs=1e-10)
    assert result["global_power_weighted_rescaled_nmse"] == pytest.approx(0.0055658286)
    assert result["global_power_weighted_rescaled_nmse"] != result["stage1_layerwise_rescaled_loss"]


def test_record_at_step_requires_exact_step() -> None:
    rows = [{"step": 0}, {"step": 500}, {"step": 1000}]
    assert record_at_step(rows, 500)["step"] == 500
    with pytest.raises(ValueError, match="exact step 250"):
        record_at_step(rows, 250)


def test_historical_per_layer_scales_reconstruct_stage1_layerwise_loss(tmp_path: Path) -> None:
    condition = tmp_path / "clean_awgn_reference"; condition.mkdir()
    record = {"step": 500, "loss": .2, "aggregate_power_ratio": .7,
              "aggregate_pearson_correlation": .8,
              "optimal_scale": {"aggregate": {"rescaled_normalized_mse": .3},
                                "per_layer": [{"rescaled_normalized_mse": .1},
                                              {"rescaled_normalized_mse": .5}]}}
    (condition / "metrics.jsonl").write_text(json.dumps(record) + "\n")
    paired, _ = build_report_data(tmp_path)
    assert paired[0]["global_power_weighted_rescaled_nmse"] == pytest.approx(.3)
    assert paired[0]["stage1_layerwise_rescaled_loss"] == pytest.approx(.3)


def test_report_data_uses_metrics_step_500_not_latest_summary(tmp_path: Path) -> None:
    condition = tmp_path / "full_barrage_estimated_csi"
    condition.mkdir()
    rows = [
        {"step": 500, "loss": 0.25, "aggregate_power_ratio": 0.6,
         "aggregate_pearson_correlation": 0.8},
        {"step": 1000, "loss": 0.02, "aggregate_power_ratio": 0.96,
         "aggregate_pearson_correlation": 0.99},
    ]
    (condition / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    (condition / "summary.json").write_text(json.dumps({"condition": condition.name, "steps": 1000,
                                                          "final_loss": 0.02}))
    hashes = {"latent_target": "a", "initial_model_parameters": "b", "legitimate_channel": "c",
              "awgn": "d", "raw_jammer_waveform": "e", "jammer_channel": "f",
              "jammer_mask": "g", "pilot_mask": "h"}
    (condition / "fixed_realization_hashes.json").write_text(json.dumps(hashes))

    paired, extensions = build_report_data(tmp_path, paired_step=500)

    assert paired[0]["final_loss"] == pytest.approx(0.25)
    assert extensions[0]["base_steps"] == 500
    assert extensions[0]["extended_steps"] == 1000
    assert extensions[0]["extended_final_loss"] == pytest.approx(0.02)


def test_extension_rejects_hash_or_lineage_mismatch(tmp_path: Path) -> None:
    condition = tmp_path / "full_barrage_estimated_csi"
    condition.mkdir()
    rows = [{"step": 500, "loss": 0.25}, {"step": 1000, "loss": 0.02}]
    (condition / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    (condition / "summary.json").write_text(json.dumps({"condition": condition.name, "steps": 1000,
                                                          "final_loss": 0.02,
                                                          "extension_base_hashes": {"latent_target": "other"}}))
    (condition / "fixed_realization_hashes.json").write_text(json.dumps({"latent_target": "current"}))

    _, extensions = build_report_data(tmp_path, paired_step=500)

    assert extensions == []
