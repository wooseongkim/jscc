from __future__ import annotations

import torch

from speech_jscc.diagnostics.content_generalization import (
    content_realization_seed,
    summarize_content_metrics,
)


def test_g2_realization_is_fixed_and_g3_changes_each_step() -> None:
    assert content_realization_seed("g2_fixed_clean", 23, 1) == content_realization_seed("g2_fixed_clean", 23, 99)
    assert content_realization_seed("g3_random_clean", 23, 1) != content_realization_seed("g3_random_clean", 23, 2)


def test_content_metrics_include_aggregate_per_layer_and_explicit_layer0() -> None:
    target = torch.ones(1, 2, 2, 3)
    reconstruction = target.clone(); reconstruction[:, 1] *= .5
    result = summarize_content_metrics(reconstruction, target, group="seen")
    assert set(result) >= {"aggregate", "per_layer", "layer0_summary"}
    assert len(result["per_layer"]) == 2
    assert result["layer0_summary"]["layer"] == 0
    assert result["layer0_summary"]["normalized_mse"] == 0.0


def test_scale_only_alignment_does_not_hide_power_ratio() -> None:
    target = torch.randn(1, 2, 2, 3)
    result = summarize_content_metrics(.1 * target, target, group="unseen")
    assert result["aggregate"]["cosine_similarity"] > .99
    assert result["aggregate"]["power_ratio"] < .02
