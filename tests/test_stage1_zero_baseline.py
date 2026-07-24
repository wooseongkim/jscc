from __future__ import annotations

import math

import torch

from speech_jscc.diagnostics.metrics import (
    aggregate_latent_rows,
    latent_metric_rows,
    normalized_layer_loss,
    zero_predictor_loss,
)


def test_zero_reconstruction_has_unit_per_layer_nmse() -> None:
    target = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])

    loss, layers = zero_predictor_loss(target, torch.ones(2), 1.0e-6)

    torch.testing.assert_close(layers, torch.ones(2))
    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_perfect_reconstruction_has_zero_normalized_loss() -> None:
    target = torch.randn(2, 3, 4, 5)

    torch.testing.assert_close(
        normalized_layer_loss(target, target, 1.0e-6),
        torch.zeros(3),
    )


def test_scaled_target_has_analytic_normalized_loss() -> None:
    target = torch.randn(2, 3, 4, 5)

    actual = normalized_layer_loss(0.25 * target, target, 1.0e-6)

    torch.testing.assert_close(actual, torch.full((3,), 0.75**2))


def test_latent_metric_rows_record_required_finite_fields() -> None:
    target = torch.tensor(
        [
            [[[1.0, 2.0, 3.0]], [[-1.0, 0.5, 2.0]]],
            [[[2.0, 3.0, 4.0]], [[-2.0, 1.0, 3.0]]],
        ]
    )
    reconstruction = 0.5 * target

    rows = latent_metric_rows(
        reconstruction,
        target,
        epsilon=1.0e-6,
        predictor="trained",
        scenario="clean_snr10",
        sample_ids=["a", "b"],
    )

    assert len(rows) == 4
    required = {
        "target_power",
        "reconstruction_power",
        "power_ratio",
        "raw_mse",
        "normalized_mse",
        "zero_normalized_mse",
        "trained_minus_zero_normalized_mse",
        "cosine_similarity",
        "pearson_correlation",
        "target_mean",
        "reconstruction_mean",
        "target_std",
        "reconstruction_std",
        "reconstruction_bias",
        "near_zero_fraction",
        "finite",
    }
    assert required.issubset(rows[0])
    assert rows[0]["sample_id"] == "a"
    assert rows[0]["predictor"] == "trained"
    assert rows[0]["scenario"] == "clean_snr10"
    assert rows[0]["power_ratio"] == 0.25
    assert rows[0]["normalized_mse"] == 0.25
    assert rows[0]["cosine_similarity"] > 0.999
    assert rows[0]["pearson_correlation"] > 0.999
    assert rows[0]["finite"] is True
    for row in rows:
        for key in required - {"finite"}:
            assert math.isfinite(float(row[key]))


def test_degenerate_reconstruction_correlation_is_zero_and_flagged() -> None:
    target = torch.tensor([[[[1.0, 2.0, 3.0]]]])
    reconstruction = torch.zeros_like(target)

    row = latent_metric_rows(
        reconstruction,
        target,
        epsilon=1.0e-6,
        predictor="zero",
        scenario="test",
    )[0]

    assert row["pearson_correlation"] == 0.0
    assert row["correlation_degenerate"] is True
    assert row["zero_normalized_mse"] == 1.0
    assert row["normalized_mse"] == 1.0


def test_aggregate_latent_rows_preserve_group_boundaries() -> None:
    target = torch.arange(1, 13, dtype=torch.float32).reshape(2, 2, 1, 3)
    rows = latent_metric_rows(
        0.5 * target,
        target,
        epsilon=1.0e-6,
        predictor="trained",
        scenario="clean",
    )

    aggregate = aggregate_latent_rows(rows, ("scenario", "predictor", "layer"))

    assert len(aggregate) == 2
    assert {row["layer"] for row in aggregate} == {0, 1}
    assert all(row["count"] == 2 for row in aggregate)
    assert all(abs(float(row["normalized_mse"]) - 0.25) < 1.0e-6 for row in aggregate)
