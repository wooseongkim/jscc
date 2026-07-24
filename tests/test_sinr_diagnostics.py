from __future__ import annotations

import math

import pytest
import torch

from eval_channel_estimator import (
    _aggregate,
    _make_realization,
    _row_for_estimator,
    data_resource_equalization_metrics,
    validate_aggregate_consistency,
    validate_pairing_and_consistency,
)


def _config() -> dict:
    return {
        "seed": 3,
        "device": "cpu",
        "channel": {
            "fading": "multipath_block",
            "num_taps": 2,
            "pdp_decay": 0.7,
            "grid_shape": [8, 4],
            "pilot_spacing": 2,
            "pilot_time_spacing": 2,
            "target_power": 1.0,
            "jammed_fraction": 0.25,
            "estimator_num_taps": 2,
            "estimator_ridge_lambda": 1e-6,
        },
    }


def test_data_only_sinr_excludes_pilots_and_matches_hand_calculation() -> None:
    transmitted = torch.ones(1, 4, 2, dtype=torch.complex64)
    pilot_mask = torch.zeros_like(transmitted, dtype=torch.bool)
    pilot_mask[:, 0, :] = True
    faded_signal = transmitted.clone()
    faded_signal[:, 0, :] = 100.0 + 0j
    faded_jammer = torch.full_like(transmitted, 0.5 + 0j)
    noise = torch.full_like(transmitted, 0.5j)
    channel = {
        "signal_fading": torch.ones_like(transmitted),
        "faded_signal": faded_signal,
        "faded_jammer": faded_jammer,
        "noise": noise,
        "received": faded_signal + faded_jammer + noise,
    }

    metrics = data_resource_equalization_metrics(
        channel=channel,
        transmitted=transmitted,
        pilot_mask=pilot_mask,
        channel_estimate=torch.ones_like(transmitted),
    )

    assert metrics["data_resource_count"] == 6
    assert metrics["pilot_resource_count"] == 2
    assert metrics["post_eq_sinr_linear"] == pytest.approx(2.0)
    assert metrics["post_eq_sinr_db"] == pytest.approx(10.0 * math.log10(2.0))


def test_data_evm_and_symbol_mse_match_hand_calculation() -> None:
    transmitted = torch.ones(1, 2, 2, dtype=torch.complex64)
    pilot_mask = torch.zeros_like(transmitted, dtype=torch.bool)
    pilot_mask[:, 0, 0] = True
    error = torch.zeros_like(transmitted)
    error[:, 0, 1] = 1.0 + 0j
    error[:, 1, 0] = 2.0 + 0j
    error[:, 1, 1] = 0.0 + 0j
    channel = {
        "signal_fading": torch.ones_like(transmitted),
        "faded_signal": transmitted,
        "faded_jammer": torch.zeros_like(transmitted),
        "noise": error,
        "received": transmitted + error,
    }

    metrics = data_resource_equalization_metrics(
        channel=channel,
        transmitted=transmitted,
        pilot_mask=pilot_mask,
        channel_estimate=torch.ones_like(transmitted),
    )

    assert metrics["equalized_symbol_mse"] == pytest.approx((1.0 + 4.0 + 0.0) / 3.0)
    assert metrics["data_evm"] == pytest.approx(math.sqrt(5.0 / 3.0))


def test_per_seed_rows_have_consistent_signed_differences_and_oracle_pairing() -> None:
    config = _config()
    realization = _make_realization(
        config,
        seed=10,
        snr_db=30.0,
        jsr_db=0.0,
        jammer_type="none",
        device=torch.device("cpu"),
    )
    rows = [
        _row_for_estimator(
            seed=10,
            snr_db=30.0,
            jsr_db=0.0,
            jammer_type="none",
            estimator=estimator,
            realization=realization,
            estimator_num_taps=2,
            ridge_lambda=1e-6,
        )
        for estimator in ["block_frequency_ls", "dft_tap_ls", "oracle"]
    ]

    validate_pairing_and_consistency(rows, ["block_frequency_ls", "dft_tap_ls", "oracle"])
    oracle_values = {row["oracle_post_eq_sinr_db"] for row in rows}
    assert len(oracle_values) == 1
    for row in rows:
        direct = row["estimated_post_eq_sinr_db"] - row["oracle_post_eq_sinr_db"]
        assert row["estimated_minus_oracle_sinr_db"] == pytest.approx(direct)
        assert row["oracle_minus_estimated_sinr_db"] == pytest.approx(-direct)
    oracle = next(row for row in rows if row["estimator"] == "oracle")
    assert oracle["estimated_minus_oracle_sinr_db"] == pytest.approx(0.0)


def test_pairing_validation_detects_regenerated_noise() -> None:
    config = _config()
    first = _make_realization(config, seed=10, snr_db=30.0, jsr_db=0.0, jammer_type="none", device=torch.device("cpu"))
    second = _make_realization(config, seed=11, snr_db=30.0, jsr_db=0.0, jammer_type="none", device=torch.device("cpu"))
    row_a = _row_for_estimator(seed=10, snr_db=30.0, jsr_db=0.0, jammer_type="none", estimator="dft_tap_ls", realization=first, estimator_num_taps=2, ridge_lambda=1e-6)
    row_b = _row_for_estimator(seed=10, snr_db=30.0, jsr_db=0.0, jammer_type="none", estimator="oracle", realization=second, estimator_num_taps=2, ridge_lambda=1e-6)
    row_b["scenario_id"] = row_a["scenario_id"]

    with pytest.raises(ValueError, match="noise_hash|signal_fading_hash"):
        validate_pairing_and_consistency([row_a, row_b], ["dft_tap_ls", "oracle"])


def test_aggregate_mean_difference_matches_difference_of_paired_db_means() -> None:
    rows = [
        {
            "estimator": "x",
            "requested_snr_db": 1.0,
            "jammer_type": "none",
            "finite_status": "finite",
            "estimated_post_eq_sinr_db": 10.0,
            "oracle_post_eq_sinr_db": 8.0,
            "estimated_minus_oracle_sinr_db": 2.0,
            "oracle_minus_estimated_sinr_db": -2.0,
            "estimated_post_eq_sinr_linear": 10.0,
            "oracle_post_eq_sinr_linear": 6.3095734448,
            "csi_nmse_linear": 0.0,
            "csi_nmse_db": -120.0,
            "pilot_evm": 0.0,
            "equalized_symbol_mse": 0.0,
            "data_evm": 0.0,
            "oracle_equalized_symbol_mse": 0.0,
            "oracle_data_evm": 0.0,
        },
        {
            "estimator": "x",
            "requested_snr_db": 1.0,
            "jammer_type": "none",
            "finite_status": "finite",
            "estimated_post_eq_sinr_db": 20.0,
            "oracle_post_eq_sinr_db": 21.0,
            "estimated_minus_oracle_sinr_db": -1.0,
            "oracle_minus_estimated_sinr_db": 1.0,
            "estimated_post_eq_sinr_linear": 100.0,
            "oracle_post_eq_sinr_linear": 125.89254118,
            "csi_nmse_linear": 0.0,
            "csi_nmse_db": -120.0,
            "pilot_evm": 0.0,
            "equalized_symbol_mse": 0.0,
            "data_evm": 0.0,
            "oracle_equalized_symbol_mse": 0.0,
            "oracle_data_evm": 0.0,
        },
    ]

    aggregate = _aggregate(rows)
    validate_aggregate_consistency(rows, aggregate)
    assert aggregate[0]["estimated_minus_oracle_sinr_db"]["mean"] == pytest.approx(0.5)
    assert aggregate[0]["estimated_post_eq_sinr_db"]["mean_per_seed_sinr_db"] == pytest.approx(15.0)
    assert aggregate[0]["estimated_post_eq_sinr_db"]["db_of_mean_linear_sinr"] != pytest.approx(15.0)
