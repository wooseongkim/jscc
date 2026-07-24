from __future__ import annotations

import csv
from pathlib import Path

import torch
import yaml

from eval_channel_estimator import run_estimator_comparison


def _config(path: Path) -> Path:
    payload = {
        "seed": 9,
        "device": "cpu",
        "channel": {
            "fading": "multipath_block",
            "num_taps": 3,
            "pdp": "exponential",
            "pdp_decay": 0.7,
            "block_fading_over_time": True,
            "assume_ideal_cp": True,
            "channel_estimator": "dft_tap_ls",
            "estimator_num_taps": 3,
            "estimator_ridge_lambda": 1e-6,
            "grid_shape": [12, 6],
            "pilot_spacing": 2,
            "pilot_time_spacing": 2,
            "target_power": 1.0,
            "jammed_fraction": 0.25,
        },
        "diagnostics": {"batch_size": 4},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_estimator_comparison_reuses_realization_and_writes_aggregates(tmp_path: Path) -> None:
    config = _config(tmp_path / "config.yaml")
    output = tmp_path / "out"

    report = run_estimator_comparison(
        config,
        snr_values=[30.0, 10.0],
        jsr_values=[0.0],
        jammer_types=["none"],
        num_seeds=3,
        estimators=["inverse_distance_2d", "block_frequency_ls", "dft_tap_ls", "oracle"],
        output_dir=output,
    )

    rows = list(csv.DictReader((output / "per_seed_results.csv").open()))
    assert (output / "aggregate_results.csv").exists()
    assert (output / "report.json").exists()
    assert (output / "report.md").exists()
    assert set(row["estimator"] for row in rows) == {
        "inverse_distance_2d",
        "block_frequency_ls",
        "dft_tap_ls",
        "oracle",
    }
    assert {float(row["requested_snr_db"]) for row in rows} == {30.0, 10.0}
    assert report["metadata"]["channel_estimator"]["name"] == "dft_tap_ls"
    assert report["aggregate"]
    assert all(row["status"] == "finite" for row in rows)


def test_dft_tap_ls_outperforms_block_interpolation_in_noiseless_constructed_case(tmp_path: Path) -> None:
    config = _config(tmp_path / "config.yaml")
    output = tmp_path / "out"
    report = run_estimator_comparison(
        config,
        snr_values=[120.0],
        jsr_values=[0.0],
        jammer_types=["none"],
        num_seeds=5,
        estimators=["block_frequency_ls", "dft_tap_ls", "oracle"],
        output_dir=output,
    )

    aggregate = {
        (item["estimator"], float(item["requested_snr_db"])): item
        for item in report["aggregate"]
    }
    assert aggregate[("dft_tap_ls", 120.0)]["csi_nmse_linear"]["median"] < 1e-6
    assert (
        aggregate[("block_frequency_ls", 120.0)]["csi_nmse_linear"]["median"]
        > aggregate[("dft_tap_ls", 120.0)]["csi_nmse_linear"]["median"]
    )
