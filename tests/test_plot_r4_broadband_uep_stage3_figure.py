"""Tests for the data reduction behind the Stage-3 Figure 3 curve."""

import csv
import importlib.util
from pathlib import Path


def _module():
    path = Path("scripts/plot_r4_broadband_uep_stage3_figure.py")
    spec = importlib.util.spec_from_file_location("plot_stage3", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_jsr_curve_uses_stage3_means_for_the_three_deployable_profiles(tmp_path):
    summary = tmp_path / "per_condition_summary.csv"
    rows = []
    values = {
        "U0": {-5.0: 0.0, 0.0: 0.0, 5.0: 0.0, 10.0: 0.0},
        "codebook": {-5.0: -0.2, 0.0: 0.1, 5.0: 3.0, 10.0: 7.0},
        "high": {-5.0: -0.3, 0.0: 0.2, 5.0: 2.9, 10.0: 8.0},
    }
    for profile, by_jsr in values.items():
        for snr in (5.0, 10.0):
            for jsr, delta in by_jsr.items():
                rows.append({
                    "profile": profile,
                    "snr_db": snr,
                    "target_jsr_db": jsr,
                    "rows": 2,
                    "mean_delta_si_sdr_vs_u0_db": delta,
                })
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    result = _module().build_jsr_series(
        summary,
        codebook_profile="codebook",
        high_jsr_profile="high",
        candidate_profiles=["codebook", "high"],
    )

    assert result["jsr_db"] == [-5.0, 0.0, 5.0, 10.0]
    assert result["u0"] == [0.0, 0.0, 0.0, 0.0]
    assert result["codebook_policy"] == [-0.2, 0.1, 3.0, 7.0]
    assert result["high_jsr_candidate"] == [-0.3, 0.2, 2.9, 8.0]
    assert "oracle_best_evaluated" not in result
