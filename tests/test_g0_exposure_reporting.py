from pathlib import Path


def test_external_script_has_required_guards_and_arguments():
    text=Path("scripts/run_g0_exposure_normalized_external.sh").read_text()
    assert "set -euo pipefail" in text
    for flag in ("--batch-size","--max-epochs","--device","--resume","--overwrite","--dry-run"): assert flag in text
    assert "PENDING_LOG" in text and "SUBSETS=(16 64 256 full)" in text


def test_reporter_names_required_artifacts():
    text=Path("scripts/summarize_g0_exposure.py").read_text()
    for name in ("exposure_normalized_report.md","aggregate_by_epoch.csv","aggregate_by_subset.csv","per_layer_by_epoch.csv","exposure_manifest.json"): assert name in text
