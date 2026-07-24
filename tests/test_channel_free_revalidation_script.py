from pathlib import Path


def test_external_script_is_safe_and_exposes_controlled_matrix():
    text = Path("scripts/run_channel_free_revalidation_external.sh").read_text()
    assert "set -euo pipefail" in text
    for token in ("--experiment", "--device", "--resume", "--overwrite", "--dry-run"):
        assert token in text
    for experiment in ("baseline", "cf1", "cf2", "cf3", "cf4", "cf5", "eval"):
        assert experiment in text
    assert "J5" not in text and "jammer" not in text
