from pathlib import Path
import subprocess
import sys


def test_content_external_script_is_guarded_and_uses_pending_logs() -> None:
    text = Path("scripts/run_stage1_content_generalization_external.sh").read_text()
    assert "set -euo pipefail" in text
    assert "--dry-run" in text and "--overwrite" in text and "--resume" in text
    assert "PENDING_LOG" in text
    assert "subset_16" in text or 'SUBSETS=(16 64 256 full)' in text
    assert "stop_first_failing_stage" in text


def test_content_evaluation_script_never_starts_training() -> None:
    text = Path("scripts/evaluate_stage1_content_generalization.sh").read_text()
    assert "set -euo pipefail" in text
    assert "summarize_content_generalization.py" in text
    assert "diagnose_stage1_content_generalization.py" not in text


def test_summarizer_creates_an_absent_result_root(tmp_path: Path) -> None:
    root = tmp_path / "new-root"
    subprocess.run([sys.executable, "scripts/summarize_content_generalization.py", "--root", str(root)], check=True)
    assert (root / "content_generalization_report.md").exists()
    assert (root / "experiment_manifest.json").exists()
