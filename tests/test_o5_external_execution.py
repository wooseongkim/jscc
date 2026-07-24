from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_external_scripts_use_strict_mode_and_do_not_default_to_overwrite() -> None:
    for name in ("run_o5_root_cause_external.sh", "run_o5_extension_external.sh", "run_o5_all_external.sh"):
        path = Path("scripts") / name
        text = path.read_text()
        assert "set -euo pipefail" in text
        assert "--allow_long_run" in text
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_primary_script_contains_all_conditions_and_separate_logs() -> None:
    text = Path("scripts/run_o5_root_cause_external.sh").read_text()
    for condition in ("clean_awgn_reference", "full_barrage_estimated_csi", "full_barrage_oracle_csi",
                      "data_only_barrage_estimated_csi", "data_only_barrage_oracle_csi",
                      "pilot_only_jammer_estimated_csi", "full_barrage_oracle_subtraction"):
        assert condition in text
    assert "tee" in text and "run.log" in text and "--overwrite" in text


def test_dry_run_prints_without_starting_optimization(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "diagnose_o5_root_cause.py", "--config", "configs/train_stage1_fixed_tx_uniform.yaml",
         "--condition", "full_barrage_estimated_csi", "--steps", "500", "--seed", "23",
         "--output_dir", str(tmp_path / "dry"), "--dry_run"],
        check=True, capture_output=True, text=True,
    )
    assert '"dry_run": true' in result.stdout.lower()
    assert not (tmp_path / "dry").exists()


def test_long_run_requires_explicit_acknowledgement(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "diagnose_o5_root_cause.py", "--config", "configs/train_stage1_fixed_tx_uniform.yaml",
         "--condition", "full_barrage_estimated_csi", "--steps", "6", "--seed", "23",
         "--output_dir", str(tmp_path / "blocked")], capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "allow_long_run" in result.stderr


def test_all_in_one_batch_runs_primary_then_optional_extensions() -> None:
    text = Path("scripts/run_o5_all_external.sh").read_text()
    assert "run_o5_root_cause_external.sh" in text
    assert "run_o5_extension_external.sh --execute" in text
    assert "--with-extensions" in text
    assert "--overwrite" in text
