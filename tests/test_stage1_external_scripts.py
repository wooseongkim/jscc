from __future__ import annotations

from pathlib import Path


SCRIPTS = (
    "regenerate_o5_reports.sh", "run_o6_random_clean_external.sh", "run_o6_extension_external.sh",
    "run_stage1_jammer_curriculum_external.sh", "evaluate_stage1_curriculum_stage.sh",
    "prepare_uniform_stage1_full_external.sh",
)


def test_external_scripts_are_strict_and_support_dry_run() -> None:
    for name in SCRIPTS:
        text = (Path("scripts") / name).read_text()
        assert "set -euo pipefail" in text
        assert "--dry-run" in text or name == "regenerate_o5_reports.sh"


def test_long_scripts_preserve_logs_and_do_not_default_to_overwrite() -> None:
    for name in ("run_o6_random_clean_external.sh", "run_stage1_jammer_curriculum_external.sh"):
        text = (Path("scripts") / name).read_text()
        assert "tee" in text
        assert "--overwrite" in text
        assert "refusing" in text


def test_fresh_run_does_not_create_cli_output_directory_before_cli_starts() -> None:
    for name, variable in (("run_o6_random_clean_external.sh", "ROOT"),
                           ("run_stage1_jammer_curriculum_external.sh", "out")):
        text = (Path("scripts") / name).read_text()
        assert f'mkdir -p "${variable}"' not in text
        assert "PENDING_LOG" in text


def test_curriculum_script_checks_parent_gate_before_starting_next_stage() -> None:
    text = Path("scripts/run_stage1_jammer_curriculum_external.sh").read_text()
    assert 'parent_summary="$ROOT/$parent/summary.json"' in text
    assert "parent stage gate failed" in text
