import subprocess
from pathlib import Path

def test_j4_scripts_are_external_dry_run_and_do_not_start_j5():
    for script in ("scripts/run_j4_burst_boundary.sh","scripts/run_j4_conv_conformer_external.sh"):
        result=subprocess.run(["bash",script,"--dry-run","--device","cpu"],text=True,capture_output=True,check=True)
        assert "python" in result.stdout and "j5" not in result.stdout.lower()


def test_j4_tail_script_does_not_precreate_cli_output_directory():
    text=Path("scripts/run_j4_tail_diagnostic_external.sh").read_text()
    assert 'mkdir -p "$output"' not in text
    assert 'tee "$output/run.log"' not in text
    assert 'temporary_log="${output}.run.log"' in text
