import subprocess

import pytest


@pytest.mark.parametrize("script",[
    "scripts/run_j2_boundary_sweep.sh",
    "scripts/run_j2_initialization_compare.sh",
    "scripts/run_j2_conv_conformer_external.sh",
])
def test_j2_scripts_are_safe_dry_runs(script):
    result=subprocess.run(["bash",script,"--dry-run"],text=True,capture_output=True)
    assert result.returncode==0, result.stderr
    assert "j2" in result.stdout.lower()
    assert "j3" not in result.stdout.lower()
