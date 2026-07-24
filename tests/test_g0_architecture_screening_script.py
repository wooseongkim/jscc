from pathlib import Path
import subprocess

def test_external_script_dry_run_selected_architecture(tmp_path):
    result=subprocess.run(["bash","scripts/run_g0_architecture_screening_external.sh","--architecture","conv_conformer_v1","--subset-size","16","--max-epochs","16","--output-root",str(tmp_path),"--dry-run"],text=True,capture_output=True)
    assert result.returncode==0 and "conv_conformer_v1" in result.stdout and "--allow-long-run" in result.stdout
    assert not any(tmp_path.iterdir())
