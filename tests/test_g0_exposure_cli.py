import subprocess, sys


def test_dry_run_does_not_require_long_acknowledgement(tmp_path):
    result=subprocess.run([sys.executable,"diagnose_g0_exposure_normalized.py","--config","configs/train_stage1_fixed_tx_uniform.yaml","--subset-size","16","--max-epochs","64","--output-dir",str(tmp_path/"out"),"--dry-run"],capture_output=True,text=True,check=True)
    assert '"dry_run": true' in result.stdout


def test_actual_long_run_requires_acknowledgement(tmp_path):
    result=subprocess.run([sys.executable,"diagnose_g0_exposure_normalized.py","--config","configs/train_stage1_fixed_tx_uniform.yaml","--subset-size","16","--max-epochs","64","--output-dir",str(tmp_path/"out")],capture_output=True,text=True)
    assert result.returncode != 0 and "allow-long-run" in result.stderr
