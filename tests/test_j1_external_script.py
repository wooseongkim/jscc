import subprocess
def test_j1_script_dry_run_does_not_start_j2():
    result=subprocess.run(["bash","scripts/run_j1_conv_conformer_external.sh","--dry-run"],text=True,capture_output=True)
    assert result.returncode==0 and "j1_weak_random_barrage" in result.stdout and "j2" not in result.stdout.lower()
