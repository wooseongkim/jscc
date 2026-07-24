from pathlib import Path
import subprocess
def test_all_integration_scripts_support_dry_run():
    for name in ("run_g1_mapping_equivalence.sh","run_g1_conv_conformer_external.sh","run_g2_conv_conformer_external.sh","run_g3_conv_conformer_external.sh","run_g1_g3_sequence_external.sh"):
        result=subprocess.run(["bash",str(Path("scripts")/name),"--dry-run"],text=True,capture_output=True)
        assert result.returncode==0,(name,result.stderr)
