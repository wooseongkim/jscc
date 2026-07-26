from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_training_script_has_external_safety_contract():
    text = (ROOT / "scripts/run_r4_waveform_finetune_external.sh").read_text()
    assert "set -euo pipefail" in text
    assert 'cd "$repo_root"' in text
    assert "tee" in text
    assert "--dry-run" in text
    assert "--resume" in text
    assert "--overwrite" in text
    assert "--allow-long-run" in text


def test_evaluation_script_uses_fixed_full_protocol():
    text = (ROOT / "scripts/run_r4_waveform_finetune_eval_external.sh").read_text()
    assert "set -euo pipefail" in text
    assert "--utterances 64" in text
    assert "--realizations 2" in text
    assert "configs/ofdm_nr_like_r4_repetition3_mrc.yaml" in text
