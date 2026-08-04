"""Integration coverage for the paired R4 checkpoint waveform smoke CLI."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "evaluate_r4_si_sdr_finetune_checkpoints.py"
CONFIG = ROOT / "configs/eval_r4_si_sdr_finetune_checkpoints.yaml"
CANDIDATE = ROOT / "runs/waveform_aware_wireless/r4_si_sdr_finetune/control_no_si_sdr/local_step_000001.pt"


def _run(output: Path, candidate: Path = CANDIDATE) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, str(CLI), "--config", str(CONFIG),
            "--candidate-checkpoint", str(candidate), "--max-utterances", "1",
            "--max-realizations", "1", "--snr-db", "5", "--output-dir", str(output),
            "--device", "cpu", "--overwrite",
        ],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )


@pytest.mark.skipif(not CANDIDATE.is_file(), reason="one-step fine-tune checkpoint is not available")
def test_actual_r4_waveform_smoke_is_paired_and_finite(tmp_path: Path):
    result = _run(tmp_path / "smoke")
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in (tmp_path / "smoke" / "per_sample_selection_results.jsonl").read_text().splitlines()]
    summary = json.loads((tmp_path / "smoke" / "waveform_smoke_summary.json").read_text())
    assert len(rows) == 1
    row = rows[0]
    assert row["source_checkpoint_path"].endswith("best_5db_si_sdr.pt")
    assert row["candidate_checkpoint_path"].endswith("local_step_000001.pt")
    assert row["condition_hash"]
    assert row["source_condition_hash"] == row["candidate_condition_hash"] == row["condition_hash"]
    assert row["finite"] is True
    assert isinstance(row["source_si_sdr_db"], float)
    assert isinstance(row["candidate_si_sdr_db"], float)
    assert len(row["source_per_layer_nmse"]) == len(row["candidate_per_layer_nmse"]) == 8
    assert summary["paired_row_count"] == 1
    assert summary["condition_hash_equality"] is True
    assert summary["has_nan_or_inf"] is False


def test_missing_candidate_fails_fast(tmp_path: Path):
    result = _run(tmp_path / "missing", tmp_path / "does-not-exist.pt")
    assert result.returncode != 0
    assert "checkpoint does not exist" in result.stderr


def test_official_metric_has_no_shift_or_active_speech_masking():
    source = (ROOT / "src/speech_jscc/metrics/audio_quality.py").read_text()
    evaluator = CLI.read_text()
    assert "compute_si_sdr" in source
    assert "shift" not in source.lower()
    assert "active_speech" not in source.lower()
    assert "mock" not in evaluator.lower()
