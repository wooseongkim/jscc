"""Canonical CPU source cross-check: stratified uniform and selector agree."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/waveform_aware_wireless/r4_repetition3_mrc_finetune/best_5db_si_sdr.pt"


@pytest.mark.skipif(not SOURCE.is_file(), reason="selected R4 checkpoint unavailable")
def test_cpu_source_cross_evaluator_smoke_matches(tmp_path: Path):
    stratified = tmp_path / "stratified"
    selector = tmp_path / "selector"
    common = dict(cwd=ROOT, text=True, capture_output=True)
    first = subprocess.run([sys.executable, "evaluate_r4_stratified_allocation.py", "--config", "configs/eval_r4_stratified_allocation.yaml", "--dataset-role", "expanded_selection", "--snr-db", "5", "--profiles", "uniform", "core_protection", "layer1_focused", "--max-utterances", "1", "--max-realizations", "1", "--output-dir", str(stratified), "--device", "cpu", "--overwrite"], **common)
    assert first.returncode == 0, first.stdout + first.stderr
    second = subprocess.run([sys.executable, "evaluate_r4_si_sdr_finetune_checkpoints.py", "--config", "configs/eval_r4_si_sdr_finetune_checkpoints.yaml", "--candidate-checkpoint", str(SOURCE), "--max-utterances", "1", "--max-realizations", "1", "--snr-db", "5", "--output-dir", str(selector), "--device", "cpu", "--overwrite"], **common)
    assert second.returncode == 0, second.stdout + second.stderr
    uniform = next(json.loads(line) for line in (stratified / "per_sample_profile_comparison.jsonl").read_text().splitlines() if json.loads(line)["profile"] == "uniform")
    source = json.loads((selector / "per_sample_selection_results.jsonl").read_text().splitlines()[0])
    assert uniform["utterance_id"] == source["utterance_id"]
    assert uniform["mapping_hash"] == source["mapping_hash"]
    assert abs(uniform["si_sdr_db"] - source["source_si_sdr_db"]) <= 1e-3
