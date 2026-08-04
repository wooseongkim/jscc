from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_discovery_deduplicates_model_state_and_keeps_aliases(tmp_path: Path):
    from speech_jscc.evaluation.r4_si_sdr_checkpoint_selection import discover_checkpoints
    import torch

    root = tmp_path / "runs"
    experiment = root / "control_no_si_sdr"
    experiment.mkdir(parents=True)
    payload = {"model": {"weight": torch.tensor([1.0])}, "local_step": 250}
    torch.save(payload, experiment / "local_step_000250.pt")
    torch.save(payload, experiment / "last.pt")
    (experiment / "broken.pt").write_text("not a checkpoint")
    records, representatives = discover_checkpoints(root, ["control_no_si_sdr"], {"control_no_si_sdr": 0.0})
    assert any(not item["loadable"] for item in records)
    assert len(representatives) == 1
    assert representatives[0]["local_step"] == 250
    assert representatives[0]["aliases"] == ["last.pt"]


def test_statistics_and_ranking_handle_smoke_and_no_passing_candidate():
    from speech_jscc.evaluation.r4_si_sdr_checkpoint_selection import (
        paired_statistics, rank_experiments, smoke_constraints,
    )
    stats = paired_statistics([0.1, -0.2], samples=20, seed=7)
    assert stats["count"] == 2
    assert stats["insufficient_for_inference"] is True
    assert paired_statistics([0.1, -0.2], samples=20, seed=7) == stats
    constraints = smoke_constraints(rows=[{"finite": True, "condition_hash": "a", "mapping_hash": "m"}])
    ranking = rank_experiments([{"experiment": "si_sdr_low", "constraint_pass": False, "metrics": {}}], tolerance=1e-6)
    assert constraints["pass"] is True
    assert ranking["si_sdr_low"]["status"] == "no_passing_candidate"


def test_selection_artifact_is_expanded_only(tmp_path: Path):
    from speech_jscc.evaluation.r4_si_sdr_checkpoint_selection import write_selection_artifact
    path = tmp_path / "experiment_best_checkpoints.json"
    write_selection_artifact(path, source={"path": "source.pt", "sha256": "x", "model_state_hash": "y", "global_step": 5750}, selected={})
    value = json.loads(path.read_text())
    assert value["selection_split"] == "expanded_selection"
    assert value["selection_uses_legacy_metrics"] is False
    assert "legacy" not in json.dumps(value).lower().replace("selection_uses_legacy_metrics", "")
