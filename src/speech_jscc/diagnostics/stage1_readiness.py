from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from speech_jscc.diagnostics.random_distribution import ENGINE_VERSION, PARENT_STAGE


REQUIRED_STAGES = ("o6_random_clean", "j1_weak_barrage", "j2_moderate_barrage", "j3_strong_barrage", "j4_mixed_sparse", "j5_full_mixture")


def classify_distribution_evidence(
    o6_summary: dict[str, Any] | None,
    j1_summary: dict[str, Any] | None,
    *,
    g3_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    o6_passed = bool((o6_summary or {}).get("gate", {}).get("passed", False))
    g3_passed = bool((g3_summary or {}).get("gate", {}).get("passed", False))
    j1_status = "not_run"
    if j1_summary is not None:
        j1_status = "PASS" if o6_passed and j1_summary.get("gate", {}).get("passed", False) else "exploratory_failed_parent"
    return {
        "o6_random_clean": "PASS" if o6_passed else "FAIL",
        "j1_weak_barrage": j1_status,
        "g3_random_clean": "PASS" if g3_passed else ("FAIL" if g3_summary is not None else "not_run"),
        "curriculum_resume_allowed": o6_passed and g3_passed and j1_status != "exploratory_failed_parent",
    }


def validate_j_stage_parent(stage: str, parent: dict[str, Any]) -> None:
    if parent.get("evidence_status") == "exploratory_failed_parent":
        raise ValueError("exploratory_failed_parent checkpoint cannot parent a curriculum stage")
    expected = PARENT_STAGE.get(stage)
    if parent.get("stage_name") != expected:
        raise ValueError(f"{stage} requires parent {expected}")


def evaluate_readiness(
    root: Path,
    *,
    expected: dict[str, Any],
    fixed_path_passed: bool = False,
    tests_passed: bool = False,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not fixed_path_passed: reasons.append("O1-O5 fixed path and C1 extension evidence not confirmed")
    if not tests_passed: reasons.append("relevant test-suite pass evidence not recorded")
    previous = None
    summaries = {}
    for stage in REQUIRED_STAGES:
        path = root / stage / "summary.json"
        if not path.exists():
            reasons.append(f"missing external result: {stage}"); continue
        summary = json.loads(path.read_text()); summaries[stage] = summary
        provenance = summary.get("provenance", {})
        if provenance.get("diagnostic_engine_version") != ENGINE_VERSION: reasons.append(f"{stage}: diagnostic_engine_version mismatch")
        if provenance.get("stage_name") != stage: reasons.append(f"{stage}: stage_name mismatch")
        if not summary.get("gate", {}).get("passed", False): reasons.append(f"{stage}: gate failed")
        for key, value in expected.items():
            if key == "validation_suite_hash" and stage != "o6_random_clean": continue
            if provenance.get(key) != value: reasons.append(f"{stage}: {key} mismatch")
        if not provenance.get("validation_suite_hash"): reasons.append(f"{stage}: validation_suite_hash missing")
        parent_path = provenance.get("parent_checkpoint")
        parent_hash = provenance.get("parent_checkpoint_hash")
        if parent_path:
            candidate = Path(parent_path)
            if not candidate.exists(): reasons.append(f"{stage}: parent checkpoint missing")
            elif __import__("hashlib").sha256(candidate.read_bytes()).hexdigest() != parent_hash:
                reasons.append(f"{stage}: parent checkpoint hash mismatch")
        expected_parent = PARENT_STAGE.get(stage)
        if expected_parent != provenance.get("parent_stage"):
            reasons.append(f"{stage}: parent lineage mismatch")
        if stage != "o6_random_clean" and provenance.get("initialization_mode") != "curriculum_resume":
            reasons.append(f"{stage}: initialization mode is not curriculum_resume")
        history = provenance.get("curriculum_history", [])
        if previous and (not history or history[-2:] != [previous, stage]): reasons.append(f"{stage}: curriculum history mismatch")
        previous = stage
    ready = not reasons
    command = None
    if ready:
        command = ("python train_stage1_fixed_tx.py --config configs/train_stage1_fixed_tx_uniform.yaml "
                   "--steps 20000 --output_dir runs/stage1_uniform_pilot_reserved_v1_scientific")
    return {"ready": ready, "reasons": reasons, "stages": summaries, "uniform_training_command": command}
