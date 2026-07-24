from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any


ENGINE_VERSION = "stage1_random_distribution_v1"
STAGE_DEFINITION_VERSION = "stage1_curriculum_v1"
SEED_DERIVATION_VERSION = "sha256_domain_v1"

STAGE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "o6_random_clean": {
        "clean_probability": 1.0, "jammer_probabilities": {},
        "snr_db_range": [5.0, 15.0], "jsr_db_range": None, "jammed_fraction": 0.0,
    },
    "j1_weak_barrage": {
        "clean_probability": .20, "jammer_probabilities": {"barrage": 1.0},
        "snr_db_range": [5.0, 15.0], "jsr_db_range": [-15.0, -10.0], "jammed_fraction": 1.0,
    },
    "j2_moderate_barrage": {
        "clean_probability": .20, "jammer_probabilities": {"barrage": 1.0},
        "snr_db_range": [5.0, 15.0], "jsr_db_range": [-10.0, -5.0], "jammed_fraction": 1.0,
    },
    "j3_strong_barrage": {
        "clean_probability": .20, "jammer_probabilities": {"barrage": 1.0},
        "snr_db_range": [5.0, 15.0], "jsr_db_range": [-5.0, 0.0], "jammed_fraction": 1.0,
    },
    "j4_mixed_sparse": {
        "clean_probability": .20,
        "jammer_probabilities": {"barrage": .50, "narrowband": .25, "burst": .25},
        "snr_db_range": [5.0, 15.0], "jsr_db_range": [-10.0, 0.0], "jammed_fraction": .25,
    },
    "j5_full_mixture": {
        "clean_probability": .20,
        "jammer_probabilities": {"barrage": .40, "narrowband": .25, "burst": .25, "pilot": .10},
        "snr_db_range": [-2.0, 15.0], "jsr_db_range": [-10.0, 8.0], "jammed_fraction": .25,
    },
}

PARENT_STAGE = {
    "j1_weak_barrage": "o6_random_clean", "j2_moderate_barrage": "j1_weak_barrage",
    "j3_strong_barrage": "j2_moderate_barrage", "j4_mixed_sparse": "j3_strong_barrage",
    "j5_full_mixture": "j4_mixed_sparse",
}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    if path.exists():
        for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            digest.update(str(item.relative_to(path)).encode())
            digest.update(str(item.stat().st_size).encode())
            digest.update(str(item.stat().st_mtime_ns).encode())
    return digest.hexdigest()


def _manifest_ids(path: Path) -> list[str]:
    values = []
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        item = json.loads(line) if path.suffix == ".jsonl" else {"audio_path": line.strip()}
        values.append(str(item["audio_path"]))
    return values


def _reject_test_source(path: Path) -> None:
    if "test" in {part.lower() for part in path.parts} or "test" in path.stem.lower():
        raise ValueError(f"test data is forbidden: {path}")


def build_subset_manifest(
    train_manifest: Path,
    validation_manifest: Path,
    cache_dir: Path,
    *,
    train_count: int = 16,
    validation_count: int = 8,
    seed: int = 23,
) -> dict[str, Any]:
    for path in (train_manifest, validation_manifest, cache_dir):
        _reject_test_source(path)
    train_ids = _manifest_ids(train_manifest)
    validation_ids = _manifest_ids(validation_manifest)
    if len(train_ids) < train_count or len(validation_ids) < validation_count:
        raise ValueError("manifests are too small for diagnostic subsets")
    train_rng = random.Random(seed + 61001); validation_rng = random.Random(seed + 61002)
    selected_train = sorted(train_rng.sample(train_ids, train_count))
    selected_validation = sorted(validation_rng.sample(validation_ids, validation_count))
    if set(selected_train) & set(selected_validation):
        raise ValueError("train and validation utterance IDs overlap")
    return {
        "diagnostic_engine_version": ENGINE_VERSION,
        "train_utterance_ids": selected_train,
        "validation_utterance_ids": selected_validation,
        "train_manifest": str(train_manifest), "validation_manifest": str(validation_manifest),
        "train_manifest_hash": file_hash(train_manifest), "validation_manifest_hash": file_hash(validation_manifest),
        "latent_cache": str(cache_dir), "latent_cache_hash": tree_hash(cache_dir),
    }


class SeedDeriver:
    def __init__(self, root_seed: int): self.root_seed = int(root_seed)

    def seed(self, domain: str, index: int, extra: str = "") -> int:
        payload = f"{SEED_DERIVATION_VERSION}|{self.root_seed}|{domain}|{index}|{extra}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def build_validation_suite(seed: int, train_ids: list[str], validation_ids: list[str]) -> dict[str, Any]:
    derive = SeedDeriver(seed)
    scenarios = []
    for suite, ids in (("V1", train_ids), ("V2", validation_ids)):
        for index, utterance_id in enumerate(ids):
            scenarios.append({"suite": suite, "utterance_id": utterance_id, "snr_db": 10.0,
                              "channel_seed": derive.seed(f"{suite}_channel", index),
                              "noise_seed": derive.seed(f"{suite}_noise", index)})
    for snr in (5.0, 10.0, 15.0):
        for index, utterance_id in enumerate(validation_ids):
            scenarios.append({"suite": "V3", "utterance_id": utterance_id, "snr_db": snr,
                              "channel_seed": derive.seed("V3_channel", index, str(snr)),
                              "noise_seed": derive.seed("V3_noise", index, str(snr))})
    encoded = json.dumps(scenarios, sort_keys=True, separators=(",", ":")).encode()
    return {"scenarios": scenarios, "validation_suite_hash": hashlib.sha256(encoded).hexdigest()}


def build_stage_validation_suite(stage: str, seed: int, train_ids: list[str], validation_ids: list[str]) -> dict[str, Any]:
    suite = build_validation_suite(seed, train_ids, validation_ids)
    if stage == "o6_random_clean":
        return suite
    derive = SeedDeriver(seed)
    types = ["barrage"]
    if stage in {"j4_mixed_sparse", "j5_full_mixture"}: types += ["narrowband", "burst"]
    if stage == "j5_full_mixture": types += ["pilot"]
    scenarios = list(suite["scenarios"])
    for jammer_type in types:
        for jsr in (-10.0, -5.0, 0.0):
            for index, utterance_id in enumerate(validation_ids):
                label = f"{jammer_type}_jsr{int(jsr)}"
                scenarios.append({"suite": label, "utterance_id": utterance_id, "snr_db": 10.0,
                                  "jsr_db": jsr, "jammer_type": jammer_type,
                                  "channel_seed": derive.seed(f"{label}_channel", index),
                                  "noise_seed": derive.seed(f"{label}_noise", index),
                                  "jammer_seed": derive.seed(f"{label}_jammer", index)})
    encoded = json.dumps(scenarios, sort_keys=True, separators=(",", ":")).encode()
    return {"scenarios": scenarios, "validation_suite_hash": hashlib.sha256(encoded).hexdigest()}


def validate_external_steps(steps: int, *, allow_long_run: bool) -> None:
    if steps < 0: raise ValueError("steps must be nonnegative")
    if steps > 5 and not allow_long_run:
        raise ValueError("steps > 5 require --allow-long-run external acknowledgement")


def validate_curriculum_parent(stage: str, initialization_mode: str, parent: dict[str, Any] | None) -> None:
    if initialization_mode not in {"curriculum_resume", "fresh_initialization_control"}:
        raise ValueError("invalid initialization mode")
    if initialization_mode == "fresh_initialization_control":
        if parent is not None:
            raise ValueError("initialization mode cannot mix fresh control with curriculum parent")
        return
    expected = PARENT_STAGE.get(stage)
    if expected is None:
        if parent is not None: raise ValueError("O6 curriculum root cannot have a parent")
        return
    if parent is None or parent.get("stage_name") != expected:
        raise ValueError(f"stage {stage} requires parent stage {expected}")
    if parent.get("initialization_mode") != "curriculum_resume":
        raise ValueError("initialization mode mismatch in curriculum lineage")


def catastrophic_forgetting(before: float, after: float) -> dict[str, float]:
    return {
        "previous_stage_validation_loss_before": float(before),
        "previous_stage_validation_loss_after": float(after),
        "relative_degradation": (float(after) - float(before)) / max(float(before), 1e-12),
    }


def stage_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    if float(metrics.get("relative_improvement_over_zero", -1)) < .05: reasons.append("zero_baseline_improvement_below_5_percent")
    if float(metrics.get("power_ratio", 0)) < .01: reasons.append("power_ratio_below_0.01")
    if float(metrics.get("cosine_similarity", 0)) <= 0: reasons.append("nonpositive_cosine")
    if float(metrics.get("pearson_correlation", 0)) <= 0: reasons.append("nonpositive_correlation")
    if not bool(metrics.get("finite", False)): reasons.append("nonfinite_metrics")
    if metrics.get("channel_hash_diversity", 0) < 2: reasons.append("channel_realizations_did_not_change")
    if metrics.get("noise_hash_diversity", 0) < 2: reasons.append("noise_realizations_did_not_change")
    if float(metrics.get("previous_stage_relative_degradation", 0)) > .5: reasons.append("severe_catastrophic_forgetting")
    return {"passed": not reasons, "reasons": reasons}


def sample_stage_distribution(stage: str, seed: int) -> dict[str, Any]:
    definition = STAGE_DEFINITIONS[stage]
    rng = random.Random(seed)
    snr = rng.uniform(*definition["snr_db_range"])
    clean = rng.random() < definition["clean_probability"]
    if clean or not definition["jammer_probabilities"]:
        return {"snr_db": snr, "jsr_db": 0.0, "jammer_type": "none", "clean": True}
    draw = rng.random(); cumulative = 0.0; jammer = None
    for name, probability in definition["jammer_probabilities"].items():
        cumulative += probability
        if draw <= cumulative:
            jammer = name; break
    jammer = jammer or next(reversed(definition["jammer_probabilities"]))
    return {"snr_db": snr, "jsr_db": rng.uniform(*definition["jsr_db_range"]),
            "jammer_type": jammer, "clean": False}
