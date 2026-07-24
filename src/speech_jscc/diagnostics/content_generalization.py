from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from channels.pilot import extract_data_resources, insert_data_and_pilots, make_pilot_mask
from evaluation.paired import run_mode_on_paired_batch
from models.resource_allocator import allocate_resources, deallocate_resources

from speech_jscc.diagnostics.random_distribution import SeedDeriver, file_hash, tree_hash
from speech_jscc.diagnostics.metrics import latent_metric_rows


CONTENT_ENGINE_VERSION = "stage1_content_generalization_v1"
CONTENT_STAGE_VERSION = "g0_g3_v1"
CONTENT_STAGES = (
    "g0_direct", "g1_pilot_reserved_identity", "g2_fixed_clean", "g3_random_clean"
)
SUBSET_SIZES = ("16", "64", "256", "full")


def parse_speaker_id(path: str | Path) -> str:
    text = Path(path).as_posix()
    match = re.search(r"/(\d+)/(\d+)/(\d+)-(\d+)-(\d+)\.[^/]+$", "/" + text.lstrip("/"))
    return match.group(1) if match else "unknown"


def _reject_test(path: Path) -> None:
    if "test" in {part.lower() for part in path.parts} or "test" in path.stem.lower():
        raise ValueError(f"test data is forbidden: {path}")


def _manifest_ids(path: Path) -> list[str]:
    output = []
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        item = json.loads(line) if path.suffix == ".jsonl" else {"audio_path": line.strip()}
        output.append(str(item["audio_path"]))
    return output


def _speaker_round_robin(candidates: dict[str, list[str]], seed: int) -> list[str]:
    rng = random.Random(seed)
    speakers = sorted(candidates); rng.shuffle(speakers)
    queues = {}
    for speaker in speakers:
        values = sorted(candidates[speaker]); rng.shuffle(values); queues[speaker] = values
    output = []
    while any(queues.values()):
        for speaker in speakers:
            if queues[speaker]: output.append(queues[speaker].pop())
    return output


def build_content_subsets(
    train_manifest: Path,
    validation_manifest: Path,
    cache_dir: Path,
    *,
    seed: int,
    validation_items_per_group: int = 8,
) -> dict[str, Any]:
    for path in (train_manifest, validation_manifest, cache_dir): _reject_test(path)
    train_ids, validation_ids = _manifest_ids(train_manifest), _manifest_ids(validation_manifest)
    grouped: dict[str, list[str]] = defaultdict(list)
    for utterance_id in train_ids: grouped[parse_speaker_id(utterance_id)].append(utterance_id)
    holdout = {}; candidates = {}
    for speaker, values in grouped.items():
        ordered = sorted(values); random.Random(seed + int(hashlib.sha256(speaker.encode()).hexdigest()[:8], 16)).shuffle(ordered)
        if len(ordered) >= 2:
            holdout[speaker], candidates[speaker] = ordered[0], ordered[1:]
        else:
            candidates[speaker] = ordered
    nested = _speaker_round_robin(candidates, seed + 71000)
    unseen_pool = sorted(validation_ids); random.Random(seed + 71001).shuffle(unseen_pool)
    base_train = nested[:min(16, len(nested))]
    base_speakers = sorted({parse_speaker_id(item) for item in base_train})
    fixed_seen = base_train[:validation_items_per_group]
    fixed_same = [holdout[speaker] for speaker in base_speakers if speaker in holdout][:validation_items_per_group]
    fixed_unseen = [item for item in unseen_pool if parse_speaker_id(item) not in set(base_speakers)][:validation_items_per_group]
    subsets = {}
    for label in SUBSET_SIZES:
        count = len(nested) if label == "full" else min(int(label), len(nested))
        selected = nested[:count]; speakers = sorted({parse_speaker_id(item) for item in selected})
        same = fixed_same
        seen = fixed_seen
        unseen = fixed_unseen
        subsets[label] = {
            "train_ids": selected,
            "seen_utterance_ids": seen,
            "same_speaker_unseen_ids": same,
            "unseen_speaker_ids": unseen,
            "train_speaker_ids": speakers,
            "unknown_speaker_count": sum(parse_speaker_id(item) == "unknown" for item in selected),
        }
    return {
        "diagnostic_engine_version": CONTENT_ENGINE_VERSION,
        "train_manifest": str(train_manifest), "validation_manifest": str(validation_manifest),
        "train_manifest_hash": file_hash(train_manifest), "validation_manifest_hash": file_hash(validation_manifest),
        "latent_cache": str(cache_dir), "latent_cache_hash": tree_hash(cache_dir), "subsets": subsets,
    }


def build_content_validation_suite(subset: dict[str, Any], seed: int) -> dict[str, Any]:
    derive = SeedDeriver(seed); scenarios = []
    groups = (
        ("seen_utterance_unseen_channel", subset["seen_utterance_ids"]),
        ("same_speaker_unseen_utterance_unseen_channel", subset["same_speaker_unseen_ids"]),
        ("unseen_speaker_unseen_utterance_unseen_channel", subset["unseen_speaker_ids"]),
    )
    for group, ids in groups:
        for index, utterance_id in enumerate(ids):
            scenarios.append({"group": group, "utterance_id": utterance_id,
                              "speaker_id": parse_speaker_id(utterance_id), "snr_db": 10.0,
                              "channel_seed": derive.seed(group + "_channel", index),
                              "noise_seed": derive.seed(group + "_noise", index)})
    encoded = json.dumps(scenarios, sort_keys=True, separators=(",", ":")).encode()
    return {"scenarios": scenarios, "validation_suite_hash": hashlib.sha256(encoded).hexdigest()}


def aggregate_dataset_statistics(examples: Iterable[dict[str, Any]], preprocessing: dict[str, Any]) -> dict[str, Any]:
    values = list(examples)
    if not values: raise ValueError("dataset statistics require at least one example")
    latents = torch.cat([item["latent"].detach().float().flatten().cpu() for item in values])
    durations = [float(item["duration_seconds"]) for item in values]
    speakers = [str(item.get("speaker_id", "unknown")) for item in values]
    encoded = json.dumps(preprocessing, sort_keys=True, separators=(",", ":")).encode()
    return {
        "utterance_count": len(values), "speaker_count": len(set(speakers) - {"unknown"}),
        "unknown_speaker_count": speakers.count("unknown"), "latent_power": float(latents.square().mean()),
        "latent_mean": float(latents.mean()), "latent_std": float(latents.std(unbiased=False)),
        "utterance_duration_mean_seconds": sum(durations) / len(durations),
        "utterance_duration_min_seconds": min(durations), "utterance_duration_max_seconds": max(durations),
        "preprocessing": preprocessing,
        "preprocessing_hash": hashlib.sha256(encoded).hexdigest(),
    }


def content_realization_seed(stage: str, root_seed: int, step: int) -> int:
    derive = SeedDeriver(root_seed)
    if stage == "g2_fixed_clean": return derive.seed("g2_fixed_channel_noise", 0)
    if stage == "g3_random_clean": return derive.seed("g3_random_channel_noise", step)
    return derive.seed(stage + "_identity", 0)


def summarize_content_metrics(reconstruction: torch.Tensor, target: torch.Tensor, *, group: str, epsilon: float = 1e-6) -> dict[str, Any]:
    rows = latent_metric_rows(reconstruction, target, epsilon=epsilon, predictor="model", scenario=group)
    numeric = ("target_power", "reconstruction_power", "power_ratio", "raw_mse", "normalized_mse",
               "zero_normalized_mse", "cosine_similarity", "pearson_correlation")
    per_layer = []
    for layer in range(target.shape[1]):
        members = [row for row in rows if row["layer"] == layer]
        item = {"layer": layer, **{key: sum(float(row[key]) for row in members) / len(members) for key in numeric}}
        item["relative_improvement_over_zero"] = (item["zero_normalized_mse"] - item["normalized_mse"]) / max(item["zero_normalized_mse"], epsilon)
        item["finite"] = all(row["finite"] for row in members); per_layer.append(item)
    aggregate = {key: sum(float(item[key]) for item in per_layer) / len(per_layer) for key in numeric}
    aggregate["relative_improvement_over_zero"] = (aggregate["zero_normalized_mse"] - aggregate["normalized_mse"]) / max(aggregate["zero_normalized_mse"], epsilon)
    aggregate["finite"] = all(item["finite"] for item in per_layer)
    return {"group": group, "aggregate": aggregate, "per_layer": per_layer, "layer0_summary": dict(per_layer[0])}


def forward_content_path(
    stage: str,
    codec,
    model,
    target: torch.Tensor,
    config: dict[str, Any],
    *,
    batch=None,
) -> dict[str, Any]:
    if stage not in CONTENT_STAGES: raise ValueError(f"unknown content stage: {stage}")
    state = torch.zeros(target.shape[0], model.encoder.channel_state_dim, device=target.device, dtype=target.dtype)
    gates = torch.ones(target.shape[0], model.encoder.num_layers, device=target.device, dtype=target.dtype)
    if stage in {"g2_fixed_clean", "g3_random_clean"}:
        if batch is None: raise ValueError(f"{stage} requires a paired channel batch")
        return run_mode_on_paired_batch(
            codec, model, batch, state, gates, equalizer="estimated", fading="multipath_block",
            channel_estimator="dft_tap_ls", estimator_num_taps=config["channel"].get("estimator_num_taps", 6),
            estimator_ridge_lambda=config["channel"].get("estimator_ridge_lambda", 1e-6),
            allocation_mode="uniform", resource_reliability=torch.ones_like(batch.noise.real),
            receiver_state_mode="observable_v1", decode_waveform=False,
        )
    symbols = model.encoder(target, state, layer_gates=gates)
    if stage == "g0_direct":
        decoder_input = symbols; mapping = None; grid = None
    else:
        allocation = allocate_resources(symbols, torch.ones_like(symbols.real), model.encoder.layer_channel_uses, mode="uniform")
        grid_shape = tuple(config["model"]["grid_shape"])
        pilot_mask = make_pilot_mask((target.shape[0], *grid_shape), config["channel"].get("pilot_spacing", 4),
                                     time_spacing=config["channel"].get("pilot_time_spacing", 4), device=target.device)
        grid, _ = insert_data_and_pilots(allocation.symbols, pilot_mask)
        recovered = extract_data_resources(grid, pilot_mask)
        decoder_input = deallocate_resources(recovered, allocation.resource_to_source)
        mapping = allocation.resource_to_source
    reconstruction = model.decoder(decoder_input, state)
    return {"reconstruction": reconstruction, "data_symbols": symbols, "decoder_input": decoder_input,
            "resource_to_source": mapping, "transmitted": grid,
            "resource_mapping_version": "pilot_reserved_v1" if stage != "g0_direct" else "direct_bypass"}


def content_group_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    aggregate = metrics["aggregate"]; reasons = []
    if aggregate["relative_improvement_over_zero"] < .05: reasons.append("zero_baseline_improvement_below_5_percent")
    if aggregate["power_ratio"] < .01: reasons.append("power_ratio_below_0.01")
    if aggregate["cosine_similarity"] <= 0: reasons.append("nonpositive_cosine")
    if aggregate["pearson_correlation"] <= 0: reasons.append("nonpositive_correlation")
    if not aggregate["finite"]: reasons.append("nonfinite_metrics")
    return {"passed": not reasons, "reasons": reasons}


def ladder_decision(stage: str, subset_size: str, passed: bool) -> str:
    if stage not in CONTENT_STAGES: raise ValueError(f"unknown content stage: {stage}")
    if subset_size not in SUBSET_SIZES: raise ValueError(f"unknown subset size: {subset_size}")
    if passed: return "next_stage"
    return "stop_first_failing_stage" if subset_size == "full" else "next_subset"
