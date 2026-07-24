from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from channels.pilot import extract_data_resources
from speech_jscc.checkpoint import normalization_config
from speech_jscc.config import load_config, resolve_device
from speech_jscc.diagnostics.metrics import latent_metric_rows
from speech_jscc.diagnostics.o5_root_cause import stable_tensor_hash
from speech_jscc.diagnostics.random_distribution import (
    ENGINE_VERSION, SEED_DERIVATION_VERSION, STAGE_DEFINITION_VERSION, STAGE_DEFINITIONS,
    SeedDeriver, build_subset_manifest, build_stage_validation_suite, sample_stage_distribution,
    catastrophic_forgetting, stage_gate, validate_curriculum_parent, validate_external_steps,
)
from speech_jscc.diagnostics.o5_root_cause import linear_slope
from speech_jscc.experiment import build_components
from speech_jscc.training.stage1 import build_stage1_optimizer, stage1_fixed_tx_step
from train_latent_jscc import RepresentationSource
from train_stage1_fixed_tx import _make_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="External Stage-1 random-distribution diagnostic")
    parser.add_argument("--config", required=True); parser.add_argument("--stage", required=True, choices=STAGE_DEFINITIONS)
    parser.add_argument("--steps", required=True, type=int); parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--device")
    parser.add_argument("--checkpoint-every", type=int, default=250); parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=25); parser.add_argument("--allow-long-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume"); parser.add_argument("--parent-checkpoint")
    parser.add_argument("--initialization-mode", choices=("curriculum_resume", "fresh_initialization_control"), default="fresh_initialization_control")
    return parser.parse_args()


def _git() -> tuple[str, bool]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], text=True, capture_output=True, check=False).stdout.strip())
    return commit, dirty


def _find_index(source: RepresentationSource, utterance_id: str) -> int:
    if source.dataset is None: raise ValueError("random-distribution diagnostics require manifest-backed real latents")
    matches = [i for i, path in enumerate(source.dataset.paths) if path.as_posix().endswith(utterance_id) or path.name == Path(utterance_id).name]
    if len(matches) != 1: raise ValueError(f"utterance ID does not uniquely resolve: {utterance_id}")
    return matches[0]


def _example(source: RepresentationSource, utterance_id: str) -> tuple[torch.Tensor, torch.Tensor]:
    latent, waveform = source.dataset[_find_index(source, utterance_id)]
    return latent.unsqueeze(0), waveform.unsqueeze(0)


def _latent_summary(reconstruction: torch.Tensor, target: torch.Tensor, loss: float) -> dict[str, Any]:
    rows = latent_metric_rows(reconstruction, target, epsilon=1e-6, predictor="model", scenario="distribution")
    def mean(key: str) -> float: return sum(float(row[key]) for row in rows) / len(rows)
    zero = mean("zero_normalized_mse")
    return {
        "loss": loss, "zero_predictor_loss": zero,
        "relative_improvement_over_zero": (zero - loss) / max(zero, 1e-12),
        "power_ratio": mean("power_ratio"), "cosine_similarity": mean("cosine_similarity"),
        "pearson_correlation": mean("pearson_correlation"), "finite": all(row["finite"] for row in rows),
        "per_layer_normalized_mse": [next(row["normalized_mse"] for row in rows if row["layer"] == layer) for layer in range(target.shape[1])],
        "per_layer_power_ratio": [next(row["power_ratio"] for row in rows if row["layer"] == layer) for layer in range(target.shape[1])],
    }


def _parameter_grad_norm(module: torch.nn.Module) -> float:
    return float(torch.sqrt(sum((p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None), torch.tensor(0.0))))


def _data_evm(result: dict[str, Any], pilot_mask: torch.Tensor) -> float:
    transmitted = extract_data_resources(result["transmitted"], pilot_mask)
    equalized = extract_data_resources(result["equalized"], pilot_mask)
    return float((equalized - transmitted).abs().square().mean() / transmitted.abs().square().mean().clamp_min(1e-12))


def _validation(model, codec, config, sources, subset, suite, device, layer_weights) -> dict[str, Any]:
    model.eval(); grouped: dict[str, list[dict[str, Any]]] = {}
    with torch.no_grad():
        for index, scenario in enumerate(suite["scenarios"]):
            source = sources["train"] if scenario["suite"] == "V1" else sources["val"]
            target, waveform = _example(source, scenario["utterance_id"])
            batch = _make_batch(codec, model, config, target=target, waveform=waveform,
                                snr_db=scenario["snr_db"], jsr_db=scenario.get("jsr_db", 0.0), jammer_type=scenario.get("jammer_type", "none"),
                                seed=scenario["channel_seed"], device=device)
            result = stage1_fixed_tx_step(codec, model, batch, None, layer_weights,
                latent_normalization=normalization_config(config), channel_estimator="dft_tap_ls",
                estimator_num_taps=6, estimator_ridge_lambda=config["channel"].get("estimator_ridge_lambda", 1e-6))
            metrics = _latent_summary(result["reconstruction"], target, float(result["loss"]))
            metrics.update({"csi_nmse": float(result["csi_nmse"].mean()), "pilot_evm": float(result["pilot_evm"].mean()),
                            "data_evm": _data_evm(result, batch.pilot_mask),
                            "post_equalization_sinr_db": float(10 * torch.log10(result["post_equalization_sinr"].clamp_min(1e-12)).mean())})
            key = scenario["suite"] if scenario["suite"] != "V3" else f"V3_{int(scenario['snr_db'])}dB"
            grouped.setdefault(key, []).append(metrics)
    model.train()
    aggregate = {}
    for key, rows in grouped.items():
        aggregate[key] = {name: sum(float(row[name]) for row in rows) / len(rows) for name in
                          ("loss", "zero_predictor_loss", "relative_improvement_over_zero", "power_ratio", "cosine_similarity", "pearson_correlation", "csi_nmse", "pilot_evm", "data_evm", "post_equalization_sinr_db")}
        aggregate[key]["finite"] = all(row["finite"] for row in rows)
        aggregate[key]["per_layer_normalized_mse"] = [sum(float(row["per_layer_normalized_mse"][layer]) for row in rows) / len(rows) for layer in range(8)]
        aggregate[key]["per_layer_power_ratio"] = [sum(float(row["per_layer_power_ratio"][layer]) for row in rows) / len(rows) for layer in range(8)]
    return aggregate


def _provenance(args, config, subset, suite, parent, cumulative_steps: int) -> dict[str, Any]:
    commit, dirty = _git()
    parent_path = Path(args.parent_checkpoint).resolve() if args.parent_checkpoint else None
    parent_hash = hashlib.sha256(parent_path.read_bytes()).hexdigest() if parent_path else None
    return {
        "diagnostic_engine_version": ENGINE_VERSION, "stage_name": args.stage,
        "stage_definition_version": STAGE_DEFINITION_VERSION,
        "resolved_stage_distribution": STAGE_DEFINITIONS[args.stage],
        "seed_derivation_version": SEED_DERIVATION_VERSION,
        "train_utterance_ids": subset["train_utterance_ids"], "validation_utterance_ids": subset["validation_utterance_ids"],
        "manifest_hashes": {"train": subset["train_manifest_hash"], "validation": subset["validation_manifest_hash"]},
        "latent_cache_hash": subset["latent_cache_hash"], "validation_suite_hash": suite["validation_suite_hash"],
        "initialization_mode": args.initialization_mode, "parent_stage": parent.get("stage_name") if parent else None,
        "parent_checkpoint": str(parent_path) if parent_path else None, "parent_checkpoint_hash": parent_hash,
        "cumulative_optimizer_steps": cumulative_steps, "stage_local_steps": args.steps,
        "curriculum_history": (parent.get("curriculum_history", []) + [args.stage]) if parent else [args.stage],
        "git_commit": commit, "working_tree_dirty": dirty,
    }


def main() -> None:
    args = parse_args(); validate_external_steps(args.steps, allow_long_run=args.allow_long_run)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "stage": args.stage, "steps": args.steps, "output_dir": args.output_dir,
                          "command": " ".join(sys.argv)}, indent=2)); return
    out = Path(args.output_dir)
    if out.exists() and not args.overwrite and not args.resume: raise SystemExit(f"refusing existing output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config); config["seed"] = args.seed
    if args.device: config["device"] = args.device
    data = config["data"]
    subset = build_subset_manifest(Path(data["train_manifest"]), Path(data["valid_manifest"]), Path(data["latent_cache_dir"]), seed=args.seed)
    suite = build_stage_validation_suite(args.stage, args.seed, subset["train_utterance_ids"], subset["validation_utterance_ids"])
    device = resolve_device(config.get("device", "auto")); torch.manual_seed(args.seed); random.seed(args.seed)
    codec, model = build_components(config, device); codec.eval()
    for p in codec.parameters(): p.requires_grad_(False)
    optimizer = build_stage1_optimizer(model, learning_rate=config["train"]["learning_rate"], weight_decay=0.0)
    sources = {"train": RepresentationSource(config, codec, device, "train"), "val": RepresentationSource(config, codec, device, "val")}
    parent = None; parent_payload = None; saved_provenance = None; start = 0; history = []; channel_hashes = set(); noise_hashes = set(); cumulative = args.steps
    if args.parent_checkpoint:
        payload = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=False); parent_payload = payload; parent = payload["provenance"]
        validate_curriculum_parent(args.stage, args.initialization_mode, parent)
        model.load_state_dict(payload["model"], strict=True); optimizer.load_state_dict(payload["optimizer"])
        cumulative = int(parent["cumulative_optimizer_steps"]) + args.steps
    elif args.initialization_mode == "curriculum_resume" and args.stage != "o6_random_clean":
        raise SystemExit("curriculum_resume requires --parent-checkpoint")
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved = payload["provenance"]
        saved_provenance = saved
        for key, value in (("stage_name", args.stage), ("initialization_mode", args.initialization_mode),
                           ("validation_suite_hash", suite["validation_suite_hash"]), ("latent_cache_hash", subset["latent_cache_hash"])):
            if saved.get(key) != value: raise SystemExit(f"resume provenance mismatch: {key}")
        model.load_state_dict(payload["model"], strict=True); optimizer.load_state_dict(payload["optimizer"])
        start = int(payload["step"]); history = payload.get("history", [])
        channel_hashes.update(row["channel_hash"] for row in history if row.get("channel_hash"))
        noise_hashes.update(row["noise_hash"] for row in history if row.get("noise_hash"))
        cumulative = int(saved["cumulative_optimizer_steps"]) - int(saved["stage_local_steps"]) + args.steps
    provenance = _provenance(args, config, subset, suite, parent, cumulative)
    if saved_provenance is not None:
        commit, dirty = _git()
        provenance = {**saved_provenance, "stage_local_steps": args.steps,
                      "cumulative_optimizer_steps": cumulative, "git_commit": commit,
                      "working_tree_dirty": dirty}
    (out / "resolved_config.yaml").write_text(yaml.safe_dump({**config, "random_distribution": provenance}, sort_keys=True))
    (out / "subset_manifest.json").write_text(json.dumps(subset, indent=2)); (out / "validation_suite.json").write_text(json.dumps(suite, indent=2))
    (out / "command.txt").write_text(" ".join(sys.argv) + "\n")
    (out / "environment.json").write_text(json.dumps({"python": sys.version, "torch": torch.__version__, "platform": platform.platform(), **{k: provenance[k] for k in ("git_commit", "working_tree_dirty")}}, indent=2))
    derive = SeedDeriver(args.seed); weights = torch.ones(8, device=device); metrics_path = out / "metrics.jsonl"
    with metrics_path.open("a" if args.resume else "w") as handle:
        for step in range(start + 1, args.steps + 1):
            utterance_id = subset["train_utterance_ids"][derive.seed("utterance", step) % len(subset["train_utterance_ids"])]
            target, waveform = _example(sources["train"], utterance_id)
            sampled = sample_stage_distribution(args.stage, derive.seed("distribution", step))
            paired_seed = derive.seed("paired_channel_noise_jammer", step)
            stage_config = json.loads(json.dumps(config)); stage_config["channel"]["jammed_fraction"] = STAGE_DEFINITIONS[args.stage]["jammed_fraction"]
            batch = _make_batch(codec, model, stage_config, target=target, waveform=waveform,
                snr_db=sampled["snr_db"], jsr_db=sampled["jsr_db"], jammer_type=sampled["jammer_type"], seed=paired_seed, device=device)
            channel_hashes.add(stable_tensor_hash(batch.signal_fading)); noise_hashes.add(stable_tensor_hash(batch.noise))
            result = stage1_fixed_tx_step(codec, model, batch, optimizer, weights,
                latent_normalization=normalization_config(config), channel_estimator="dft_tap_ls", estimator_num_taps=6,
                estimator_ridge_lambda=config["channel"].get("estimator_ridge_lambda", 1e-6), gradient_clip_norm=config["train"].get("gradient_clip_norm"))
            record = {"step": step, "utterance_id": utterance_id, **sampled,
                      **_latent_summary(result["reconstruction"], target, float(result["loss"])),
                      "per_layer_normalized_mse": result["per_layer_mse"].cpu().tolist(),
                      "csi_nmse": float(result["csi_nmse"].mean()), "pilot_evm": float(result["pilot_evm"].mean()),
                      "data_evm": _data_evm(result, batch.pilot_mask),
                      "post_equalization_sinr_db": float(10 * torch.log10(result["post_equalization_sinr"].clamp_min(1e-12)).mean()),
                      "encoder_gradient_norm": _parameter_grad_norm(model.encoder), "decoder_gradient_norm": _parameter_grad_norm(model.decoder),
                      "channel_hash": stable_tensor_hash(batch.signal_fading), "noise_hash": stable_tensor_hash(batch.noise)}
            if step == 1 or step % args.validation_every == 0 or step == args.steps:
                record["validation"] = _validation(model, codec, config, sources, subset, suite, device, weights)
            history.append(record); handle.write(json.dumps(record) + "\n"); handle.flush()
            payload = {"diagnostic_type": "stage1_random_distribution", "provenance": provenance,
                       "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "history": history,
                       "rng_state": {"torch": torch.get_rng_state(), "python": random.getstate()}}
            if step % args.checkpoint_every == 0 or step == args.steps: torch.save(payload, out / "diagnostic_last.pt")
    final_validation = history[-1].get("validation", {})
    v2 = final_validation.get("V2", {})
    v2.update({"channel_hash_diversity": len(channel_hashes), "noise_hash_diversity": len(noise_hashes)})
    forgetting = None
    if parent_payload:
        previous_validation = next((row.get("validation") for row in reversed(parent_payload.get("history", [])) if row.get("validation")), {})
        if previous_validation.get("V2") and final_validation.get("V2"):
            forgetting = catastrophic_forgetting(previous_validation["V2"]["loss"], final_validation["V2"]["loss"])
    if forgetting: v2["previous_stage_relative_degradation"] = forgetting["relative_degradation"]
    if args.stage == "o6_random_clean":
        scenario_gates = {"V2": stage_gate(v2)}
    else:
        scenario_gates = {}
        for name, metrics in final_validation.items():
            candidate = dict(metrics); candidate.update({"channel_hash_diversity": len(channel_hashes), "noise_hash_diversity": len(noise_hashes)})
            if forgetting: candidate["previous_stage_relative_degradation"] = forgetting["relative_degradation"]
            scenario_gates[name] = stage_gate(candidate)
    gate = {"passed": all(item["passed"] for item in scenario_gates.values()),
            "reasons": sorted({reason for item in scenario_gates.values() for reason in item["reasons"]}),
            "scenario_gates": scenario_gates}
    losses = [float(row["loss"]) for row in history]; window = losses[max(0, int(len(losses) * .8)):]
    summary = {"provenance": provenance, "steps": args.steps, "best_step": min(history, key=lambda row: row["loss"])["step"],
               "train_final": history[-1], "validation": final_validation, "gate": gate,
               "path_learnability_pass": gate["passed"],
               "distribution_readiness": {"passed": gate["passed"], "v2_reported_separately": "V2" in final_validation,
                   "snr_slices_stable": all(name in final_validation for name in ("V3_5dB", "V3_10dB", "V3_15dB")),
                   "v2_materially_worse_than_v1": bool(v2 and final_validation.get("V1") and v2["loss"] > 1.25 * final_validation["V1"]["loss"])},
               "channel_hash_diversity": len(channel_hashes), "noise_hash_diversity": len(noise_hashes)}
    summary["fraction_finite_batches"] = sum(bool(row["finite"]) for row in history) / len(history)
    summary["final_window_loss_slope"] = linear_slope(window)
    summary["catastrophic_forgetting"] = forgetting
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
