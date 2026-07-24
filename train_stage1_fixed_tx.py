from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch

from evaluation.paired import generate_paired_evaluation_batch
from speech_jscc.checkpoint import normalization_config
from speech_jscc.config import load_config, resolve_device
from speech_jscc.experiment import build_components
from speech_jscc.layer_importance import file_sha256, resolve_layer_importance_config
from speech_jscc.training.stage1 import (
    STAGE1_LABEL,
    assert_stage1_startup_invariants,
    build_stage1_checkpoint_payload,
    build_stage1_optimizer,
    stage1_fixed_tx_step,
    stage1_metadata,
)
from train_latent_jscc import RepresentationSource, _validate_range, sample_jammer_type, sample_uniform_db


VALIDATION_SCENARIOS = (
    {"name": "clean_snr5", "snr_db": 5.0, "jsr_db": 0.0, "jammer_type": "none"},
    {"name": "clean_snr10", "snr_db": 10.0, "jsr_db": 0.0, "jammer_type": "none"},
    {"name": "clean_snr15", "snr_db": 15.0, "jsr_db": 0.0, "jammer_type": "none"},
    {"name": "barrage_snr10_jsr0", "snr_db": 10.0, "jsr_db": 0.0, "jammer_type": "barrage"},
    {"name": "narrowband_snr10_jsr0", "snr_db": 10.0, "jsr_db": 0.0, "jammer_type": "narrowband"},
    {"name": "burst_snr10_jsr0", "snr_db": 10.0, "jsr_db": 0.0, "jammer_type": "burst"},
    {"name": "pilot_snr10_jsr0", "snr_db": 10.0, "jsr_db": 0.0, "jammer_type": "pilot"},
)


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    train = config.setdefault("train", {})
    if args.steps is not None:
        train["steps"] = int(args.steps)
    if args.batch_size is not None:
        train["batch_size"] = int(args.batch_size)
        train["val_batch_size"] = int(args.batch_size)
    if args.output_dir is not None:
        root = Path(args.output_dir)
        train["checkpoint_best"] = str(root / "stage1_best.pt")
        train["checkpoint_last"] = str(root / "stage1_last.pt")
        train["metrics_jsonl"] = str(root / "metrics.jsonl")
        train["reconstruction_dir"] = str(root / "reconstructions")


def _validate_stage1_config(config: dict[str, Any]) -> None:
    train = config.get("train", {})
    channel = config.get("channel", {})
    if train.get("training_stage") != "stage1_fixed_tx":
        raise ValueError("train.training_stage must be stage1_fixed_tx")
    required = {
        "transmitter_state_mode": "neutral",
        "receiver_state_mode": "observable_v1",
        "gate_mode": "all_ones",
        "allocation_mode": "uniform",
        "power_mode": "uniform",
        "equalizer": "estimated",
    }
    for key, expected in required.items():
        if train.get(key) != expected:
            raise ValueError(f"train.{key} must be {expected!r}")
    for key in ("use_learned_gate", "use_refiner", "use_jammer_estimator"):
        if bool(train.get(key, False)):
            raise ValueError(f"Stage-1 requires train.{key}: false")
    if channel.get("fading") != "multipath_block":
        raise ValueError("Stage-1 requires channel.fading: multipath_block")
    if channel.get("channel_estimator") != "dft_tap_ls":
        raise ValueError("Stage-1 requires channel.channel_estimator: dft_tap_ls")
    if int(config["codec"].get("waveform_samples", 0)) != 16000:
        raise ValueError("Stage-1 baseline configuration must use codec.waveform_samples: 16000")


def _sample_training_channel(config: dict[str, Any], device: torch.device) -> tuple[float, float, str, bool]:
    channel = config["channel"]
    snr_value = sample_uniform_db(1, channel["snr_db_range"], device).item()
    if random.random() < float(channel.get("clean_probability", 0.2)):
        return snr_value, 0.0, "none", True
    jsr_value = sample_uniform_db(1, channel["jsr_db_range"], device).item()
    return snr_value, jsr_value, sample_jammer_type(channel["jammer_probabilities"]), False


def _make_batch(
    codec,
    model,
    config: dict[str, Any],
    *,
    target: torch.Tensor,
    waveform: torch.Tensor | None,
    snr_db: float,
    jsr_db: float,
    jammer_type: str,
    seed: int,
    device: torch.device,
):
    channel = config["channel"]
    return generate_paired_evaluation_batch(
        codec,
        batch_size=target.shape[0],
        waveform_samples=config["codec"]["waveform_samples"],
        channel_shape=tuple(config["model"].get("grid_shape", model.encoder.channel_shape)),
        snr_db=snr_db,
        jsr_db=jsr_db,
        jammer_type=jammer_type,
        jammed_fraction=channel["jammed_fraction"],
        pilot_spacing=channel.get("pilot_spacing", 4),
        pilot_time_spacing=channel.get("pilot_time_spacing", 4),
        target_power=config["model"]["target_power"],
        seed=seed,
        device=device,
        fading=channel.get("fading", "multipath_block"),
        num_taps=channel.get("num_taps", 6),
        pdp_decay=channel.get("pdp_decay", 0.7),
        channel_estimator=channel.get("channel_estimator", "dft_tap_ls"),
        estimator_num_taps=channel.get("estimator_num_taps"),
        estimator_ridge_lambda=channel.get("estimator_ridge_lambda", 1.0e-6),
        waveform=waveform,
        representation=target,
    )


def _fixed_validation_batches(codec, model, config, source, device: torch.device) -> list[tuple[str, Any]]:
    batches = []
    batch_size = int(config["train"].get("val_batch_size", config["train"]["batch_size"]))
    for index, scenario in enumerate(VALIDATION_SCENARIOS):
        target, waveform = source.next_batch(batch_size)
        batches.append(
            (
                scenario["name"],
                _make_batch(
                    codec,
                    model,
                    config,
                    target=target,
                    waveform=waveform,
                    snr_db=scenario["snr_db"],
                    jsr_db=scenario["jsr_db"],
                    jammer_type=scenario["jammer_type"],
                    seed=int(config["seed"]) + 500_000 + index,
                    device=device,
                ),
            )
        )
    return batches


def _validate(model, codec, config, layer_weights, latent_normalization, fixed_batches) -> dict[str, Any]:
    model.eval()
    scenario_metrics = {}
    losses = []
    with torch.no_grad():
        for name, batch in fixed_batches:
            result = stage1_fixed_tx_step(
                codec,
                model,
                batch,
                None,
                layer_weights,
                latent_normalization=latent_normalization,
                channel_estimator=config["channel"].get("channel_estimator", "dft_tap_ls"),
                estimator_num_taps=config["channel"].get("estimator_num_taps"),
                estimator_ridge_lambda=config["channel"].get("estimator_ridge_lambda", 1.0e-6),
            )
            loss = float(result["loss"])
            losses.append(loss)
            scenario_metrics[name] = {
                "weighted_latent_loss": loss,
                "csi_nmse": float(result["csi_nmse"].mean()),
                "pilot_evm": float(result["pilot_evm"].mean()),
            }
    model.train()
    return {
        "val_weighted_latent_loss": sum(losses) / len(losses),
        "validation_scenarios": scenario_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage-1 fixed-Tx SpeechTokenizer latent JSCC")
    parser.add_argument("--config", default="configs/train_stage1_fixed_tx_uniform.yaml")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    config = load_config(args.config)
    _apply_cli_overrides(config, args)
    _validate_stage1_config(config)
    torch.manual_seed(int(config["seed"]))
    random.seed(int(config["seed"]))
    device = resolve_device(config.get("device", "auto"))
    codec, model = build_components(config, device)
    codec.eval()
    for parameter in codec.parameters():
        parameter.requires_grad_(False)
    if tuple(codec.representation_shape) != (8, 50, 1024) and config["codec"].get("type") == "speechtokenizer":
        raise ValueError(f"SpeechTokenizer Stage-1 requires representation shape [8,50,1024], got {codec.representation_shape}")

    source = RepresentationSource(config, codec, device, split="train")
    validation_source = RepresentationSource(config, codec, device, split="val")
    if source.dataset is None and source.store is None and config["codec"].get("type") == "speechtokenizer":
        raise ValueError("SpeechTokenizer Stage-1 requires train/valid manifests or cached representations")
    optimizer = build_stage1_optimizer(
        model,
        learning_rate=config["train"]["learning_rate"],
        weight_decay=config["train"].get("weight_decay", 0.0),
    )
    invariants = assert_stage1_startup_invariants(
        codec, model, optimizer, allocation_mode=config["train"]["allocation_mode"]
    )
    _validate_range("snr_db_range", config["channel"]["snr_db_range"])
    _validate_range("jsr_db_range", config["channel"]["jsr_db_range"])

    resolved_importance = resolve_layer_importance_config(
        config,
        section="train",
        expected_representation_shape=codec.representation_shape,
    )
    importance_options = config.get("layer_importance") or {}
    if importance_options.get("require_non_smoke"):
        if resolved_importance.artifact is None:
            raise ValueError("weighted Stage-1 requires a layer_importance artifact")
        artifact_meta = resolved_importance.artifact.metadata
        evaluated_items = int((artifact_meta.get("dataset") or {}).get("evaluated_items", 0))
        marker = json.dumps(artifact_meta, sort_keys=True).lower()
        if evaluated_items < 20 or "smoke" in marker or "test_artifact" in marker:
            raise ValueError(
                "weighted Stage-1 refuses smoke/test layer-importance artifacts; "
                f"evaluated_items={evaluated_items}"
            )
    layer_weights_value = resolved_importance.layer_weights or config["train"].get("layer_weights")
    layer_weights = torch.tensor(layer_weights_value, device=device, dtype=torch.float32)
    if layer_weights.shape != (model.encoder.num_layers,):
        raise ValueError("Stage-1 layer weights must match the codec layer count")
    latent_normalization = normalization_config(config)
    fixed_validation = _fixed_validation_batches(codec, model, config, validation_source, device)

    log_path = Path(config["train"]["metrics_jsonl"])
    best_path = Path(config["train"]["checkpoint_best"])
    last_path = Path(config["train"]["checkpoint_last"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_metric: float | None = None
    best_step = 0
    print(
        f"stage={STAGE1_LABEL} device={device} source={source.description} "
        f"validation_source={validation_source.description} invariants={invariants}"
    )

    model.train()
    with log_path.open("w", encoding="utf-8") as handle:
        for step in range(1, int(config["train"]["steps"]) + 1):
            target, waveform = source.next_batch(int(config["train"]["batch_size"]))
            snr_db, jsr_db, jammer_type, clean = _sample_training_channel(config, device)
            batch = _make_batch(
                codec,
                model,
                config,
                target=target,
                waveform=waveform,
                snr_db=snr_db,
                jsr_db=jsr_db,
                jammer_type=jammer_type,
                seed=int(config["seed"]) + step,
                device=device,
            )
            result = stage1_fixed_tx_step(
                codec,
                model,
                batch,
                optimizer,
                layer_weights,
                latent_normalization=latent_normalization,
                channel_estimator=config["channel"].get("channel_estimator", "dft_tap_ls"),
                estimator_num_taps=config["channel"].get("estimator_num_taps"),
                estimator_ridge_lambda=config["channel"].get("estimator_ridge_lambda", 1.0e-6),
                gradient_clip_norm=config["train"].get("gradient_clip_norm"),
            )
            should_validate = step == 1 or step % int(config["train"]["validate_every"]) == 0
            metrics = {
                "step": step,
                "model_input_metrics": {
                    "receiver_state_mean": result["receiver_state"].mean(dim=0).cpu().tolist(),
                    "receiver_state_std": result["receiver_state"].std(dim=0, unbiased=False).cpu().tolist(),
                },
                "offline_diagnostic_metrics": {
                    "csi_nmse": float(result["csi_nmse"].mean()),
                    "pilot_evm": float(result["pilot_evm"].mean()),
                    "post_equalization_sinr_db": float(10.0 * torch.log10(result["post_equalization_sinr"].clamp_min(1e-12)).mean()),
                },
                "loss": float(result["loss"]),
                "per_layer_raw_mse": result["per_layer_mse"].cpu().tolist(),
                "transmit_power": float(result["transmitted"].abs().square().mean()),
                "layer_gate_mean": result["layer_gates"].mean(dim=0).cpu().tolist(),
                "layer_power_fractions": result["layer_power_fractions"].mean(dim=0).cpu().tolist(),
                "transmitter_state_mean": result["transmitter_state"].mean(dim=0).cpu().tolist(),
                "requested_snr_db": snr_db,
                "requested_jsr_db": jsr_db,
                "jammer_type": jammer_type,
                "clean": clean,
                "batch_shared_channel_sampling": True,
            }
            if should_validate:
                validation_metrics = _validate(model, codec, config, layer_weights, latent_normalization, fixed_validation)
                metrics.update(validation_metrics)
                current = float(validation_metrics["val_weighted_latent_loss"])
                if best_metric is None or current < best_metric:
                    best_metric = current
                    best_step = step
                    metadata = stage1_metadata(
                        config,
                        representation_shape=codec.representation_shape,
                        layer_weights=layer_weights.detach().cpu().tolist(),
                        representation_source=source.description,
                        layer_importance={
                            "path": resolved_importance.artifact_path,
                            "artifact_hash": resolved_importance.artifact_hash,
                        },
                    )
                    metadata["config_hash"] = file_sha256(args.config) if Path(args.config).exists() else None
                    torch.save(
                        build_stage1_checkpoint_payload(
                            model,
                            optimizer,
                            step=step,
                            best_metric=best_metric,
                            config=config,
                            metadata=metadata,
                        ),
                        best_path,
                    )
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
            handle.flush()
            if step == 1 or step % int(config["train"]["log_every"]) == 0:
                print(json.dumps(metrics, sort_keys=True))

    metadata = stage1_metadata(
        config,
        representation_shape=codec.representation_shape,
        layer_weights=layer_weights.detach().cpu().tolist(),
        representation_source=source.description,
        layer_importance={
            "path": resolved_importance.artifact_path,
            "artifact_hash": resolved_importance.artifact_hash,
        },
    )
    metadata["best_step"] = best_step
    metadata["config_hash"] = file_sha256(args.config) if Path(args.config).exists() else None
    torch.save(
        build_stage1_checkpoint_payload(
            model,
            optimizer,
            step=int(config["train"]["steps"]),
            best_metric=best_metric,
            config=config,
            metadata=metadata,
        ),
        last_path,
    )
    print(f"saved_best={best_path} saved_last={last_path} metrics={log_path} best_metric={best_metric}")


if __name__ == "__main__":
    main()
