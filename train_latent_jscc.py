from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from evaluation.paired import (
    PairedEvaluationBatch,
    estimate_transmitter_channel_state,
    estimate_transmitter_feedback,
    generate_paired_evaluation_batch,
    run_mode_on_paired_batch,
)
from channels.reliability import estimate_unreliable_mask
from models.latent_refiner import LatentRefiner, save_latent_refiner_checkpoint
from models.learned_gate import (
    LearnedLayerGate,
    gate_budget_loss,
    gate_smoothness_loss,
    save_learned_gate_checkpoint,
)
from speech_jscc.channels import compute_jsr
from speech_jscc.checkpoint import build_checkpoint_metadata, normalization_config
from speech_jscc.config import load_config, resolve_device
from speech_jscc.data import (
    CachedCodecDataset,
    codec_cache_namespace,
    resolve_waveform_splits,
    synthetic_waveforms,
)
from speech_jscc.experiment import build_components
from speech_jscc.layer_importance import resolve_layer_importance_config


def _validate_range(name: str, bounds: list[float]) -> tuple[float, float]:
    if len(bounds) != 2 or float(bounds[0]) > float(bounds[1]):
        raise ValueError(f"{name} must be [minimum, maximum]")
    return float(bounds[0]), float(bounds[1])


def sample_uniform_db(batch_size: int, bounds: list[float], device: torch.device) -> Tensor:
    low, high = _validate_range("dB range", bounds)
    return torch.empty(batch_size, device=device).uniform_(low, high)


def sample_jammer_type(probabilities: dict[str, float]) -> str:
    supported = {"barrage", "narrowband", "burst", "pilot"}
    if not probabilities or not set(probabilities).issubset(supported):
        raise ValueError(f"jammer_probabilities keys must be a nonempty subset of {sorted(supported)}")
    names = list(probabilities)
    weights = [float(probabilities[name]) for name in names]
    if any(weight < 0 for weight in weights) or abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("jammer probabilities must be nonnegative and sum to 1")
    return random.choices(names, weights=weights, k=1)[0]


def layer_weighted_latent_mse(
    reconstruction: Tensor,
    target: Tensor,
    layer_weights: Tensor,
    normalization: str | dict[str, Any] = "none",
) -> tuple[Tensor, Tensor]:
    """Return weighted loss and raw detached per-layer MSE `[L]`.

    ``per_layer_power`` divides each layer error by that target layer's power,
    preventing high-energy early RVQ layers from dominating solely by scale.
    """
    if reconstruction.shape != target.shape or reconstruction.ndim != 4:
        raise ValueError("reconstruction and target must match [B,L,T,D]")
    weights = layer_weights.to(device=reconstruction.device, dtype=reconstruction.dtype)
    if weights.shape != (target.shape[1],) or torch.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("layer_weights must contain L nonnegative values with positive sum")
    per_layer = (reconstruction - target).square().mean(dim=(0, 2, 3))
    options = {"mode": normalization} if isinstance(normalization, str) else normalization
    mode = options.get("mode", "none")
    epsilon = float(options.get("epsilon", 1e-8))
    if mode in {"none", "raw"}:
        loss_values = per_layer
    elif mode in {"per_layer_power", "per_layer_nmse"}:
        target_power = target.square().mean(dim=(0, 2, 3)).detach()
        loss_values = per_layer / target_power.clamp_min(epsilon)
    elif mode == "global_power":
        loss_values = per_layer / target.square().mean().detach().clamp_min(epsilon)
    else:
        raise ValueError(
            "latent normalization mode must be none, per_layer_power, or global_power"
        )
    loss = (loss_values * weights).sum() / weights.sum()
    return loss, per_layer.detach()


class RepresentationSource:
    """Load precomputed tensors, a waveform corpus, or mock synthetic inputs."""

    def __init__(
        self,
        config: dict[str, Any],
        codec,
        device: torch.device,
        split: str = "train",
    ):
        self.config = config
        self.codec = codec
        self.device = device
        self.split = split
        self.store: Tensor | None = None
        self.dataset: CachedCodecDataset | None = None
        self.generator = torch.Generator().manual_seed(
            int(config.get("seed", 0)) + (0 if split == "train" else 100_000)
        )
        path_value = config.get("data", {}).get("representations_path")
        if path_value:
            loaded = torch.load(Path(path_value), map_location="cpu", weights_only=True)
            if isinstance(loaded, dict):
                loaded = loaded.get(split, loaded.get("representations"))
            if not isinstance(loaded, Tensor) or loaded.ndim != 4:
                raise ValueError("representation file must be a tensor or contain 'representations' [N,L,T,D]")
            if tuple(loaded.shape[1:]) != codec.representation_shape:
                raise ValueError(
                    f"loaded representation shape {tuple(loaded.shape[1:])} does not match "
                    f"codec shape {codec.representation_shape}"
                )
            self.store = loaded.to(dtype=torch.float32)
            return

        train_paths, val_paths = resolve_waveform_splits(
            config.get("data", {}), config.get("seed", 0)
        )
        paths = train_paths if split == "train" else val_paths
        if paths:
            sample_rate = int(getattr(codec, "sample_rate", config["codec"].get("sample_rate", 16000)))
            cache_dir = config.get("data", {}).get("latent_cache_dir")
            self.dataset = CachedCodecDataset(
                paths,
                codec,
                sample_rate=sample_rate,
                waveform_samples=config["codec"]["waveform_samples"],
                device=device,
                split=split,
                cache_dir=cache_dir,
                cache_namespace=codec_cache_namespace(config, codec),
            )

    @property
    def description(self) -> str:
        if self.store is not None:
            return "precomputed" if self.split == "train" else "precomputed:val"
        if self.dataset is not None:
            cache = "cached" if self.dataset.cache_dir is not None else "uncached"
            return f"waveform_corpus:{self.split}:{cache}"
        return "mock-codec synthetic waveforms"

    def next_batch(self, batch_size: int) -> tuple[Tensor, Tensor | None]:
        if self.store is not None:
            indices = torch.randint(
                0, self.store.shape[0], (batch_size,), generator=self.generator
            )
            return self.store[indices].to(self.device), None
        if self.dataset is not None:
            indices = torch.randint(
                0, len(self.dataset), (batch_size,), generator=self.generator
            ).tolist()
            examples = [self.dataset[index] for index in indices]
            return (
                torch.stack([example[0] for example in examples]),
                torch.stack([example[1] for example in examples]),
            )
        waveform = synthetic_waveforms(
            batch_size,
            self.config["codec"]["waveform_samples"],
            self.device,
        )
        with torch.no_grad():
            representation = self.codec.encode_waveform(waveform)
        return representation, waveform


def joint_learned_gate_step(
    codec,
    model,
    learned_gate: LearnedLayerGate,
    latent_refiner: LatentRefiner,
    paired_batch: PairedEvaluationBatch,
    optimizer: torch.optim.Optimizer | None,
    layer_weights: Tensor,
    *,
    lambda_budget: float,
    lambda_smooth: float,
    lambda_refine: float,
    power_penalty_weight: float,
    gradient_clip_norm: float | None = None,
    transmitter_csi: bool = True,
    refiner_mask_mode: str = "estimated",
    allocation_mode: str = "reliability_greedy",
    importance_order: list[int] | tuple[int, ...] | None = None,
    unreliable_fraction: float = 0.25,
    latent_normalization: str | dict[str, Any] = "none",
    channel_estimator: str = "auto",
    estimator_num_taps: int | None = None,
    estimator_ridge_lambda: float = 1.0e-6,
) -> dict[str, Tensor]:
    """Run one joint JSCC/gate update on a fixed channel realization."""
    channel_shape = tuple(model.encoder.channel_shape)
    fading = "flat" if len(channel_shape) == 1 else "ofdm"
    feedback = estimate_transmitter_feedback(
        paired_batch,
        transmitter_csi=transmitter_csi,
        fading=fading,
        channel_estimator=channel_estimator,
        estimator_num_taps=estimator_num_taps,
        estimator_ridge_lambda=estimator_ridge_lambda,
    )
    transmitter_state = feedback["state"].detach()
    reliability = feedback["reliability"].detach()
    alpha = learned_gate(transmitter_state)
    result = run_mode_on_paired_batch(
        codec,
        model,
        paired_batch,
        transmitter_state,
        alpha,
        equalizer="estimated",
        fading=fading,
        channel_estimator=channel_estimator,
        estimator_num_taps=estimator_num_taps,
        estimator_ridge_lambda=estimator_ridge_lambda,
        allocation_mode=allocation_mode,
        importance_order=importance_order,
        resource_reliability=reliability,
    )
    reconstruction_loss, per_layer_mse = layer_weighted_latent_mse(
        result["reconstruction"],
        paired_batch.representation,
        layer_weights,
        latent_normalization,
    )
    if refiner_mask_mode == "oracle":
        refiner_mask = paired_batch.jammer_mask
    elif refiner_mask_mode == "estimated":
        refiner_mask = estimate_unreliable_mask(reliability, unreliable_fraction)
    else:
        raise ValueError("refiner_mask_mode must be 'oracle' or 'estimated'")
    refined = latent_refiner(
        result["reconstruction"], result["decoder_state"], refiner_mask
    )
    refine_loss, refined_layer_mse = layer_weighted_latent_mse(
        refined, paired_batch.representation, layer_weights, latent_normalization
    )
    budget_loss = gate_budget_loss(alpha)
    smoothness_loss = gate_smoothness_loss(alpha)
    power_dimensions = tuple(range(1, result["transmitted"].ndim))
    sample_power = result["transmitted"].abs().square().mean(power_dimensions)
    power_penalty = (sample_power - model.encoder.target_power).square().mean()
    total_loss = (
        reconstruction_loss
        + float(lambda_refine) * refine_loss
        + float(lambda_budget) * budget_loss
        + float(lambda_smooth) * smoothness_loss
        + float(power_penalty_weight) * power_penalty
    )
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters())
                + list(learned_gate.parameters())
                + list(latent_refiner.parameters()),
                gradient_clip_norm,
            )
        optimizer.step()
    return {
        "loss": total_loss.detach(),
        "reconstruction_loss": reconstruction_loss.detach(),
        "refine_loss": refine_loss.detach(),
        "budget_loss": budget_loss.detach(),
        "smoothness_loss": smoothness_loss.detach(),
        "power_penalty": power_penalty.detach(),
        "sample_power": sample_power.detach(),
        "per_layer_mse": per_layer_mse,
        "refined_layer_mse": refined_layer_mse,
        "alpha": alpha.detach(),
        "encoder_state": result["encoder_state"].detach(),
        "decoder_state": result["decoder_state"].detach(),
        "effective_sinr": result["post_equalization_sinr"].detach(),
        "csi_nmse": result["csi_nmse"].detach(),
        "pilot_evm": result["pilot_evm"].detach(),
        "reconstruction": refined.detach(),
        "raw_reconstruction": result["reconstruction"].detach(),
        "decoded_waveform": codec.decode_representation(refined.detach()),
        "transmitted": result["transmitted"].detach(),
        "jammer_mask": result["jammer_mask"].detach(),
        "measured_jsr": compute_jsr(result["transmitted"], result["jammer"]).detach(),
        "resource_reliability": reliability,
        "refiner_mask": refiner_mask.detach(),
    }


def _save_examples(
    directory: Path,
    step: int,
    count: int,
    target: Tensor,
    reconstruction: Tensor,
    waveform: Tensor | None,
    decoded_waveform: Tensor,
    metadata: dict[str, Any],
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step_{step:06d}.pt"
    examples: dict[str, Any] = {
        "target_representation": target[:count].detach().cpu(),
        "reconstructed_representation": reconstruction[:count].detach().cpu(),
        "decoded_waveform": decoded_waveform[:count].detach().cpu(),
        "metadata": metadata,
    }
    if waveform is not None:
        examples["source_waveform"] = waveform[:count].detach().cpu()
    torch.save(examples, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train continuous-latent speech JSCC")
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    device = resolve_device(config["device"])
    codec, model = build_components(config, device)
    if config["model"]["channel_state_dim"] != 8:
        raise ValueError("Sprint 3 channel state requires model.channel_state_dim: 8")
    learned_gate = LearnedLayerGate(
        config["model"]["channel_state_dim"],
        config["model"]["layers"],
        config["train"].get("learned_gate_hidden_dim", 32),
    ).to(device)
    latent_refiner = LatentRefiner(
        (config["model"]["layers"], config["model"]["frames"], config["model"]["latent_dim"]),
        config["model"]["channel_state_dim"],
        config["train"].get("refiner_hidden_dim", 64),
    ).to(device)
    source = RepresentationSource(config, codec, device, split="train")
    validation_source = RepresentationSource(config, codec, device, split="val")
    if config["codec"].get("type", "mock").lower() == "speechtokenizer":
        if source.dataset is None and source.store is None:
            raise ValueError(
                "SpeechTokenizer training requires data.waveform_dir, waveform_paths, "
                "train/val manifests, or precomputed representations"
            )
        if codec.model.training or any(parameter.requires_grad for parameter in codec.parameters()):
            raise RuntimeError("SpeechTokenizer must remain frozen during JSCC training")

    train_config = config["train"]
    resolved_importance = resolve_layer_importance_config(
        config,
        section="train",
        expected_representation_shape=codec.representation_shape,
    )
    if resolved_importance.artifact_path is not None:
        if resolved_importance.layer_weights is not None:
            train_config["layer_weights"] = resolved_importance.layer_weights
        if resolved_importance.layer_importance_order is not None:
            train_config["layer_importance_order"] = resolved_importance.layer_importance_order
        if resolved_importance.base_layers is not None:
            train_config["base_layers"] = resolved_importance.base_layers
        print(
            "layer_importance_artifact="
            f"{resolved_importance.artifact_path} hash={resolved_importance.artifact_hash} "
            f"weights={train_config.get('layer_weights')} "
            f"order={train_config.get('layer_importance_order')} "
            f"base_layers={train_config.get('base_layers')}"
        )
    channel_config = config["channel"]
    layer_weights = torch.tensor(train_config["layer_weights"], device=device, dtype=torch.float32)
    if layer_weights.shape != (config["model"]["layers"],):
        raise ValueError("train.layer_weights must have one value per codec layer")
    snr_range = channel_config["snr_db_range"]
    jsr_range = channel_config["jsr_db_range"]
    _validate_range("snr_db_range", snr_range)
    _validate_range("jsr_db_range", jsr_range)
    probabilities = channel_config["jammer_probabilities"]

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(learned_gate.parameters()),
        # Refiner parameters are optimized jointly with JSCC and the gate.
        lr=train_config["learning_rate"],
    )
    optimizer.add_param_group({"params": list(latent_refiner.parameters())})
    log_path = Path(train_config["metrics_jsonl"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    reconstruction_dir = Path(train_config["reconstruction_dir"])
    latent_normalization = normalization_config(config)
    print(
        f"device={device} representation_source={source.description} "
        f"validation_source={validation_source.description} "
        f"latent_normalization={latent_normalization['mode']}"
    )

    model.train()
    learned_gate.train()
    latent_refiner.train()
    with log_path.open("w", encoding="utf-8") as metric_log:
        for step in range(1, train_config["steps"] + 1):
            batch_size = train_config["batch_size"]
            target, waveform = source.next_batch(batch_size)
            snr_value = sample_uniform_db(1, snr_range, device).item()
            jsr_value = sample_uniform_db(1, jsr_range, device).item()
            jammer_type = sample_jammer_type(probabilities)
            channel_shape = tuple(model.encoder.channel_shape)
            fading = channel_config.get(
                "fading", "flat" if len(channel_shape) == 1 else "ofdm"
            )
            paired_batch = generate_paired_evaluation_batch(
                codec,
                batch_size=batch_size,
                waveform_samples=config["codec"]["waveform_samples"],
                channel_shape=channel_shape,
                snr_db=snr_value,
                jsr_db=jsr_value,
                jammer_type=jammer_type,
                jammed_fraction=channel_config["jammed_fraction"],
                pilot_spacing=channel_config.get("pilot_spacing", 4),
                pilot_time_spacing=channel_config.get("pilot_time_spacing"),
                target_power=config["model"]["target_power"],
                seed=config["seed"] + step,
                device=device,
                fading=fading,
                num_taps=channel_config.get("num_taps", 6),
                pdp_decay=channel_config.get("pdp_decay", 0.7),
                channel_estimator=channel_config.get("channel_estimator", "auto"),
                estimator_num_taps=channel_config.get("estimator_num_taps"),
                estimator_ridge_lambda=channel_config.get("estimator_ridge_lambda", 1e-6),
                waveform=waveform,
                representation=target,
            )
            step_result = joint_learned_gate_step(
                codec,
                model,
                learned_gate,
                latent_refiner,
                paired_batch,
                optimizer,
                layer_weights,
                lambda_budget=train_config["lambda_budget"],
                lambda_smooth=train_config["lambda_smooth"],
                lambda_refine=train_config["lambda_refine"],
                power_penalty_weight=train_config["power_penalty_weight"],
                gradient_clip_norm=train_config.get("gradient_clip_norm"),
                transmitter_csi=train_config.get("transmitter_csi", True),
                channel_estimator=channel_config.get("channel_estimator", "auto"),
                estimator_num_taps=channel_config.get("estimator_num_taps"),
                estimator_ridge_lambda=channel_config.get("estimator_ridge_lambda", 1e-6),
                refiner_mask_mode=train_config.get("refiner_mask_mode", "estimated"),
                allocation_mode=train_config.get("allocation_mode", "reliability_greedy"),
                importance_order=train_config.get("layer_importance_order"),
                unreliable_fraction=train_config.get("unreliable_fraction", 0.25),
                latent_normalization=latent_normalization,
            )

            should_log = step == 1 or step % train_config["log_every"] == 0
            if should_log:
                metrics = {
                    "step": step,
                    "loss": step_result["loss"].item(),
                    "latent_loss": step_result["reconstruction_loss"].item(),
                    "refine_loss": step_result["refine_loss"].item(),
                    "budget_loss": step_result["budget_loss"].item(),
                    "smoothness_loss": step_result["smoothness_loss"].item(),
                    "power_penalty": step_result["power_penalty"].item(),
                    "transmit_power": step_result["sample_power"].mean().item(),
                    "effective_sinr_db": (
                        10.0 * torch.log10(step_result["effective_sinr"].clamp_min(1e-12))
                    ).mean().item(),
                    "requested_snr_db": snr_value,
                    "requested_jsr_db": jsr_value,
                    "measured_jsr_db": (
                        10.0 * torch.log10(step_result["measured_jsr"].clamp_min(1e-12))
                    ).mean().item(),
                    "jammer_type": jammer_type,
                    "jammer_mask_ratio": step_result["jammer_mask"].float().mean().item(),
                    "csi_nmse": step_result["csi_nmse"].mean().item(),
                    "pilot_evm": step_result["pilot_evm"].mean().item(),
                    "layer_mse": step_result["per_layer_mse"].cpu().tolist(),
                    "refined_layer_mse": step_result["refined_layer_mse"].cpu().tolist(),
                    "alpha_mean": step_result["alpha"].mean(dim=0).cpu().tolist(),
                    "encoder_state_mean": step_result["encoder_state"].mean(dim=0).cpu().tolist(),
                    "decoder_state_mean": step_result["decoder_state"].mean(dim=0).cpu().tolist(),
                    "latent_normalization": latent_normalization,
                }
                if validation_source.dataset is not None or validation_source.store is not None:
                    val_target, val_waveform = validation_source.next_batch(
                        train_config.get("val_batch_size", batch_size)
                    )
                    val_batch = generate_paired_evaluation_batch(
                        codec,
                        batch_size=val_target.shape[0],
                        waveform_samples=config["codec"]["waveform_samples"],
                        channel_shape=channel_shape,
                        snr_db=snr_value,
                        jsr_db=jsr_value,
                        jammer_type=jammer_type,
                        jammed_fraction=channel_config["jammed_fraction"],
                        pilot_spacing=channel_config.get("pilot_spacing", 4),
                        pilot_time_spacing=channel_config.get("pilot_time_spacing"),
                        target_power=config["model"]["target_power"],
                        seed=config["seed"] + 1_000_000 + step,
                        device=device,
                        fading=fading,
                        num_taps=channel_config.get("num_taps", 6),
                        pdp_decay=channel_config.get("pdp_decay", 0.7),
                        channel_estimator=channel_config.get("channel_estimator", "auto"),
                        estimator_num_taps=channel_config.get("estimator_num_taps"),
                        estimator_ridge_lambda=channel_config.get("estimator_ridge_lambda", 1e-6),
                        waveform=val_waveform,
                        representation=val_target,
                    )
                    with torch.no_grad():
                        val_result = joint_learned_gate_step(
                            codec,
                            model,
                            learned_gate,
                            latent_refiner,
                            val_batch,
                            None,
                            layer_weights,
                            lambda_budget=train_config["lambda_budget"],
                            lambda_smooth=train_config["lambda_smooth"],
                            lambda_refine=train_config["lambda_refine"],
                            power_penalty_weight=train_config["power_penalty_weight"],
                            transmitter_csi=train_config.get("transmitter_csi", True),
                            channel_estimator=channel_config.get("channel_estimator", "auto"),
                            estimator_num_taps=channel_config.get("estimator_num_taps"),
                            estimator_ridge_lambda=channel_config.get("estimator_ridge_lambda", 1e-6),
                            refiner_mask_mode=train_config.get("refiner_mask_mode", "estimated"),
                            allocation_mode=train_config.get("allocation_mode", "reliability_greedy"),
                            importance_order=train_config.get("layer_importance_order"),
                            unreliable_fraction=train_config.get("unreliable_fraction", 0.25),
                            latent_normalization=latent_normalization,
                        )
                    metrics["val_loss"] = val_result["loss"].item()
                    metrics["val_latent_loss"] = val_result["reconstruction_loss"].item()
                    metrics["val_layer_mse"] = val_result["per_layer_mse"].cpu().tolist()
                example_path = _save_examples(
                    reconstruction_dir,
                    step,
                    train_config["reconstruction_examples"],
                    target,
                    step_result["reconstruction"],
                    waveform,
                    step_result["decoded_waveform"],
                    metrics,
                )
                metrics["reconstruction_file"] = str(example_path)
                line = json.dumps(metrics, sort_keys=True)
                metric_log.write(line + "\n")
                metric_log.flush()
                print(line)

    checkpoint = Path(train_config["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    metadata = build_checkpoint_metadata(
        config,
        codec,
        representation_source=source.description,
    )
    if resolved_importance.artifact_path is not None:
        metadata["layer_importance"] = {
            "path": resolved_importance.artifact_path,
            "artifact_hash": resolved_importance.artifact_hash,
            "layer_weights_mean_one": resolved_importance.artifact.layer_weights_mean_one
            if resolved_importance.artifact is not None
            else None,
            "layer_importance_order": resolved_importance.layer_importance_order,
            "base_layers": resolved_importance.base_layers,
        }
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "learned_gate": save_learned_gate_checkpoint(learned_gate),
            "latent_refiner": save_latent_refiner_checkpoint(latent_refiner),
            "step": train_config["steps"],
            "config": config,
            "metadata": metadata,
        },
        checkpoint,
    )
    print(f"saved_checkpoint={checkpoint} metrics={log_path}")


if __name__ == "__main__":
    main()
