"""Supervised training utilities for R4 jammer estimation and latent refinement."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from channels.multipath import exponential_pdp
from channels.physical_ofdm import active_grid_masks
from channels.temporal_multipath import correlated_tap_trajectory, jakes_slot_correlation
from models.adaptive_latent_refiner import (
    MoEAdaptiveLatentRefiner,
    no_jammer_identity_regularization,
    save_adaptive_latent_refiner_checkpoint,
)
from models.jammer_estimator import (
    JammerEstimator,
    jammer_estimation_loss,
    save_jammer_estimator_checkpoint,
)
from models.observable_channel_state import OBSERVABLE_RECEIVER_STATE_FEATURES
from speech_jscc.config import resolve_device
from speech_jscc.experiment import build_components
from speech_jscc.training.channel_free_feasibility import (
    decode_frozen_representation_with_gradient,
    multi_resolution_stft_loss,
)
from speech_jscc.training.channel_free_revalidation import per_layer_nmse
from speech_jscc.training.r4_waveform_finetune import (
    R4ForwardCondition,
    R4WaveformForward,
    freeze_codec_for_input_gradient,
)
from speech_jscc.training.si_sdr_loss import negative_si_sdr_loss, si_sdr
from train_channel_free_conv_conformer import fixed_paths, load_batch, sha256


TRAINER_JAMMER_TYPE_CLASSES: tuple[str, ...] = (
    "no_jammer", "broadband_awgn", "subband", "burst", "tone",
)
JAMMER_TYPE_TO_INDEX = {name: index for index, name in enumerate(TRAINER_JAMMER_TYPE_CLASSES)}
JAMMER_ALIASES = {"narrowband": "subband"}


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    start: int
    stop: int
    estimator_lr: float
    refiner_lr: float

    def contains(self, step: int) -> bool:
        return self.start <= step < self.stop


def canonical_jammer_type(name: str) -> str:
    canonical = JAMMER_ALIASES.get(name, name)
    if canonical not in JAMMER_TYPE_TO_INDEX:
        raise ValueError(f"unsupported R4 jammer type: {name}")
    return canonical


def build_phase_schedule(config: dict) -> tuple[PhaseSpec, ...]:
    names_and_steps = (
        ("estimator_pretrain", int(config["estimator_pretrain_steps"])),
        ("oracle_mask_refiner", int(config["oracle_mask_refiner_steps"])),
        ("learned_mask_moe", int(config["learned_mask_moe_steps"])),
    )
    start = 0
    result = []
    for name, steps in names_and_steps:
        if steps < 0:
            raise ValueError(f"{name} steps must be nonnegative")
        result.append(
            PhaseSpec(name, start, start + steps, float(config["estimator_lr"]), float(config["refiner_lr"]))
        )
        start += steps
    if start <= 0:
        raise ValueError("at least one training phase must contain steps")
    return tuple(result)


def phase_for_step(phases: tuple[PhaseSpec, ...], step: int) -> PhaseSpec:
    for phase in phases:
        if phase.contains(step):
            return phase
    raise ValueError(f"step {step} is outside configured phases")


def sample_jammer_condition(config: dict, *, step: int, seed: int) -> dict[str, float | str | None]:
    names = [canonical_jammer_type(str(name)) for name in config["jammer_types"]]
    probabilities = [float(config["jammer_probabilities"][name]) for name in config["jammer_types"]]
    if len(names) != len(set(names)):
        raise ValueError("jammer_types contains duplicate canonical classes")
    if any(value < 0 for value in probabilities) or abs(sum(probabilities) - 1.0) > 1e-6:
        raise ValueError("jammer probabilities must be nonnegative and sum to one")
    rng = random.Random((int(seed) << 20) ^ int(step))
    jammer_type = rng.choices(names, weights=probabilities, k=1)[0]
    snr_db = float(rng.choice(list(config["snr_db_choices"])))
    jsr_db = None if jammer_type == "no_jammer" else float(rng.choice(list(config["jsr_db_choices"])))
    return {"jammer_type": jammer_type, "snr_db": snr_db, "jsr_db": jsr_db}


def build_refiner_optimizer(
    estimator: JammerEstimator,
    refiner: MoEAdaptiveLatentRefiner,
    model: nn.Module,
    config: dict,
    phase: PhaseSpec,
) -> torch.optim.Optimizer:
    parameters: list[dict] = []
    if phase.name in {"estimator_pretrain", "learned_mask_moe"}:
        parameters.append({"params": list(estimator.parameters()), "lr": phase.estimator_lr})
    if phase.name in {"oracle_mask_refiner", "learned_mask_moe"}:
        parameters.append({"params": list(refiner.parameters()), "lr": phase.refiner_lr})
    if bool(config.get("unfreeze_jscc_decoder", False)) and phase.name != "estimator_pretrain":
        parameters.append({"params": list(model.decoder.parameters()), "lr": phase.refiner_lr})
    if not parameters:
        raise ValueError(f"phase {phase.name} has no trainable parameters")
    return torch.optim.AdamW(parameters, weight_decay=float(config.get("weight_decay", 0.0)))


__all__ = [
    "JAMMER_ALIASES",
    "JAMMER_TYPE_TO_INDEX",
    "TRAINER_JAMMER_TYPE_CLASSES",
    "PhaseSpec",
    "build_phase_schedule",
    "build_refiner_optimizer",
    "canonical_jammer_type",
    "phase_for_step",
    "sample_jammer_condition",
]


ROOT = Path(__file__).resolve().parents[3]


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _weights(config: dict) -> dict[str, float]:
    loss = config["loss"]
    return {
        "type": float(loss["lambda_type"]),
        "mask": float(loss["lambda_mask"]),
        "dice": float(loss["lambda_dice"]),
        "latent": float(loss["lambda_latent"]),
        "stft": float(loss["lambda_stft"]),
        "si_sdr": float(loss["lambda_si_sdr"]),
        "identity": float(loss["lambda_identity"]),
    }


def _one_hot_labels(batch: int, jammer_type: str, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    index = JAMMER_TYPE_TO_INDEX[jammer_type]
    return torch.nn.functional.one_hot(
        torch.full((batch,), index, device=device, dtype=torch.long),
        num_classes=len(TRAINER_JAMMER_TYPE_CLASSES),
    ).to(dtype)


def _sample_paths(paths: list[Path], *, step: int, batch_size: int) -> list[Path]:
    if not paths:
        raise ValueError("training manifest is empty")
    return [paths[(step * batch_size + offset) % len(paths)] for offset in range(batch_size)]


def _build_engine_modules(codec: nn.Module, model: nn.Module, config: dict) -> tuple[R4WaveformForward, JammerEstimator, MoEAdaptiveLatentRefiner]:
    estimator = JammerEstimator(
        num_jammer_types=len(TRAINER_JAMMER_TYPE_CLASSES),
        hidden_dim=int(config["estimator_hidden_dim"]),
        jammer_type_classes=TRAINER_JAMMER_TYPE_CLASSES,
    )
    refiner = MoEAdaptiveLatentRefiner(
        representation_shape=tuple(codec.representation_shape),
        channel_state_dim=len(OBSERVABLE_RECEIVER_STATE_FEATURES),
        num_experts=len(TRAINER_JAMMER_TYPE_CLASSES),
        hidden_dim=int(config["refiner_hidden_dim"]),
        state_features=int(config["refiner_state_features"]),
    )
    device = next(model.parameters()).device
    estimator = estimator.to(device)
    refiner = refiner.to(device)
    return R4WaveformForward(codec, model, uep_profile_name=str(config["uep_profile"])), estimator, refiner


def _set_phase_trainability(
    phase: PhaseSpec,
    estimator: JammerEstimator,
    refiner: MoEAdaptiveLatentRefiner,
    model: nn.Module,
    config: dict,
) -> None:
    estimator.requires_grad_(phase.name in {"estimator_pretrain", "learned_mask_moe"})
    refiner.requires_grad_(phase.name in {"oracle_mask_refiner", "learned_mask_moe"})
    model.encoder.requires_grad_(False)
    model.decoder.requires_grad_(bool(config.get("unfreeze_jscc_decoder", False)) and phase.name != "estimator_pretrain")


def compute_refiner_losses(
    *,
    phase: PhaseSpec,
    estimator: JammerEstimator,
    refiner: MoEAdaptiveLatentRefiner,
    physical_result,
    raw_reconstruction: Tensor,
    target_representation: Tensor,
    waveform_target: Tensor,
    jammer_type: str,
    config: dict,
    codec: nn.Module,
) -> tuple[Tensor, dict[str, Tensor], Tensor, Tensor]:
    """Compute phase losses without passing labels to estimator inference."""
    masks = active_grid_masks(physical_result.allocation.profile, device=raw_reconstruction.device)
    estimate = estimator(
        physical_result.received_resources,
        physical_result.pilots,
        masks.pilot,
        physical_result.estimated_channel,
        physical_result.noise_variance,
    )
    batch = raw_reconstruction.shape[0]
    label_index = torch.full((batch,), JAMMER_TYPE_TO_INDEX[jammer_type], device=raw_reconstruction.device, dtype=torch.long)
    estimate_total, estimate_components = jammer_estimation_loss(
        estimate,
        jammer_type=label_index,
        jammer_mask=physical_result.jammer_mask,
        bce_weight=1.0,
        dice_weight=1.0,
        pos_weight=config["loss"].get("mask_pos_weight"),
    )
    del estimate_total
    zero = raw_reconstruction.new_zeros(())
    components = {
        "type": estimate_components["type_ce"],
        "mask": estimate_components["mask_bce"],
        "dice": estimate_components["mask_dice"],
        "latent": zero,
        "stft": zero,
        "si_sdr": zero,
        "identity": zero,
    }
    if phase.name == "estimator_pretrain":
        refined = raw_reconstruction
    elif phase.name == "oracle_mask_refiner":
        refined = refiner(
            raw_reconstruction,
            physical_result.decoder_state,
            physical_result.jammer_mask.to(dtype=raw_reconstruction.dtype),
            _one_hot_labels(batch, jammer_type, device=raw_reconstruction.device, dtype=raw_reconstruction.dtype),
        )
    else:
        refined = refiner(raw_reconstruction, physical_result.decoder_state, estimate.mask_prob, estimate.posterior)
    if phase.name != "estimator_pretrain":
        components["latent"] = per_layer_nmse(refined, target_representation).mean()
        weights = _weights(config)
        decoded = None
        if weights["stft"] or weights["si_sdr"]:
            decoded = decode_frozen_representation_with_gradient(codec, refined)
        if weights["stft"]:
            components["stft"] = multi_resolution_stft_loss(decoded, waveform_target, fft_sizes=tuple(config["loss"]["fft_sizes"]))
        if weights["si_sdr"]:
            components["si_sdr"] = negative_si_sdr_loss(decoded, waveform_target, clip_db=config["loss"].get("si_sdr_clip_db"))[0]
        if jammer_type == "no_jammer":
            components["identity"] = no_jammer_identity_regularization(
                refined, raw_reconstruction,
                _one_hot_labels(batch, "no_jammer", device=raw_reconstruction.device, dtype=raw_reconstruction.dtype),
            )
    total = sum(_weights(config)[name] * value for name, value in components.items())
    return total, components, refined, estimate.posterior


def _checkpoint_payload(
    *,
    estimator: JammerEstimator,
    refiner: MoEAdaptiveLatentRefiner,
    optimizer: torch.optim.Optimizer,
    source_checkpoint: Path,
    config: dict,
    global_step: int,
    phase: PhaseSpec,
    best_metrics: dict[str, float],
) -> dict:
    return {
        "diagnostic_type": "r4_jammer_refiner",
        "estimator": save_jammer_estimator_checkpoint(estimator),
        "adaptive_refiner": save_adaptive_latent_refiner_checkpoint(refiner),
        "optimizer": optimizer.state_dict(),
        "scheduler": None,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": sha256(source_checkpoint),
        "config": config,
        "global_step": int(global_step),
        "phase": phase.name,
        "best_metrics": best_metrics,
        "jammer_label_vocabulary": list(TRAINER_JAMMER_TYPE_CLASSES),
        "refiner_mode": "learned_posterior_moe_refiner",
    }


def _validation_rows(
    *,
    engine: R4WaveformForward,
    estimator: JammerEstimator,
    refiner: MoEAdaptiveLatentRefiner,
    codec: nn.Module,
    target: Tensor,
    waveform: Tensor,
    taps: Tensor,
    config: dict,
    seed: int,
) -> list[dict]:
    rows = []
    masks = active_grid_masks(engine.profile, device=target.device)
    with torch.no_grad():
        for index, jammer_type in enumerate(TRAINER_JAMMER_TYPE_CLASSES):
            jsr = None if jammer_type == "no_jammer" else float(config["jsr_db_choices"][index % len(config["jsr_db_choices"])])
            condition = R4ForwardCondition(
                snr_db=float(config["snr_db_choices"][index % len(config["snr_db_choices"])]) ,
                tti=0, tap_coefficients=taps, noise_seed=int(seed + index), fixed_mapping=True,
            )
            result = engine.forward(
                target, channel_condition=condition, jammer_type=jammer_type,
                jammer_jsr_db=jsr, jammer_seed=int(seed + 100 + index), training=False,
            )
            estimate = estimator(
                result.received_resources, result.pilots, masks.pilot,
                result.estimated_channel, result.noise_variance,
            )
            # Use the same post-decoder learned path as phase 3; labels are only metrics.
            refined = refiner(result.raw_reconstruction, result.decoder_state, estimate.mask_prob, estimate.posterior)
            refined_waveform = decode_frozen_representation_with_gradient(codec, refined)
            raw_waveform = decode_frozen_representation_with_gradient(codec, result.raw_reconstruction)
            refined_si_sdr = si_sdr(refined_waveform, waveform).mean()
            raw_si_sdr = si_sdr(raw_waveform, waveform).mean()
            predicted = estimate.posterior.argmax(dim=-1)
            target_mask = result.jammer_mask.bool()
            predicted_mask = estimate.mask_prob >= 0.5
            intersection = (predicted_mask & target_mask).sum().float()
            union = (predicted_mask | target_mask).sum().float().clamp_min(1.0)
            tp = intersection
            f1 = (2 * tp / (predicted_mask.sum() + target_mask.sum()).float().clamp_min(1.0))
            rows.append({
                "jammer_type": jammer_type,
                "type_accuracy": float((predicted == JAMMER_TYPE_TO_INDEX[jammer_type]).float().mean()),
                "mask_iou": float(intersection / union),
                "mask_f1": float(f1),
                "mask_bce": float(torch.nn.functional.binary_cross_entropy_with_logits(estimate.mask_logits, target_mask.to(estimate.mask_logits.dtype))),
                "latent_nmse": float(per_layer_nmse(refined, target).mean()),
                "si_sdr_db": float(refined_si_sdr),
                "raw_si_sdr_db": float(raw_si_sdr),
                "si_sdr_delta_vs_raw_db": float(refined_si_sdr - raw_si_sdr),
                "no_jammer_degradation": float(per_layer_nmse(refined, target).mean() - per_layer_nmse(result.raw_reconstruction, target).mean()) if jammer_type == "no_jammer" else None,
            })
    return rows


def _write_validation_csv(path: Path, rows: list[dict]) -> None:
    keys = ["global_step", "phase", "jammer_type", "type_accuracy", "mask_iou", "mask_f1", "mask_bce", "latent_nmse", "si_sdr_db", "raw_si_sdr_db", "si_sdr_delta_vs_raw_db", "no_jammer_degradation"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_r4_jammer_refiner.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--allow-long-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(json.dumps(__import__("yaml").safe_load(config_path.read_text())))
    if args.seed is not None:
        config["seed"] = int(args.seed)
    steps = int(args.max_steps or config["max_steps"])
    phases = build_phase_schedule(config)
    if steps > phases[-1].stop:
        raise SystemExit("max steps exceeds configured phase schedule")
    if steps > 5 and not args.allow_long_run:
        raise SystemExit("jammer-refiner training longer than five steps requires --allow-long-run")
    source_checkpoint = Path(args.checkpoint or config["source_checkpoint"])
    if not source_checkpoint.is_absolute():
        source_checkpoint = ROOT / source_checkpoint
    if not source_checkpoint.is_file():
        raise SystemExit(f"source checkpoint does not exist: {source_checkpoint}")
    output = Path(args.output_dir or config["output_root"])
    if not output.is_absolute():
        output = ROOT / output
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"refusing existing output directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=False)
    device_name = args.device or config["device"]
    device = resolve_device(device_name)
    torch.manual_seed(int(config["seed"]))
    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    model_config = copy.deepcopy(payload["config"])
    for key in ("config_path", "checkpoint_path"):
        if key in model_config["codec"] and not Path(model_config["codec"][key]).is_absolute():
            model_config["codec"][key] = str(ROOT / model_config["codec"][key])
    model_config["device"] = str(device_name)
    codec, model = build_components(model_config, device)
    model.load_state_dict(payload["model"], strict=True)
    freeze_codec_for_input_gradient(codec)
    if any(parameter.requires_grad for parameter in codec.parameters()):
        raise RuntimeError("SpeechTokenizer codec must be frozen")
    engine, estimator, refiner = _build_engine_modules(codec, model, config)
    train_paths, validation_paths = fixed_paths(model_config, int(model_config["seed"]))
    batch_size = int(config["batch_size"])
    physical_config = __import__("yaml").safe_load((ROOT / "configs/ofdm_nr_like_r4.yaml").read_text())
    pdp = exponential_pdp(physical_config["channel"]["num_taps"], physical_config["channel"]["pdp_decay"])
    rho = jakes_slot_correlation(physical_config["physical"]["user_speed_mps"], physical_config["physical"]["carrier_frequency_hz"], engine.profile.tti_duration_s)
    trajectory = correlated_tap_trajectory(slots=max(steps, 1), batch_size=batch_size, pdp=pdp, rho=rho, seed=int(config["seed"]))
    (output / "resolved_config.yaml").write_text(__import__("yaml").safe_dump(config, sort_keys=False))
    _json(output / "source_checkpoint.json", {"path": str(source_checkpoint), "sha256": sha256(source_checkpoint), "resume_mode": "weights_only", "restored": ["codec", "jscc_model"], "reset": ["optimizer", "scheduler", "rng"]})
    _json(output / "jammer_label_vocabulary.json", {"classes": list(TRAINER_JAMMER_TYPE_CLASSES), "aliases": JAMMER_ALIASES})
    (output / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n")
    _json(output / "environment.json", {"device": str(device), "torch": torch.__version__})
    best = {"validation_si_sdr": float("-inf"), "validation_latent": float("inf")}
    optimizer = None
    active_phase = None
    report = None
    train_rows: list[dict] = []
    validation_rows: list[dict] = []
    for step in range(steps):
        phase = phase_for_step(phases, step)
        if active_phase != phase.name:
            active_phase = phase.name
            _set_phase_trainability(phase, estimator, refiner, model, config)
            optimizer = build_refiner_optimizer(estimator, refiner, model, config, phase)
        sampled = sample_jammer_condition(config, step=step, seed=int(config["seed"]))
        waveform = load_batch(_sample_paths(train_paths, step=step, batch_size=batch_size), model_config, device)
        with torch.no_grad():
            target = codec.encode_waveform(waveform)
        condition = R4ForwardCondition(
            snr_db=float(sampled["snr_db"]), tti=step,
            tap_coefficients=trajectory[step].to(device), noise_seed=int(config["seed"]) + step,
            fixed_mapping=step == 0,
        )
        result = engine.forward(target, channel_condition=condition, delayed_csi=report, training=True, jammer_type=str(sampled["jammer_type"]), jammer_jsr_db=sampled["jsr_db"], jammer_seed=int(config["seed"]) + 100000 + step, jammer_subband_fraction=float(config["jammer"]["subband_fraction"]), jammer_burst_fraction=float(config["jammer"]["burst_fraction"]), jammer_tone_count=int(config["jammer"]["tone_count"]))
        report = result.next_delayed_csi
        total, components, refined, posterior = compute_refiner_losses(phase=phase, estimator=estimator, refiner=refiner, physical_result=result, raw_reconstruction=result.raw_reconstruction, target_representation=target, waveform_target=waveform, jammer_type=str(sampled["jammer_type"]), config=config, codec=codec)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_([parameter for group in optimizer.param_groups for parameter in group["params"]], float(config["gradient_clip_norm"]))
        optimizer.step()
        finite = bool(torch.isfinite(total) and all(torch.isfinite(value) for value in components.values()))
        if not finite:
            raise FloatingPointError(f"nonfinite jammer-refiner loss at step {step}")
        row = {"global_step": step + 1, "phase": phase.name, **sampled, "total_loss": float(total.detach()), "loss_components": {key: float(value.detach()) for key, value in components.items()}, "posterior_entropy": float(-(posterior * posterior.clamp_min(1e-8).log()).sum(-1).mean().detach()), "codec_parameters_with_grad": sum(parameter.grad is not None for parameter in codec.parameters()), "finite": finite}
        train_rows.append(row)
        validate_now = (step + 1) % int(config["validation_interval"]) == 0 or step + 1 == steps
        if validate_now:
            validation_count = min(int(config.get("validation_examples", 1)), target.shape[0])
            validation_target = target[:validation_count]
            validation_waveform = waveform[:validation_count]
            validation_taps = trajectory[step][:validation_count].to(device)
            rows = _validation_rows(engine=engine, estimator=estimator, refiner=refiner, codec=codec, target=validation_target, waveform=validation_waveform, taps=validation_taps, config=config, seed=int(config["seed"]) + step * 10)
            for item in rows:
                item.update({"global_step": step + 1, "phase": phase.name})
            validation_rows.extend(rows)
            mean_latent = sum(item["latent_nmse"] for item in rows) / len(rows)
            mean_si_sdr = sum(item["si_sdr_db"] for item in rows) / len(rows)
            if mean_latent < best["validation_latent"]:
                best["validation_latent"] = mean_latent
                torch.save(_checkpoint_payload(estimator=estimator, refiner=refiner, optimizer=optimizer, source_checkpoint=source_checkpoint, config=config, global_step=step + 1, phase=phase, best_metrics=best), output / "best_validation_latent.pt")
            if mean_si_sdr > best["validation_si_sdr"]:
                best["validation_si_sdr"] = mean_si_sdr
                torch.save(_checkpoint_payload(estimator=estimator, refiner=refiner, optimizer=optimizer, source_checkpoint=source_checkpoint, config=config, global_step=step + 1, phase=phase, best_metrics=best), output / "best_validation_si_sdr.pt")
        if (step + 1) % int(config["checkpoint_interval"]) == 0 or step + 1 == steps:
            torch.save(_checkpoint_payload(estimator=estimator, refiner=refiner, optimizer=optimizer, source_checkpoint=source_checkpoint, config=config, global_step=step + 1, phase=phase, best_metrics=best), output / "last.pt")
    _jsonl(output / "training_metrics.jsonl", train_rows)
    _jsonl(output / "validation_metrics.jsonl", validation_rows)
    _write_validation_csv(output / "validation_metrics.csv", validation_rows)
    _json(output / "training_completion.json", {"status": "completed", "steps": steps, "all_losses_finite": all(row["finite"] for row in train_rows), "codec_frozen": not any(parameter.requires_grad for parameter in codec.parameters()), "best_metrics": best})


if __name__ == "__main__":
    main()
