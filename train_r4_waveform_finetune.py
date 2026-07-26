from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from channels.multipath import exponential_pdp
from channels.temporal_multipath import correlated_tap_trajectory, jakes_slot_correlation
from speech_jscc.config import resolve_device
from speech_jscc.experiment import build_components
from speech_jscc.training.channel_free_feasibility import (
    decode_frozen_representation_with_gradient,
)
from speech_jscc.training.channel_free_revalidation import component_gradient_norms
from speech_jscc.training.channel_free_revalidation import per_layer_nmse, summed_latent_statistics
from speech_jscc.training.r4_waveform_finetune import (
    CheckpointSelector,
    R4Curriculum,
    R4ForwardCondition,
    R4WaveformForward,
    clean_gate,
    component_gradient_norm_for_module,
    freeze_codec_for_input_gradient,
    r4_training_objective,
    validate_initial_checkpoint_metadata,
)
from src.evaluation.waveform_metrics import waveform_metrics
from train_channel_free_conv_conformer import fixed_paths, load_batch, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_r4_waveform_finetune.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--allow-long-run", action="store_true")
    parser.add_argument("--verify-waveform-gradient", action="store_true")
    return parser.parse_args()


def effective_training_steps(
    requested_steps: int, smoke_steps: int | None, *, allow_long_run: bool
) -> int:
    if smoke_steps is not None:
        if not 1 <= int(smoke_steps) <= 5:
            raise ValueError("smoke steps must be in [1,5]")
        return int(smoke_steps)
    if int(requested_steps) > 5 and not allow_long_run:
        raise ValueError("long R4 waveform fine-tuning requires --allow-long-run")
    return int(requested_steps)


def restore_training_state(
    path: Path,
    *,
    model,
    optimizer,
    selector: CheckpointSelector,
    validation_manifest: dict,
    device: torch.device,
) -> dict:
    resume = torch.load(path, map_location=device, weights_only=False)
    if resume.get("diagnostic_type") != "r4_waveform_finetune":
        raise ValueError("resume checkpoint is not R4 waveform fine-tuning")
    if resume["validation_manifest"] != validation_manifest:
        raise ValueError("validation manifest/seeds changed across resume")
    model.load_state_dict(resume["model"], strict=True)
    optimizer.load_state_dict(resume["optimizer"])
    selector.load_state_dict(resume["selector"])
    torch.set_rng_state(resume["torch_rng_state"].cpu())
    random.setstate(resume["python_rng_state"])
    return {
        "global_step": int(resume["global_step"]),
        "curriculum_stage": str(resume["curriculum_stage"]),
        "delayed_csi": resume["delayed_csi"],
        "selector": selector,
    }


def _git_metadata() -> dict:
    def run(*args):
        return subprocess.run(args, text=True, capture_output=True, check=False).stdout.strip()
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def _loss_weights(step: int, config: dict) -> dict[str, float]:
    loss = config["loss"]
    if step < 4000:
        ramp = 0.0
    elif step < 12000:
        ramp = (step - 4000 + 1) / 8000
    else:
        ramp = 1.0
    return {
        "latent": float(loss["latent_mse_weight"]),
        "stft": float(loss["multires_stft_weight"]) * ramp,
        "waveform": float(loss["waveform_l1_weight"]) * ramp,
        "channel_free": float(loss["channel_free_weight"]),
    }


def _set_learning_rate(optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(value)


def _module_gradient_norm(module: torch.nn.Module) -> float:
    squared = sum(
        parameter.grad.detach().float().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    )
    return math.sqrt(float(squared))


def _pure_neural(codec, model, target):
    state = target.new_zeros((target.shape[0], model.encoder.channel_state_dim))
    symbols = model.encoder(target, state)
    reconstruction = model.decoder(symbols, state)
    waveform = decode_frozen_representation_with_gradient(codec, reconstruction)
    return reconstruction, waveform


def _save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    step: int,
    curriculum_stage: str,
    selector: CheckpointSelector,
    validation_manifest: dict,
    delayed_csi,
    model_config: dict,
    fine_tune_config: dict,
    metrics: dict,
    source: Path,
) -> None:
    torch.save(
        {
            "diagnostic_type": "r4_waveform_finetune",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": int(step),
            "curriculum_stage": curriculum_stage,
            "selector": selector.state_dict(),
            "validation_manifest": validation_manifest,
            "delayed_csi": delayed_csi,
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
            "config": model_config,
            "fine_tune_config": fine_tune_config,
            "validation": metrics,
            "source_checkpoint": str(source),
            "source_checkpoint_sha256": sha256(source),
        },
        path,
    )


def _validation_stub(selector: CheckpointSelector) -> tuple[dict, object]:
    # A smoke run deliberately does not perform expensive held-out waveform validation.
    metrics = {
        "5db_delta_si_sdr_vs_initial_r4": -1.0e9,
        "validation_average_delta_si_sdr_vs_initial_r4": -1.0e9,
    }
    thresholds = {
        "min_pure_neural_si_sdr_delta_db": -0.1,
        "min_noiseless_r4_si_sdr_delta_db": -0.1,
        "max_clean_stft_increase": 0.002,
        "max_clean_latent_mse_ratio": 1.05,
    }
    values = {
        "pure_neural_si_sdr_delta_db": -1.0e9,
        "noiseless_r4_si_sdr_delta_db": -1.0e9,
        "clean_stft_increase": 1.0e9,
        "clean_latent_mse_ratio": 1.0e9,
    }
    return metrics, clean_gate(values, thresholds)


def _average(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _validate(
    *,
    codec,
    model,
    initial_model,
    paths,
    checkpoint_config,
    specification,
    device,
    seeds: list[int],
    count: int,
) -> tuple[dict, object]:
    model.eval()
    initial_model.eval()
    current_engine = R4WaveformForward(
        codec, model,
        estimator_ridge_lambda=specification["physical"]["estimator_ridge_lambda"],
        epsilon=specification["physical"]["epsilon"],
    )
    initial_engine = R4WaveformForward(
        codec, initial_model,
        estimator_ridge_lambda=specification["physical"]["estimator_ridge_lambda"],
        epsilon=specification["physical"]["epsilon"],
    )
    physical = yaml.safe_load(Path(specification["physical_profile_config"]).read_text())
    pdp = exponential_pdp(physical["channel"]["num_taps"], physical["channel"]["pdp_decay"])
    rho = jakes_slot_correlation(
        physical["physical"]["user_speed_mps"],
        physical["physical"]["carrier_frequency_hz"],
        current_engine.profile.tti_duration_s,
    )
    rows: list[dict] = []
    pure_rows: list[dict] = []
    noiseless_rows: list[dict] = []
    selected_paths = list(paths[:count])
    with torch.no_grad():
        for snr in map(float, specification["validation"]["snr_db"]):
            for validation_seed in seeds:
                trajectory = correlated_tap_trajectory(
                    slots=len(selected_paths), batch_size=1, pdp=pdp, rho=rho,
                    seed=validation_seed + round(snr * 100),
                )
                report = initial_report = None
                for tti, path in enumerate(selected_paths):
                    waveform = load_batch([path], checkpoint_config, device)
                    target = codec.encode_waveform(waveform)
                    condition = R4ForwardCondition(
                        snr_db=snr, tti=tti,
                        tap_coefficients=trajectory[tti].to(device),
                        noise_seed=validation_seed + tti + round(snr * 1000),
                    )
                    current = current_engine.forward(
                        target, waveform, condition, report, training=False
                    )
                    initial = initial_engine.forward(
                        target, waveform, condition, initial_report, training=False
                    )
                    report, initial_report = current.next_delayed_csi, initial.next_delayed_csi
                    clean_waveform = codec.decode_representation(target)
                    clean_metric = waveform_metrics(
                        waveform, clean_waveform, int(checkpoint_config["codec"]["sample_rate"])
                    )
                    current_metric = waveform_metrics(
                        waveform, current.decoded_waveform,
                        int(checkpoint_config["codec"]["sample_rate"]),
                    )
                    initial_metric = waveform_metrics(
                        waveform, initial.decoded_waveform,
                        int(checkpoint_config["codec"]["sample_rate"]),
                    )
                    layer = per_layer_nmse(current.reconstruction, target)
                    summed = summed_latent_statistics(current.reconstruction, target)
                    rows.append({
                        "snr_db": snr, "seed": validation_seed, "path": str(path),
                        "si_sdr_absolute": current_metric["si_sdr_db"],
                        "delta_si_sdr_vs_clean_codec": current_metric["si_sdr_db"] - clean_metric["si_sdr_db"],
                        "delta_si_sdr_vs_initial_r4": current_metric["si_sdr_db"] - initial_metric["si_sdr_db"],
                        "waveform_snr_db": current_metric["waveform_snr_db"],
                        "delta_waveform_snr_vs_clean_codec": current_metric["waveform_snr_db"] - clean_metric["waveform_snr_db"],
                        "delta_waveform_snr_vs_initial_r4": current_metric["waveform_snr_db"] - initial_metric["waveform_snr_db"],
                        "stft_l1": current_metric["stft_l1"],
                        "stft_ratio_vs_clean_codec": current_metric["stft_l1"] / max(clean_metric["stft_l1"], 1e-12),
                        "stft_increase_vs_initial_r4": current_metric["stft_l1"] - initial_metric["stft_l1"],
                        "aggregate_layer_nmse": float(layer.mean()),
                        "per_layer_nmse": [float(value) for value in layer],
                        "summed_latent_nmse": float(summed["nmse"]),
                        "csi_nmse": float(current.csi_nmse),
                        "pilot_evm": float(current.pilot_evm),
                        "effective_sinr_db": float(10 * torch.log10(current.effective_sinr)),
                    })
                    if snr == 10 and validation_seed == seeds[0]:
                        current_pure, current_pure_wave = _pure_neural(codec, model, target)
                        initial_pure, initial_pure_wave = _pure_neural(codec, initial_model, target)
                        cp = waveform_metrics(waveform, current_pure_wave, int(checkpoint_config["codec"]["sample_rate"]))
                        ip = waveform_metrics(waveform, initial_pure_wave, int(checkpoint_config["codec"]["sample_rate"]))
                        pure_rows.append({
                            "si_sdr_delta": cp["si_sdr_db"] - ip["si_sdr_db"],
                            "stft_increase": cp["stft_l1"] - ip["stft_l1"],
                            "latent_ratio": float(
                                per_layer_nmse(current_pure, target).mean()
                                / per_layer_nmse(initial_pure, target).mean().clamp_min(1e-12)
                            ),
                        })
                        noiseless = R4ForwardCondition(
                            snr_db=100, tti=0,
                            tap_coefficients=trajectory[tti].to(device),
                            noise_seed=validation_seed + tti,
                            noiseless=True, perfect_csi=True, fixed_mapping=True,
                        )
                        cn = current_engine.forward(target, waveform, noiseless, None, training=False)
                        inn = initial_engine.forward(target, waveform, noiseless, None, training=False)
                        cnm = waveform_metrics(waveform, cn.decoded_waveform, int(checkpoint_config["codec"]["sample_rate"]))
                        inm = waveform_metrics(waveform, inn.decoded_waveform, int(checkpoint_config["codec"]["sample_rate"]))
                        noiseless_rows.append({"si_sdr_delta": cnm["si_sdr_db"] - inm["si_sdr_db"]})
    by_snr = {}
    for snr in map(float, specification["validation"]["snr_db"]):
        members = [row for row in rows if row["snr_db"] == snr]
        by_snr[str(snr)] = {
            key: _average(members, key) for key in (
                "si_sdr_absolute", "delta_si_sdr_vs_clean_codec",
                "delta_si_sdr_vs_initial_r4", "waveform_snr_db",
                "delta_waveform_snr_vs_clean_codec",
                "delta_waveform_snr_vs_initial_r4", "stft_l1",
                "stft_increase_vs_initial_r4",
                "stft_ratio_vs_clean_codec", "aggregate_layer_nmse",
                "summed_latent_nmse", "csi_nmse", "pilot_evm", "effective_sinr_db",
            )
        }
        by_snr[str(snr)]["per_layer_nmse"] = [
            sum(row["per_layer_nmse"][layer] for row in members) / len(members)
            for layer in range(8)
        ]
    clean_values = {
        "pure_neural_si_sdr_delta_db": _average(pure_rows, "si_sdr_delta"),
        "noiseless_r4_si_sdr_delta_db": _average(noiseless_rows, "si_sdr_delta"),
        "clean_stft_increase": _average(pure_rows, "stft_increase"),
        "clean_latent_mse_ratio": _average(pure_rows, "latent_ratio"),
    }
    gate = clean_gate(clean_values, specification["selection"]["clean_constraints"])
    selection = {
        "5db_delta_si_sdr_vs_initial_r4": by_snr["5.0"]["delta_si_sdr_vs_initial_r4"],
        "validation_average_delta_si_sdr_vs_initial_r4": sum(
            by_snr[str(snr)]["delta_si_sdr_vs_initial_r4"] for snr in (5.0, 10.0, 15.0)
        ) / 3,
    }
    return {
        **selection,
        "by_snr": by_snr,
        "clean_regression": clean_values,
        "clean_gate_pass": gate.passed,
        "clean_gate_margins": gate.margins,
        "clean_gate_minimum_margin": gate.minimum_margin,
        "rows": rows,
    }, gate


def main() -> None:
    args = parse_args()
    specification = yaml.safe_load(Path(args.config).read_text())
    total_steps = int(args.steps or specification["training"]["total_steps"])
    batch_size = int(args.batch_size or specification["training"]["batch_size"])
    output = Path(args.output_dir or specification["output_dir"])
    smoke_steps = args.smoke_steps
    if args.dry_run:
        print(json.dumps({
            "initial_checkpoint": specification["initial_checkpoint"],
            "output_dir": str(output),
            "total_steps": total_steps,
            "batch_size": batch_size,
            "curriculum": "4000/8000/8000",
            "physical_path": "R4 time-domain OFDM + LS CSI + repetition3 coherent MRC",
            "full_training": False,
        }, indent=2))
        return
    try:
        total_steps = effective_training_steps(
            total_steps, smoke_steps, allow_long_run=args.allow_long_run
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    source = Path(specification["initial_checkpoint"])
    payload = torch.load(source, map_location="cpu", weights_only=False)
    checkpoint_config = payload["config"]
    validate_initial_checkpoint_metadata(
        checkpoint_config, specification.get("compatibility")
    )
    checkpoint_config["device"] = args.device
    device = resolve_device(args.device)
    codec, model = build_components(checkpoint_config, device)
    model.load_state_dict(payload["model"], strict=True)
    _, initial_model = build_components(checkpoint_config, device)
    initial_model.load_state_dict(payload["model"], strict=True)
    initial_model.eval().requires_grad_(False)
    freeze_codec_for_input_gradient(codec)
    trainable = list(model.encoder.parameters()) + list(model.decoder.parameters())
    optimizer = torch.optim.Adam(
        trainable,
        lr=float(specification["training"]["stage_a_learning_rate"]),
        weight_decay=float(checkpoint_config["train"].get("weight_decay", 0)),
    )
    curriculum = R4Curriculum(
        stage_a_learning_rate=specification["training"]["stage_a_learning_rate"],
        stage_b_learning_rate=specification["training"]["stage_b_learning_rate"],
        stage_c_learning_rate=specification["training"]["stage_c_learning_rate"],
        stage_c_probabilities={
            float(key): float(value)
            for key, value in specification["curriculum"]["stage_c"]["snr_probabilities"].items()
        },
    )
    engine = R4WaveformForward(
        codec,
        model,
        estimator_ridge_lambda=specification["physical"]["estimator_ridge_lambda"],
        epsilon=specification["physical"]["epsilon"],
    )
    if output.exists() and not args.resume:
        if not args.overwrite:
            raise SystemExit(f"refusing existing output directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    resolved = dict(specification)
    resolved["resolved_device"] = str(device)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n")
    (output / "environment.json").write_text(json.dumps({
        "python": platform.python_version(),
        "torch": torch.__version__,
        "git": _git_metadata(),
        "source_checkpoint_sha256": sha256(source),
    }, indent=2))
    train_paths, validation_paths = fixed_paths(checkpoint_config, int(checkpoint_config["seed"]))
    validation_manifest = {
        "paths": [str(path) for path in validation_paths[: specification["validation"]["full_utterances"]]],
        "light_seeds": specification["validation"]["light_seeds"],
        "full_seeds": specification["validation"]["full_seeds"],
    }
    (output / "validation_manifest.json").write_text(json.dumps(validation_manifest, indent=2))
    selector = CheckpointSelector(output)
    start = 0
    delayed_csi = None
    if args.resume:
        try:
            restored = restore_training_state(
                Path(args.resume), model=model, optimizer=optimizer,
                selector=selector, validation_manifest=validation_manifest,
                device=device,
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        start = restored["global_step"]
        delayed_csi = restored["delayed_csi"]
    physical = yaml.safe_load(Path(specification["physical_profile_config"]).read_text())
    pdp = exponential_pdp(physical["channel"]["num_taps"], physical["channel"]["pdp_decay"])
    rho = jakes_slot_correlation(
        physical["physical"]["user_speed_mps"],
        physical["physical"]["carrier_frequency_hz"],
        engine.profile.tti_duration_s,
    )
    trajectory = correlated_tap_trajectory(
        slots=max(total_steps, start + 1),
        batch_size=1,
        pdp=pdp,
        rho=rho,
        seed=int(specification["seed"]) + 700,
    )
    order: list[Path] = []
    epochs = (total_steps * batch_size + len(train_paths) - 1) // len(train_paths) + 1
    for epoch in range(epochs):
        items = list(train_paths)
        random.Random(int(specification["seed"]) + epoch).shuffle(items)
        order.extend(items)
    anomaly = bool(specification["training"]["anomaly_detection"])
    torch.autograd.set_detect_anomaly(anomaly)
    metrics_path = output / "metrics.jsonl"
    with metrics_path.open("a" if args.resume else "w") as log:
        for step in range(start, total_steps):
            stage = curriculum.stage(step)
            _set_learning_rate(optimizer, stage.learning_rate)
            snr = curriculum.sample_snr(step, seed=int(specification["seed"]))
            paths = order[step * batch_size : (step + 1) * batch_size]
            waveform = load_batch(paths, checkpoint_config, device)
            with torch.no_grad():
                target = codec.encode_waveform(waveform)
            model.train()
            condition = R4ForwardCondition(
                snr_db=snr,
                tti=step,
                tap_coefficients=trajectory[step].to(device),
                noise_seed=int(specification["seed"]) + 10_000 + step,
                fixed_mapping=stage.mapping_mode == "bootstrap_fixed",
            )
            result = engine.forward(
                target, waveform, condition, delayed_csi, training=True
            )
            delayed_csi = result.next_delayed_csi
            channel_free_reconstruction, _ = _pure_neural(codec, model, target)
            weights = _loss_weights(step, specification)
            if args.verify_waveform_gradient:
                weights["stft"] = float(specification["loss"]["multires_stft_weight"])
                weights["waveform"] = float(specification["loss"]["waveform_l1_weight"])
            total, components = r4_training_objective(
                result.reconstruction,
                target,
                waveform,
                lambda layers: decode_frozen_representation_with_gradient(codec, layers),
                weights=weights,
                channel_free_reconstruction=channel_free_reconstruction,
                fft_sizes=tuple(specification["loss"]["fft_sizes"]),
            )
            component_norms = component_gradient_norms(components, weights, trainable)
            waveform_encoder_component_norm = component_gradient_norm_for_module(
                components["waveform"], weights["waveform"], model.encoder
            )
            waveform_decoder_component_norm = component_gradient_norm_for_module(
                components["waveform"], weights["waveform"], model.decoder
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            encoder_norm = _module_gradient_norm(model.encoder)
            decoder_norm = _module_gradient_norm(model.decoder)
            if not all(math.isfinite(value) for value in (
                float(total.detach()), encoder_norm, decoder_norm,
                float(result.csi_nmse.detach()), float(result.pilot_evm.detach()),
            )):
                raise FloatingPointError(
                    f"nonfinite R4 training step={step} snr={snr} "
                    f"seed={condition.noise_seed}"
                )
            torch.nn.utils.clip_grad_norm_(
                trainable, float(specification["training"]["gradient_clip_norm"])
            )
            optimizer.step()
            if args.verify_waveform_gradient and not (
                waveform_encoder_component_norm > 0
                and waveform_decoder_component_norm > 0
                and all(parameter.grad is None for parameter in codec.parameters())
            ):
                raise RuntimeError(
                    "waveform-gradient verification failed: expected nonzero JSCC "
                    "encoder/decoder gradients and absent codec gradients"
                )
            row = {
                "global_step": step + 1,
                "curriculum_stage": stage.name,
                "learning_rate": stage.learning_rate,
                "sampled_nominal_snr_db": snr,
                "realized_effective_sinr_db": float(
                    10 * torch.log10(result.effective_sinr).detach()
                ),
                "total_loss": float(total.detach()),
                "loss_components": {
                    key: float(value.detach()) for key, value in components.items()
                },
                "loss_weights": weights,
                "component_gradient_norms": component_norms,
                "encoder_gradient_norm": encoder_norm,
                "decoder_gradient_norm": decoder_norm,
                "waveform_encoder_gradient_norm": waveform_encoder_component_norm,
                "waveform_decoder_gradient_norm": waveform_decoder_component_norm,
                "codec_parameters_with_grad": sum(
                    parameter.grad is not None for parameter in codec.parameters()
                ),
                "csi_nmse": float(result.csi_nmse.detach()),
                "pilot_evm": float(result.pilot_evm.detach()),
                "transmit_power": float(result.transmit_power),
                "mrc_output_power": float(result.mrc_output_power.detach()),
            }
            light_due = (step + 1) % int(specification["validation"]["light_every"]) == 0
            full_due = (step + 1) % int(specification["validation"]["full_every"]) == 0
            if smoke_steps is not None:
                validation_metrics, gate = _validation_stub(selector)
            elif light_due or full_due or step + 1 == total_steps:
                validation_metrics, gate = _validate(
                    codec=codec, model=model, initial_model=initial_model,
                    paths=validation_paths, checkpoint_config=checkpoint_config,
                    specification=specification, device=device,
                    seeds=list(
                        specification["validation"]["full_seeds"]
                        if full_due else specification["validation"]["light_seeds"]
                    ),
                    count=int(
                        specification["validation"]["full_utterances"]
                        if full_due else specification["validation"]["light_utterances"]
                    ),
                )
                (output / f"validation_step_{step + 1:06d}.json").write_text(
                    json.dumps(validation_metrics, indent=2)
                )
            else:
                validation_metrics = gate = None
            if validation_metrics is not None:
                row["validation"] = validation_metrics
                decisions = selector.consider(
                    step=step + 1, metrics=validation_metrics, gate=gate
                ) if smoke_steps is None else []
                checkpoint_metrics = {
                    **validation_metrics,
                    "clean_gate_pass": gate.passed,
                    "clean_gate_margins": gate.margins,
                    "checkpoint_decisions": decisions,
                }
                for filename in decisions:
                    _save_checkpoint(
                        output / filename,
                        model=model, optimizer=optimizer, step=step + 1,
                        curriculum_stage=stage.name, selector=selector,
                        validation_manifest=validation_manifest, delayed_csi=delayed_csi,
                        model_config=checkpoint_config, fine_tune_config=resolved,
                        metrics=checkpoint_metrics, source=source,
                    )
                _save_checkpoint(
                    output / "last.pt",
                    model=model, optimizer=optimizer, step=step + 1,
                    curriculum_stage=stage.name, selector=selector,
                    validation_manifest=validation_manifest, delayed_csi=delayed_csi,
                    model_config=checkpoint_config, fine_tune_config=resolved,
                    metrics=checkpoint_metrics, source=source,
                )
            log.write(json.dumps(row) + "\n")
            log.flush()
    summary = {
        "initial_checkpoint": str(source),
        "initial_checkpoint_sha256": sha256(source),
        "steps_completed": total_steps,
        "full_20000_step_training_completed": total_steps == 20000,
        "codec_trainable_parameters": sum(
            parameter.numel() for parameter in codec.parameters() if parameter.requires_grad
        ),
        "jscc_trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "last_checkpoint": str(output / "last.pt"),
        "validation_external_required": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
