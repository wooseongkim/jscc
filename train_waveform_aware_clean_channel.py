from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import torch
import yaml

from evaluation.paired import generate_paired_evaluation_batch, run_mode_on_paired_batch
from speech_jscc.config import load_config, resolve_device
from speech_jscc.diagnostics.random_distribution import SeedDeriver
from speech_jscc.diagnostics.waveform_aware_wireless import validate_cf2_contract
from speech_jscc.experiment import build_components
from speech_jscc.training.channel_free_feasibility import (
    decode_frozen_representation_with_gradient,
    enable_frozen_rnn_backward,
)
from speech_jscc.training.channel_free_revalidation import (
    component_gradient_norms,
    curriculum_weights,
    summed_latent_statistics,
    waveform_connected_objective,
)
from src.evaluation.waveform_metrics import waveform_metrics
from train_channel_free_conv_conformer import fixed_paths, load_batch, sha256


def decode_for_wireless_loss(codec, reconstruction):
    """Decode while leaving frozen codec weights fixed and enabling input gradients."""
    return decode_frozen_representation_with_gradient(codec, reconstruction)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/waveform_aware_wireless.yaml")
    parser.add_argument("--checkpoint",
                        default="runs/channel_free_revalidation/cf2_50frames_1920/best_waveform_si_sdr.pt")
    parser.add_argument("--zero-shot-summary",
                        default="runs/waveform_aware_wireless/clean_channel_zero_shot/summary.json")
    parser.add_argument("--output-dir",
                        default="runs/waveform_aware_wireless/clean_channel_training")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--validation-every", type=int, default=250)
    parser.add_argument("--resume")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-long-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _paired(codec, representation, waveform, config, device, seed: int, snr: float):
    return generate_paired_evaluation_batch(
        codec, batch_size=representation.shape[0],
        waveform_samples=int(config["codec"]["waveform_samples"]),
        channel_shape=tuple(config["model"]["grid_shape"]), snr_db=snr, jsr_db=-120.0,
        jammer_type="none", jammed_fraction=0.0,
        pilot_spacing=int(config["channel"]["pilot_spacing"]),
        pilot_time_spacing=int(config["channel"]["pilot_time_spacing"]),
        target_power=float(config["model"]["target_power"]), seed=seed, device=device,
        fading="multipath_block", num_taps=int(config["channel"]["num_taps"]),
        pdp_decay=float(config["channel"]["pdp_decay"]),
        channel_estimator="dft_tap_ls",
        estimator_num_taps=int(config["channel"]["estimator_num_taps"]),
        estimator_ridge_lambda=float(config["channel"]["estimator_ridge_lambda"]),
        waveform=waveform, representation=representation,
    )


def _forward(codec, model, representation, waveform, config, device, seed: int, snr: float):
    batch = _paired(codec, representation, waveform, config, device, seed, snr)
    state = representation.new_zeros((representation.shape[0], model.encoder.channel_state_dim))
    gates = representation.new_ones((representation.shape[0], model.encoder.num_layers))
    return run_mode_on_paired_batch(
        codec, model, batch, state, gates, equalizer="estimated",
        fading="multipath_block", channel_estimator="dft_tap_ls",
        estimator_num_taps=int(config["channel"]["estimator_num_taps"]),
        estimator_ridge_lambda=float(config["channel"]["estimator_ridge_lambda"]),
        allocation_mode="uniform", receiver_state_mode="observable_v1",
        decode_waveform=False,
    )


def _validate(codec, model, paths, config, device, seed: int) -> dict:
    model.eval()
    rows = []
    with torch.no_grad():
        for index, path in enumerate(paths):
            waveform = load_batch([path], config, device)
            target = codec.encode_waveform(waveform)
            snr = (5.0, 10.0, 15.0)[index % 3]
            result = _forward(codec, model, target, waveform, config, device,
                              SeedDeriver(seed).seed("wireless_validation", index), snr)
            decoded = codec.decode_representation(result["reconstruction"])
            clean = codec.decode_representation(target)
            current = waveform_metrics(waveform, decoded, int(config["codec"]["sample_rate"]))
            reference = waveform_metrics(waveform, clean, int(config["codec"]["sample_rate"]))
            rows.append({
                "summed_latent_nmse": float(
                    summed_latent_statistics(result["reconstruction"], target)["nmse"]
                ),
                "si_sdr_db": current["si_sdr_db"],
                "delta_si_sdr_db": current["si_sdr_db"] - reference["si_sdr_db"],
            })
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(json.dumps({
            "dry_run": True, "checkpoint": args.checkpoint,
            "zero_shot_summary": args.zero_shot_summary, "output_dir": args.output_dir,
            "steps": args.steps,
            "checkpoint_names": ["best_summed_latent_nmse.pt", "best_waveform_si_sdr.pt", "last.pt"],
        }, indent=2))
        return
    zero = json.loads(Path(args.zero_shot_summary).read_text())
    if bool(zero["random"]["gate"]["passed"]):
        raise SystemExit("fine-tuning is not required because random clean zero-shot passed")
    if args.steps > 5 and not args.allow_long_run:
        raise SystemExit("long clean-channel fine-tuning requires --allow-long-run")
    settings = load_config(args.config)
    source = Path(args.checkpoint)
    expected = Path("runs/channel_free_revalidation/cf2_50frames_1920/best_waveform_si_sdr.pt")
    if source.resolve() != expected.resolve():
        raise SystemExit(f"fine-tuning must initialize strictly from {expected}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    config = payload["config"]
    config["device"] = args.device
    config["model"]["grid_shape"] = settings["model"]["grid_shape"]
    config["channel"] = settings["channel"]
    device = resolve_device(args.device)
    codec, model = build_components(config, device)
    model.load_state_dict(payload["model"], strict=True)
    validate_cf2_contract(model, config)
    codec.eval().requires_grad_(False)
    enable_frozen_rnn_backward(codec.model.decoder)
    if any(parameter.requires_grad for parameter in codec.parameters()):
        raise RuntimeError("SpeechTokenizer must remain frozen")
    output = Path(args.output_dir)
    if output.exists() and not args.resume:
        if not args.overwrite:
            raise SystemExit(f"refusing existing output directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n")
    train_paths, validation_paths = fixed_paths(config, int(config["seed"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"]["learning_rate"]))
    start, history = 0, []
    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        start, history = int(resume["step"]), list(resume["history"])
    best_sum = float("inf")
    best_waveform = -float("inf")
    derive = SeedDeriver(int(config["seed"]))
    order: list[Path] = []
    epochs = (args.steps * args.batch_size + len(train_paths) - 1) // len(train_paths) + 1
    for epoch in range(epochs):
        items = list(train_paths)
        random.Random(int(config["seed"]) + epoch).shuffle(items)
        order.extend(items)
    with (output / "metrics.jsonl").open("a" if args.resume else "w") as log:
        for step in range(start + 1, args.steps + 1):
            paths = order[(step - 1) * args.batch_size:step * args.batch_size]
            waveform = load_batch(paths, config, device)
            with torch.no_grad():
                target = codec.encode_waveform(waveform)
            generator = torch.Generator().manual_seed(derive.seed("wireless_train_snr", step))
            snr = float(torch.empty(1).uniform_(5.0, 15.0, generator=generator))
            model.train()
            result = _forward(codec, model, target, waveform, config, device,
                              derive.seed("wireless_train_channel", step), snr)
            weights = curriculum_weights(step, **{
                "stage1_steps": int(config["curriculum"]["stage1_steps"]),
                "stage2_steps": int(config["curriculum"]["stage2_steps"]),
                "lambda_layer": float(config["curriculum"]["lambda_layer"]),
                "lambda_sum": float(config["curriculum"]["lambda_sum"]),
                "lambda_stft": float(config["curriculum"]["lambda_stft"]),
                "lambda_sisdr": float(config["curriculum"]["lambda_sisdr"]),
            })
            loss, components = waveform_connected_objective(
                result["reconstruction"], target, waveform,
                lambda layers: decode_for_wireless_loss(codec, layers),
                weights=weights,
                fft_sizes=tuple(config["curriculum"]["fft_sizes"]),
            )
            gradients = component_gradient_norms(components, weights, model.parameters())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           float(config["train"]["gradient_clip_norm"]))
            optimizer.step()
            row = {"step": step, "loss": float(loss.detach()), "snr_db": snr,
                   "loss_components": {key: float(value.detach())
                                       for key, value in components.items()},
                   "component_gradient_norms": gradients}
            if step % args.validation_every == 0 or step == args.steps:
                validation = _validate(codec, model, validation_paths, config, device,
                                       int(config["seed"]))
                row["validation"] = validation
                state = {
                    "diagnostic_type": "waveform_aware_clean_channel",
                    "source_checkpoint": str(source),
                    "source_checkpoint_sha256": sha256(source),
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step, "history": history + [row], "config": config,
                    "validation": validation,
                }
                if validation["summed_latent_nmse"] < best_sum:
                    best_sum = validation["summed_latent_nmse"]
                    torch.save(state, output / "best_summed_latent_nmse.pt")
                if validation["si_sdr_db"] > best_waveform:
                    best_waveform = validation["si_sdr_db"]
                    torch.save(state, output / "best_waveform_si_sdr.pt")
                torch.save(state, output / "last.pt")
            history.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
    summary = {
        "source_checkpoint": str(source), "source_checkpoint_sha256": sha256(source),
        "steps": args.steps, "best_summed_latent_nmse": best_sum,
        "best_waveform_si_sdr_db": best_waveform,
        "selected_checkpoint": str(output / "best_waveform_si_sdr.pt"),
        "codec_trainable_parameters": sum(p.numel() for p in codec.parameters()
                                          if p.requires_grad),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
