from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from scipy.io import wavfile

from speech_jscc.config import load_config, resolve_device
from speech_jscc.data import load_waveform_segment, resolve_waveform_splits
from speech_jscc.diagnostics.content_generalization import parse_speaker_id
from speech_jscc.experiment import build_components
from speech_jscc.training.channel_free_feasibility import (
    decode_frozen_representation_with_gradient,
    enable_frozen_rnn_backward,
    select_unseen_speaker_paths,
)
from speech_jscc.training.channel_free_revalidation import (
    CHANNEL_FREE_EXPERIMENTS,
    apply_experiment_definition,
    checkpoint_filenames,
    component_gradient_norms,
    curriculum_weights,
    feasibility_classification,
    framewise_summed_nmse,
    per_layer_nmse,
    summed_latent_statistics,
    waveform_connected_objective,
)
from src.evaluation.waveform_metrics import waveform_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/channel_free_revalidation.yaml")
    parser.add_argument("--experiment", choices=("cf1", "cf2", "cf3", "cf4"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--validation-samples", type=int)
    parser.add_argument("--initialization", choices=("fresh", "experiment_a"), default="fresh")
    parser.add_argument("--experiment-a-checkpoint",
                        default="runs/channel_free_feasibility/experiment_a_latent_only/best_latent.pt")
    parser.add_argument("--resume")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-long-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], check=False, capture_output=True, text=True
    ).stdout)
    return {"git_commit": commit, "working_tree_dirty": dirty}


def load_batch(paths: list[Path], config: dict, device: torch.device) -> torch.Tensor:
    return torch.stack([
        load_waveform_segment(
            path,
            int(config["codec"]["sample_rate"]),
            int(config["codec"]["waveform_samples"]),
        )
        for path in paths
    ]).to(device)


def fixed_paths(config: dict, seed: int) -> tuple[list[Path], list[Path]]:
    train, validation = resolve_waveform_splits(config["data"], seed)
    rng = random.Random(seed)
    rng.shuffle(train)
    train = train[: int(config["data"]["train_subset_size"])]
    unseen = select_unseen_speaker_paths(
        train, validation, limit=int(config["data"]["validation_utterances"]), seed=seed + 1
    )
    if len(unseen) < int(config["data"]["validation_utterances"]):
        raise ValueError("not enough unseen-speaker validation utterances")
    if {parse_speaker_id(path) for path in train} & {parse_speaker_id(path) for path in unseen}:
        raise ValueError("unseen-speaker validation overlaps training speakers")
    return train, unseen


def validate(codec, model, paths: list[Path], config: dict, device: torch.device,
             *, save_audio: Path | None = None) -> dict:
    model.eval()
    sr = int(config["codec"]["sample_rate"])
    rows, reconstructions, targets = [], [], []
    with torch.no_grad():
        for index, path in enumerate(paths):
            waveform = load_batch([path], config, device)
            target = codec.encode_waveform(waveform)
            state = target.new_zeros((1, model.encoder.channel_state_dim))
            symbols = model.encoder(target, state)
            reconstruction = model.decoder(symbols, state)
            clean = codec.decode_representation(target)
            decoded = codec.decode_representation(reconstruction)
            clean_metrics = waveform_metrics(waveform, clean, sr)
            current = waveform_metrics(waveform, decoded, sr)
            layer = per_layer_nmse(reconstruction, target)
            summed = summed_latent_statistics(reconstruction, target)
            row = {
                "utterance_id": str(path),
                "per_layer_nmse": [float(value) for value in layer],
                "aggregate_layer_nmse": float(layer.mean()),
                "summed_latent_nmse": float(summed["nmse"]),
                "summed_latent_snr_db": float(summed["snr_db"]),
                "summed_latent_correlation": float(summed["correlation"]),
                "summed_latent_power_ratio": float(summed["power_ratio"]),
                "clean_si_sdr_db": clean_metrics["si_sdr_db"],
                "si_sdr_db": current["si_sdr_db"],
                "delta_si_sdr_db": current["si_sdr_db"] - clean_metrics["si_sdr_db"],
                "clean_waveform_snr_db": clean_metrics["waveform_snr_db"],
                "waveform_snr_db": current["waveform_snr_db"],
                "delta_waveform_snr_db": current["waveform_snr_db"] - clean_metrics["waveform_snr_db"],
                "clean_stft_l1": clean_metrics["stft_l1"],
                "stft_l1": current["stft_l1"],
                "stft_ratio": current["stft_l1"] / max(clean_metrics["stft_l1"], 1e-12),
            }
            rows.append(row)
            reconstructions.append(reconstruction.cpu())
            targets.append(target.cpu())
            if save_audio is not None and index < 8:
                wavfile.write(save_audio / f"{index:03d}_reference.wav", sr,
                              clean.squeeze().cpu().numpy().astype("float32"))
                wavfile.write(save_audio / f"{index:03d}_reconstruction.wav", sr,
                              decoded.squeeze().cpu().numpy().astype("float32"))
    reconstruction = torch.cat(reconstructions)
    target = torch.cat(targets)
    layer = per_layer_nmse(reconstruction, target)
    summed = summed_latent_statistics(reconstruction, target)
    averages = {
        "per_layer_nmse": [float(value) for value in layer],
        "aggregate_layer_nmse": float(layer.mean()),
        "summed_latent_nmse": float(summed["nmse"]),
        "summed_latent_snr_db": float(summed["snr_db"]),
        "summed_latent_correlation": float(summed["correlation"]),
        "summed_latent_power_ratio": float(summed["power_ratio"]),
    }
    for key in ("clean_si_sdr_db", "si_sdr_db", "delta_si_sdr_db",
                "clean_waveform_snr_db", "waveform_snr_db", "delta_waveform_snr_db",
                "stft_ratio"):
        averages[key] = sum(row[key] for row in rows) / len(rows)
    averages["classification"] = feasibility_classification(
        averages["delta_si_sdr_db"], averages["delta_waveform_snr_db"],
        averages["stft_ratio"],
    )
    return {
        "rows": rows,
        "aggregate": averages,
        "framewise_summed_nmse": [
            float(value) for value in framewise_summed_nmse(reconstruction, target)
        ],
    }


def checkpoint_payload(model, optimizer, step: int, config: dict, experiment: str,
                       initialization: dict, history: list[dict], validation: dict) -> dict:
    return {
        "diagnostic_type": "channel_free_conv_conformer_revalidation",
        "experiment": experiment,
        "model_architecture": "conv_conformer_v1",
        "model_config": model.model_config,
        "channel_uses": model.encoder.total_channel_uses,
        "symbol_valid_mask": model.encoder.symbol_valid_mask.cpu(),
        "initialization": initialization,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": config,
        "history": history,
        "validation": validation,
    }


def main() -> None:
    args = parse_args()
    base = load_config(args.config)
    if "latent_cache_dir" in base.get("data", {}):
        raise SystemExit("latent cache is forbidden")
    config = apply_experiment_definition(base, args.experiment)
    default_steps = int(CHANNEL_FREE_EXPERIMENTS[args.experiment]["minimum_steps"])
    steps = int(args.steps or default_steps)
    batch_size = int(args.batch_size or config["train"]["batch_size"])
    validation_samples = int(args.validation_samples or config["data"]["validation_utterances"])
    dry = {
        "dry_run": True,
        "experiment": args.experiment,
        "output_dir": args.output_dir,
        "steps": steps,
        "batch_size": batch_size,
        "validation_samples": validation_samples,
        "initialization": args.initialization,
        "model": config["model"],
        "channel_components": [],
        "latent_cache_used": False,
    }
    if args.dry_run:
        print(json.dumps(dry, indent=2))
        return
    if steps > 5 and not args.allow_long_run:
        raise SystemExit("long channel-free training requires --allow-long-run")
    if args.initialization == "experiment_a" and args.experiment != "cf1":
        raise SystemExit("Experiment A initialization is shape-compatible only with CF-1")
    output = Path(args.output_dir)
    if output.exists():
        if args.resume:
            pass
        elif args.overwrite:
            shutil.rmtree(output)
        else:
            raise SystemExit(f"refusing existing output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    audio = output / "waveform_examples"
    audio.mkdir(exist_ok=True)
    config["device"] = args.device
    device = resolve_device(args.device)
    torch.manual_seed(int(config["seed"]))
    codec, model = build_components(config, device)
    codec.eval()
    codec.requires_grad_(False)
    if any(parameter.requires_grad for parameter in codec.parameters()):
        raise RuntimeError("SpeechTokenizer must remain frozen")
    enable_frozen_rnn_backward(codec.model.decoder)
    initialization = {"mode": args.initialization, "checkpoint": None, "sha256": None}
    if args.initialization == "experiment_a":
        source = Path(args.experiment_a_checkpoint)
        payload = torch.load(source, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        initialization.update(checkpoint=str(source), sha256=sha256(source))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    train_paths, validation_paths = fixed_paths(config, int(config["seed"]))
    validation_paths = validation_paths[:validation_samples]
    history: list[dict] = []
    start = 0
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume["experiment"] != args.experiment or resume["model_config"] != model.model_config:
            raise SystemExit("resume checkpoint configuration mismatch")
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        start = int(resume["step"])
        history = list(resume["history"])
    environment = {
        **git_state(),
        "torch_version": torch.__version__,
        "device": str(device),
        "python": sys.version,
        "codec_trainable_parameters": sum(p.numel() for p in codec.parameters() if p.requires_grad),
        "encoder_parameters": sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),
        "decoder_parameters": sum(p.numel() for p in model.decoder.parameters() if p.requires_grad),
    }
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "environment.json").write_text(json.dumps(environment, indent=2))
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n")
    (output / "dataset_manifest.json").write_text(json.dumps({
        "train": [str(path) for path in train_paths],
        "validation": [str(path) for path in validation_paths],
        "train_hash": hashlib.sha256("\n".join(map(str, train_paths)).encode()).hexdigest(),
        "validation_hash": hashlib.sha256("\n".join(map(str, validation_paths)).encode()).hexdigest(),
        "latent_cache_used": False,
    }, indent=2))
    best = {"per_layer_nmse": None, "summed_latent_nmse": None, "waveform_si_sdr": None}
    gradient_rows: list[dict] = []
    rng_order: list[Path] = []
    epochs = (steps * batch_size + len(train_paths) - 1) // len(train_paths) + 1
    for epoch in range(epochs):
        order = list(train_paths)
        random.Random(int(config["seed"]) + epoch).shuffle(order)
        rng_order.extend(order)
    metrics_path = output / "metrics.jsonl"
    with metrics_path.open("a" if args.resume else "w") as log:
        for step in range(start + 1, steps + 1):
            paths = rng_order[(step - 1) * batch_size:step * batch_size]
            waveform = load_batch(paths, config, device)
            with torch.no_grad():
                target = codec.encode_waveform(waveform)
            state = target.new_zeros((target.shape[0], model.encoder.channel_state_dim))
            model.train()
            optimizer.zero_grad(set_to_none=True)
            symbols = model.encoder(target, state)
            reconstruction = model.decoder(symbols, state)
            weights = curriculum_weights(step, **{
                key: config["curriculum"][key]
                for key in ("stage1_steps", "stage2_steps", "lambda_layer",
                            "lambda_sum", "lambda_stft", "lambda_sisdr")
            })
            loss, components = waveform_connected_objective(
                reconstruction, target, waveform,
                lambda layers: decode_frozen_representation_with_gradient(codec, layers),
                weights=weights,
                fft_sizes=tuple(config["curriculum"]["fft_sizes"]),
            )
            diagnostic_step = step == 1 or step % int(config["train"]["validation_every"]) == 0
            gradient_norms = component_gradient_norms(
                components, weights, model.parameters()
            ) if diagnostic_step else {}
            loss.backward()
            total_gradient = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["train"]["gradient_clip_norm"])
            ))
            optimizer.step()
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "loss_components": {name: float(value.detach()) for name, value in components.items()},
                "weighted_loss_components": {
                    name: float(value.detach()) * weights[name] for name, value in components.items()
                },
                "loss_weights": weights,
                "component_gradient_norms": gradient_norms,
                "total_gradient_norm_before_clip": total_gradient,
            }
            if diagnostic_step or step == steps:
                validation = validate(codec, model, validation_paths, config, device)
                row["validation"] = validation["aggregate"]
                payload = checkpoint_payload(
                    model, optimizer, step, config, args.experiment,
                    initialization, history + [row], validation,
                )
                candidates = {
                    "per_layer_nmse": validation["aggregate"]["aggregate_layer_nmse"],
                    "summed_latent_nmse": validation["aggregate"]["summed_latent_nmse"],
                    "waveform_si_sdr": validation["aggregate"]["si_sdr_db"],
                }
                for metric, value in candidates.items():
                    previous = best[metric]
                    improved = previous is None or (
                        value > previous["value"] if metric == "waveform_si_sdr"
                        else value < previous["value"]
                    )
                    if improved:
                        path = output / checkpoint_filenames()[metric]
                        torch.save(payload, path)
                        best[metric] = {"step": step, "value": value, "path": str(path)}
                gradient_rows.append({
                    "step": step, "component_gradient_norms": gradient_norms,
                    "total_gradient_norm_before_clip": total_gradient,
                })
            history.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            if step % int(config["train"]["checkpoint_every"]) == 0 or step == steps:
                torch.save(checkpoint_payload(
                    model, optimizer, step, config, args.experiment,
                    initialization, history, row.get("validation", {}),
                ), output / "last.pt")
    final_validation = validate(codec, model, validation_paths, config, device, save_audio=audio)
    with (output / "per_layer_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", "nmse"])
        writer.writerows(enumerate(final_validation["aggregate"]["per_layer_nmse"]))
    with (output / "framewise_summed_nmse.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "summed_nmse"])
        writer.writerows(enumerate(final_validation["framewise_summed_nmse"]))
    (output / "gradient_diagnostics.json").write_text(json.dumps(gradient_rows, indent=2))
    (output / "checkpoint_selection.json").write_text(json.dumps(best, indent=2))
    summary = {
        "experiment": args.experiment,
        "steps": steps,
        "initialization": initialization,
        "model": config["model"],
        "parameter_count": environment,
        "checkpoint_selection": best,
        "final_validation": final_validation["aggregate"],
        "temporal_symbol_pattern": config["model"].get("temporal_symbol_pattern"),
        "feasibility_claim_allowed": final_validation["aggregate"]["classification"] == "CHANNEL_FREE_FEASIBLE",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
