from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import torch
import yaml
from scipy.io import wavfile

from speech_jscc.config import load_config, resolve_device
from speech_jscc.experiment import build_components
from speech_jscc.training.channel_free_revalidation import (
    feasibility_classification,
    per_layer_nmse,
    summed_latent_statistics,
)
from src.evaluation.waveform_metrics import waveform_metrics
from train_channel_free_conv_conformer import fixed_paths, load_batch, validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/channel_free_revalidation.yaml")
    parser.add_argument("--mode", choices=("baseline", "final"), required=True)
    parser.add_argument("--baseline-kind", choices=("official", "continuous_sum", "both"),
                        default="both")
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="label=checkpoint.pt; repeat for each model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-long-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def prepare(path: Path, overwrite: bool) -> Path:
    if path.exists():
        if not overwrite:
            raise SystemExit(f"refusing existing output directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    audio = path / "waveform_examples"
    audio.mkdir()
    return audio


def average(rows: list[dict], keys: tuple[str, ...]) -> dict:
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


def baseline_evaluation(codec, paths: list[Path], config: dict,
                        device: torch.device, audio: Path) -> dict:
    sr = int(config["codec"]["sample_rate"])
    rows = []
    with torch.no_grad():
        for index, path in enumerate(paths):
            waveform = load_batch([path], config, device)
            layers = codec.encode_waveform(waveform)
            official = codec.official_reconstruct_waveform(waveform)
            continuous = codec.decode_representation(layers)
            for mode, decoded in (("official", official), ("continuous_sum", continuous)):
                metrics = waveform_metrics(waveform, decoded, sr)
                rows.append({"utterance_id": str(path), "mode": mode, **metrics})
                if index < 8:
                    wavfile.write(audio / f"{index:03d}_{mode}.wav", sr,
                                  decoded.squeeze().cpu().numpy().astype("float32"))
    summary = {}
    for mode in ("official", "continuous_sum"):
        members = [row for row in rows if row["mode"] == mode]
        summary[mode] = average(
            members, ("si_sdr_db", "waveform_snr_db", "stft_l1",
                      "multi_resolution_stft_distance")
        )
    return {"rows": rows, "summary": summary}


def matched_gaussian(codec, model, paths: list[Path], config: dict,
                     device: torch.device, seed: int = 23001) -> dict:
    targets, reconstructions = [], []
    model.eval()
    with torch.no_grad():
        for path in paths:
            waveform = load_batch([path], config, device)
            target = codec.encode_waveform(waveform)
            state = target.new_zeros((1, model.encoder.channel_state_dim))
            reconstruction = model.decoder(model.encoder(target, state), state)
            targets.append(target)
            reconstructions.append(reconstruction)
    target = torch.cat(targets)
    reconstruction = torch.cat(reconstructions)
    layer_target = per_layer_nmse(reconstruction, target)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(target.shape, generator=generator, device=device, dtype=target.dtype)
    powers = target.square().mean(dim=(0, 2, 3), keepdim=True)
    layer_noise = noise / noise.square().mean(dim=(0, 2, 3), keepdim=True).sqrt()
    layer_noise = layer_noise * (layer_target[None, :, None, None] * powers).sqrt()
    aggregate_matched = target + layer_noise
    target_sum = target.sum(1)
    wanted_sum_nmse = summed_latent_statistics(reconstruction, target)["nmse"]
    sum_noise = noise.sum(1)
    scale = (
        wanted_sum_nmse * target_sum.square().mean()
        / sum_noise.square().mean().clamp_min(1e-12)
    ).sqrt()
    summed_matched = target + noise * scale
    return {
        "aggregate_layer_nmse_matched": {
            "per_layer_nmse": [float(value) for value in per_layer_nmse(aggregate_matched, target)],
            "summed": {key: float(value) for key, value in
                       summed_latent_statistics(aggregate_matched, target).items()},
        },
        "summed_latent_nmse_matched": {
            "per_layer_nmse": [float(value) for value in per_layer_nmse(summed_matched, target)],
            "summed": {key: float(value) for key, value in
                       summed_latent_statistics(summed_matched, target).items()},
        },
    }


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(json.dumps(vars(args), indent=2))
        return
    if args.samples > 2 and not args.allow_long_run:
        raise SystemExit("64-utterance evaluation requires --allow-long-run")
    config = load_config(args.config)
    if "latent_cache_dir" in config.get("data", {}):
        raise SystemExit("latent cache is forbidden")
    config["device"] = args.device
    device = resolve_device(args.device)
    codec, _ = build_components(config, device)
    codec.eval()
    codec.requires_grad_(False)
    _, paths = fixed_paths(config, int(config["seed"]))
    paths = paths[:args.samples]
    output = Path(args.output_dir)
    audio = prepare(output, args.overwrite)
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n")
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    if args.mode == "baseline":
        result = baseline_evaluation(codec, paths, config, device, audio)
        selected = (("official", "continuous_sum") if args.baseline_kind == "both"
                    else (args.baseline_kind,))
        result["rows"] = [row for row in result["rows"] if row["mode"] in selected]
        result["summary"] = {key: result["summary"][key] for key in selected}
        with (output / "utterance_metrics.csv").open("w", newline="") as handle:
            fields = sorted({key for row in result["rows"] for key in row})
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(result["rows"])
        (output / "summary.json").write_text(json.dumps(result["summary"], indent=2))
        print(json.dumps(result["summary"], indent=2))
        return
    if not args.checkpoint:
        raise SystemExit("final evaluation requires at least one --checkpoint label=path")
    comparison = {}
    for item in args.checkpoint:
        label, value = item.split("=", 1)
        checkpoint = torch.load(value, map_location="cpu", weights_only=False)
        local = checkpoint["config"]
        local["device"] = args.device
        _, model = build_components(local, device)
        model.load_state_dict(checkpoint["model"], strict=True)
        result = validate(codec, model, paths, local, device)
        result["aggregate"]["matched_gaussian"] = matched_gaussian(
            codec, model, paths, local, device
        )
        result["aggregate"]["model_parameter_count"] = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        result["aggregate"]["optimizer_steps"] = checkpoint["step"]
        result["aggregate"]["symbol_frames"] = model.encoder.symbol_frames
        result["aggregate"]["channel_uses"] = model.encoder.total_channel_uses
        comparison[label] = result["aggregate"]
    feasible = [label for label, result in comparison.items()
                if result["classification"] == "CHANNEL_FREE_FEASIBLE"]
    summary = {
        "samples": len(paths),
        "comparison": comparison,
        "feasible_configurations": feasible,
        "classification": ("CHANNEL_FREE_FEASIBLE" if feasible
                           else "CHANNEL_FREE_CONV_CONFORMER_NOT_YET_FEASIBLE"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    with (output / "comparison.csv").open("w", newline="") as handle:
        keys = ("si_sdr_db", "delta_si_sdr_db", "waveform_snr_db",
                "delta_waveform_snr_db", "stft_ratio", "aggregate_layer_nmse",
                "summed_latent_nmse", "summed_latent_snr_db",
                "summed_latent_power_ratio", "model_parameter_count",
                "optimizer_steps", "symbol_frames", "channel_uses")
        writer = csv.DictWriter(handle, fieldnames=("configuration",) + keys)
        writer.writeheader()
        for label, result in comparison.items():
            writer.writerow({"configuration": label, **{key: result[key] for key in keys}})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
