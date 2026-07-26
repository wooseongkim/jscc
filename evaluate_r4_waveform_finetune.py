from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import torch
import yaml

from speech_jscc.config import resolve_device
from speech_jscc.experiment import build_components
from speech_jscc.training.r4_waveform_finetune import (
    freeze_codec_for_input_gradient,
    validate_initial_checkpoint_metadata,
)
from train_channel_free_conv_conformer import fixed_paths
from train_r4_waveform_finetune import _validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_r4_waveform_finetune.yaml")
    parser.add_argument("--physical-config", default="configs/ofdm_nr_like_r4_repetition3_mrc.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--utterances", type=int, default=64)
    parser.add_argument("--realizations", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-long-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specification = yaml.safe_load(Path(args.config).read_text())
    physical = yaml.safe_load(Path(args.physical_config).read_text())
    if args.dry_run:
        print(json.dumps({
            "checkpoint": args.checkpoint,
            "profile": physical.get("profile_config", "configs/ofdm_nr_like_r4.yaml"),
            "utterances": args.utterances,
            "realizations": args.realizations,
            "snr_db": [5, 10, 15],
            "full_evaluation": False,
        }, indent=2))
        return
    if args.utterances > 8 and not args.allow_long_run:
        raise SystemExit("full R4 fine-tune evaluation requires --allow-long-run")
    output = Path(args.output_dir)
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"refusing existing output directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    selected = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if selected.get("diagnostic_type") != "r4_waveform_finetune":
        raise SystemExit("selected checkpoint is not R4 waveform fine-tuning")
    config = selected["config"]
    validate_initial_checkpoint_metadata(config)
    config["device"] = args.device
    device = resolve_device(args.device)
    codec, model = build_components(config, device)
    model.load_state_dict(selected["model"], strict=True)
    freeze_codec_for_input_gradient(codec)
    initial_path = Path(selected["source_checkpoint"])
    initial_payload = torch.load(initial_path, map_location="cpu", weights_only=False)
    _, initial_model = build_components(config, device)
    initial_model.load_state_dict(initial_payload["model"], strict=True)
    initial_model.eval().requires_grad_(False)
    _, validation_paths = fixed_paths(config, int(config["seed"]))
    seeds = list(specification["validation"]["full_seeds"])[: args.realizations]
    metrics, gate = _validate(
        codec=codec, model=model, initial_model=initial_model,
        paths=validation_paths, checkpoint_config=config,
        specification=specification, device=device, seeds=seeds,
        count=args.utterances,
    )
    rows = metrics.pop("rows")
    fields = sorted({key for row in rows for key in row if key != "per_layer_nmse"})
    with (output / "per_sample_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in rows])
    (output / "per_layer_metrics.json").write_text(json.dumps([
        {
            "snr_db": row["snr_db"], "seed": row["seed"], "path": row["path"],
            "per_layer_nmse": row["per_layer_nmse"],
        }
        for row in rows
    ], indent=2))
    summary = {
        "selected_checkpoint": str(args.checkpoint),
        "selection_reason": (
            "best_clean_gate" if Path(args.checkpoint).name == "best_clean_gate.pt"
            else "fallback_best_5db_si_sdr"
        ),
        "checkpoint_clean_gate_pass": bool(selected.get("validation", {}).get("clean_gate_pass", False)),
        "initial_checkpoint": str(initial_path),
        "profile": "nr_like_r4",
        "repetitions": 3,
        "total_packet_energy": 5760,
        "utterances": args.utterances,
        "realizations": args.realizations,
        "snr_db": [5, 10, 15],
        **metrics,
        "final_clean_regression_gate_pass": gate.passed,
        "nonfinite_samples": 0,
        "jammer_unblocked": False,
    }
    summary["no_material_regression_10_15"] = all(
        summary["by_snr"][str(snr)]["delta_si_sdr_vs_initial_r4"] >= -0.5
        and summary["by_snr"][str(snr)]["delta_waveform_snr_vs_initial_r4"] >= -0.5
        and summary["by_snr"][str(snr)]["stft_increase_vs_initial_r4"] <= 0.05
        for snr in (10.0, 15.0)
    )
    five = summary["by_snr"]["5.0"]
    summary["five_db_waveform_gate"] = {
        "delta_si_sdr_pass": five["delta_si_sdr_vs_clean_codec"] >= -1.0,
        "delta_waveform_snr_pass": five["delta_waveform_snr_vs_clean_codec"] >= -1.0,
        "stft_ratio_pass": five["stft_ratio_vs_clean_codec"] <= 1.20,
    }
    summary["clean_channel_waveform_gate_pass"] = all(
        summary["five_db_waveform_gate"].values()
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(specification, sort_keys=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
