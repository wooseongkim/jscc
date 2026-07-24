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

from evaluation.paired import generate_paired_evaluation_batch, run_mode_on_paired_batch
from speech_jscc.config import load_config, resolve_device
from speech_jscc.diagnostics.waveform_aware_wireless import (
    clean_validation_conditions,
    ideal_equivalence_gate,
    ideal_ofdm_round_trip,
    ragged_tensor_diagnostics,
    validate_cf2_contract,
    validation_suite_hash,
    wireless_feasibility_gate,
)
from speech_jscc.experiment import build_components
from speech_jscc.training.channel_free_revalidation import (
    per_layer_nmse,
    summed_latent_statistics,
)
from src.evaluation.waveform_metrics import waveform_metrics
from train_channel_free_conv_conformer import fixed_paths, load_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/waveform_aware_wireless.yaml")
    parser.add_argument("--checkpoint",
                        default="runs/channel_free_revalidation/cf2_50frames_1920/best_waveform_si_sdr.pt")
    parser.add_argument("--mode", choices=("preflight", "ideal_ofdm", "clean_zero_shot", "all"),
                        default="all")
    parser.add_argument("--output-root", default="runs/waveform_aware_wireless")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--utterances", type=int, default=64)
    parser.add_argument("--realizations-per-utterance", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-long-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _prepare(path: Path, overwrite: bool) -> Path:
    if path.exists():
        if not overwrite:
            raise SystemExit(f"refusing existing output directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    (path / "waveform_examples").mkdir()
    return path


def _mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _latent_fields(reconstruction: torch.Tensor, target: torch.Tensor) -> dict:
    layer = per_layer_nmse(reconstruction, target)
    summed = summed_latent_statistics(reconstruction, target)
    return {
        "per_layer_nmse": [float(value) for value in layer],
        "aggregate_layer_nmse": float(layer.mean()),
        "summed_latent_nmse": float(summed["nmse"]),
        "summed_latent_snr_db": float(summed["snr_db"]),
        "summed_latent_correlation": float(summed["correlation"]),
        "summed_latent_power_ratio": float(summed["power_ratio"]),
    }


def _waveform_row(path: Path, waveform, clean, decoded, reconstruction, target, sr: int) -> dict:
    clean_metrics = waveform_metrics(waveform, clean, sr)
    current = waveform_metrics(waveform, decoded, sr)
    return {
        "utterance_id": str(path),
        **_latent_fields(reconstruction, target),
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


def _aggregate(rows: list[dict]) -> dict:
    keys = ("clean_si_sdr_db", "si_sdr_db", "delta_si_sdr_db",
            "clean_waveform_snr_db", "waveform_snr_db", "delta_waveform_snr_db",
            "clean_stft_l1", "stft_l1", "stft_ratio", "aggregate_layer_nmse",
            "summed_latent_nmse", "summed_latent_snr_db",
            "summed_latent_correlation", "summed_latent_power_ratio")
    return {key: _mean(rows, key) for key in keys}


def _write_rows(path: Path, rows: list[dict]) -> None:
    scalar_rows = [{key: value for key, value in row.items()
                    if not isinstance(value, (list, dict))} for row in rows]
    fields = sorted({key for row in scalar_rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scalar_rows)


def evaluate_direct_and_ideal(codec, model, paths, config, device, output: Path) -> dict:
    rows_direct, rows_ideal = [], []
    sr = int(config["codec"]["sample_rate"])
    model.eval()
    with torch.no_grad():
        for index, path in enumerate(paths):
            waveform = load_batch([path], config, device)
            target = codec.encode_waveform(waveform)
            state = target.new_zeros((1, model.encoder.channel_state_dim))
            symbols = model.encoder(target, state)
            direct_latent = model.decoder(symbols, state)
            mapped = ideal_ofdm_round_trip(symbols, model, config)
            if mapped["max_abs_error"] > 1e-7:
                raise RuntimeError("ideal OFDM data-symbol identity failed")
            ideal_latent = model.decoder(mapped["restored"], state)
            clean = codec.decode_representation(target)
            direct_audio = codec.decode_representation(direct_latent)
            ideal_audio = codec.decode_representation(ideal_latent)
            rows_direct.append(_waveform_row(path, waveform, clean, direct_audio,
                                             direct_latent, target, sr))
            rows_ideal.append(_waveform_row(path, waveform, clean, ideal_audio,
                                            ideal_latent, target, sr))
            if index < 8:
                wavfile.write(output / "waveform_examples" / f"{index:03d}_reference.wav",
                              sr, clean.squeeze().cpu().numpy().astype("float32"))
                wavfile.write(output / "waveform_examples" / f"{index:03d}_ideal.wav",
                              sr, ideal_audio.squeeze().cpu().numpy().astype("float32"))
    direct, ideal = _aggregate(rows_direct), _aggregate(rows_ideal)
    gate = ideal_equivalence_gate(direct, ideal)
    summary = {"direct_cf2": direct, "ideal_ofdm": ideal, "equivalence_gate": gate}
    _write_rows(output / "direct_utterance_metrics.csv", rows_direct)
    _write_rows(output / "ideal_utterance_metrics.csv", rows_ideal)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def evaluate_clean(codec, model, paths, config, device, output: Path,
                   realizations: int) -> dict:
    conditions = clean_validation_conditions(
        int(config["seed"]), utterance_count=len(paths),
        realizations_per_utterance=realizations,
        snr_bins=tuple(map(float, config["channel"]["random_snr_bins_db"])),
    )
    sr = int(config["codec"]["sample_rate"])
    rows = []
    model.eval()
    with torch.no_grad():
        for condition in conditions:
            path = paths[condition["utterance_index"]]
            waveform = load_batch([path], config, device)
            target = codec.encode_waveform(waveform)
            batch = generate_paired_evaluation_batch(
                codec, batch_size=1, waveform_samples=int(config["codec"]["waveform_samples"]),
                channel_shape=tuple(config["model"]["grid_shape"]),
                snr_db=float(condition["snr_db"]), jsr_db=-120.0, jammer_type="none",
                jammed_fraction=0.0, pilot_spacing=int(config["channel"]["pilot_spacing"]),
                pilot_time_spacing=int(config["channel"]["pilot_time_spacing"]),
                target_power=float(config["model"]["target_power"]), seed=int(condition["seed"]),
                device=device, fading="multipath_block",
                num_taps=int(config["channel"]["num_taps"]),
                pdp_decay=float(config["channel"]["pdp_decay"]),
                channel_estimator="dft_tap_ls",
                estimator_num_taps=int(config["channel"]["estimator_num_taps"]),
                estimator_ridge_lambda=float(config["channel"]["estimator_ridge_lambda"]),
                waveform=waveform, representation=target,
            )
            state = target.new_zeros((1, model.encoder.channel_state_dim))
            gates = target.new_ones((1, model.encoder.num_layers))
            result = run_mode_on_paired_batch(
                codec, model, batch, state, gates, equalizer="estimated",
                fading="multipath_block", channel_estimator="dft_tap_ls",
                estimator_num_taps=int(config["channel"]["estimator_num_taps"]),
                estimator_ridge_lambda=float(config["channel"]["estimator_ridge_lambda"]),
                allocation_mode="uniform", receiver_state_mode="observable_v1",
                decode_waveform=True,
            )
            clean = codec.decode_representation(target)
            row = _waveform_row(path, waveform, clean, result["decoded_waveform"],
                                result["reconstruction"], target, sr)
            row.update({
                **condition,
                "csi_nmse": float(result["csi_nmse"].mean()),
                "pilot_evm": float(result["pilot_evm"].mean()),
                "post_equalization_sinr_linear": float(result["post_equalization_sinr"].mean()),
                "post_equalization_sinr_db": float(
                    10 * torch.log10(result["post_equalization_sinr"].mean().clamp_min(1e-12))
                ),
            })
            rows.append(row)
    groups = {}
    for policy in ("fixed", "random"):
        members = [row for row in rows if row["channel_policy"] == policy]
        groups[policy] = _aggregate(members)
        groups[policy]["gate"] = wireless_feasibility_gate(groups[policy])
    groups["random_by_snr"] = {}
    for snr in config["channel"]["random_snr_bins_db"]:
        members = [row for row in rows if row["channel_policy"] == "random"
                   and row["snr_db"] == float(snr)]
        groups["random_by_snr"][str(float(snr))] = {
            **_aggregate(members),
            "csi_nmse": _mean(members, "csi_nmse"),
            "pilot_evm": _mean(members, "pilot_evm"),
            "post_equalization_sinr_db": _mean(members, "post_equalization_sinr_db"),
        }
    summary = {
        "validation_suite_hash": validation_suite_hash(conditions),
        "utterances": len(paths),
        "realizations_per_utterance": realizations,
        **groups,
        "fine_tuning_required": not groups["random"]["gate"]["passed"],
        "jammer_experiments_unblocked": groups["random"]["gate"]["passed"],
    }
    _write_rows(output / "realization_metrics.csv", rows)
    (output / "validation_suite.json").write_text(json.dumps(conditions, indent=2))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    dry = {
        "dry_run": True, "mode": args.mode, "checkpoint": args.checkpoint,
        "output_root": args.output_root, "utterances": args.utterances,
        "realizations_per_utterance": args.realizations_per_utterance,
    }
    if args.dry_run:
        print(json.dumps(dry, indent=2))
        return
    if args.utterances > 2 and not args.allow_long_run:
        raise SystemExit("full waveform-aware evaluation requires --allow-long-run")
    settings = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    config["device"] = args.device
    config["model"]["grid_shape"] = settings["model"]["grid_shape"]
    config["channel"] = settings["channel"]
    device = resolve_device(args.device)
    codec, model = build_components(config, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    codec.eval().requires_grad_(False)
    preflight = validate_cf2_contract(model, config)
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    _, preflight_paths = fixed_paths(config, int(config["seed"]))
    with torch.no_grad():
        waveform = load_batch([preflight_paths[0]], config, device)
        target = codec.encode_waveform(waveform)
        state = target.new_zeros((1, model.encoder.channel_state_dim))
        symbols, aux = model.encoder(target, state, return_aux=True)
        ragged = ragged_tensor_diagnostics(
            aux["fixed_width_symbols"], aux["symbol_valid_mask"]
        )
        mapping = ideal_ofdm_round_trip(symbols, model, config)
    if mapping["max_abs_error"] > 1e-7:
        raise SystemExit("preflight pack/map/deallocate/unpack identity failed")
    preflight["ragged_tensor"] = ragged
    preflight["mapping_round_trip_max_abs_error"] = mapping["max_abs_error"]
    (root / "preflight.json").write_text(json.dumps(preflight, indent=2))
    if args.mode == "preflight":
        print(json.dumps(preflight, indent=2))
        return
    paths = preflight_paths[:args.utterances]
    if args.mode in {"ideal_ofdm", "all"}:
        ideal_output = _prepare(root / "ideal_ofdm", args.overwrite)
        ideal = evaluate_direct_and_ideal(codec, model, paths, config, device, ideal_output)
        if not ideal["equivalence_gate"]["passed"]:
            raise SystemExit("ideal OFDM equivalence failed; clean-channel evaluation blocked")
        if args.mode == "ideal_ofdm":
            print(json.dumps(ideal, indent=2))
            return
    elif not (root / "ideal_ofdm" / "summary.json").exists():
        raise SystemExit("clean zero-shot requires a passing ideal OFDM summary")
    else:
        ideal = json.loads((root / "ideal_ofdm" / "summary.json").read_text())
        if not ideal["equivalence_gate"]["passed"]:
            raise SystemExit("clean zero-shot blocked by failed ideal OFDM")
    clean_output = _prepare(root / "clean_channel_zero_shot", args.overwrite)
    clean = evaluate_clean(codec, model, paths, config, device, clean_output,
                           args.realizations_per_utterance)
    final = {
        "source_checkpoint": args.checkpoint,
        "preflight": preflight,
        "ideal_ofdm": ideal["equivalence_gate"],
        "clean_channel_zero_shot": clean,
        "fine_tuning_required": clean["fine_tuning_required"],
        "selected_checkpoint": args.checkpoint,
        "jammer_experiments_unblocked": clean["jammer_experiments_unblocked"],
    }
    final_dir = root / "final_comparison"
    final_dir.mkdir(exist_ok=True)
    (final_dir / "summary.json").write_text(json.dumps(final, indent=2))
    (final_dir / "report.md").write_text(
        "# Waveform-aware wireless integration\n\n"
        f"- Ideal OFDM pass: **{ideal['equivalence_gate']['passed']}**\n"
        f"- Random clean pass: **{not clean['fine_tuning_required']}**\n"
        f"- Fine-tuning required: **{clean['fine_tuning_required']}**\n"
        f"- Jammer experiments unblocked: **{clean['jammer_experiments_unblocked']}**\n"
    )
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
