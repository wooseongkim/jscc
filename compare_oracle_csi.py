from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from channels.pilot import (
    csi_nmse,
    equalize_with_csi,
    estimate_channel_ls,
    extract_data_resources,
    insert_data_and_pilots,
    make_pilot_mask,
    pilot_evm,
)
from evaluation.paired import generate_paired_evaluation_batch
from models.observable_channel_state import build_observable_receiver_state_v1
from models.resource_allocator import allocate_resources, deallocate_resources
from speech_jscc.config import load_config, resolve_device
from speech_jscc.diagnostics.oracle_csi_comparison import (
    channel_power_statistics,
    empirical_symbol_metrics,
    paired_evaluation_grid,
    residual_decomposition,
    suite_hash,
    summarize_distribution,
    tensor_hash,
)
from speech_jscc.diagnostics.waveform_aware_wireless import validate_cf2_contract
from speech_jscc.experiment import build_components
from speech_jscc.training.channel_free_revalidation import (
    per_layer_nmse,
    summed_latent_statistics,
)
from src.evaluation.waveform_metrics import waveform_metrics
from train_channel_free_conv_conformer import fixed_paths, load_batch


CONDITIONS = ("awgn_identity", "multipath_oracle_csi", "multipath_estimated_csi")
SUMMARY_METRICS = {
    "post_eq_sinr_empirical_db": True,
    "equalized_symbol_nmse": False,
    "aggregate_layer_nmse": False,
    "summed_latent_nmse": False,
    "summed_latent_snr_db": True,
    "si_sdr_db": True,
    "delta_si_sdr_db": True,
    "waveform_snr_db": True,
    "delta_waveform_snr_db": True,
    "stft_ratio": False,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/oracle_csi_comparison.yaml")
    parser.add_argument("--checkpoint",
                        default="runs/waveform_aware_wireless/clean_channel_training/best_waveform_si_sdr.pt")
    parser.add_argument("--output-dir", default="runs/oracle_csi_comparison")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--utterances", type=int, default=64)
    parser.add_argument("--realizations", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-long-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _prepare(root: Path, overwrite: bool):
    preserved_audit = root / "implementation_audit.md"
    audit = preserved_audit.read_text() if preserved_audit.exists() else None
    if root.exists():
        children = [p for p in root.iterdir() if p.name != "implementation_audit.md"]
        if children and not overwrite:
            raise SystemExit(f"refusing existing output directory: {root}")
        if overwrite:
            for child in children:
                shutil.rmtree(child) if child.is_dir() else child.unlink()
    root.mkdir(parents=True, exist_ok=True)
    if audit is not None:
        preserved_audit.write_text(audit)
    for name in (*CONDITIONS, "paired_comparison"):
        (root / name).mkdir()


def _row_metrics(codec, reconstruction, target, waveform, clean, sample_rate):
    decoded = codec.decode_representation(reconstruction)
    current = waveform_metrics(waveform, decoded, sample_rate)
    baseline = waveform_metrics(waveform, clean, sample_rate)
    layer = per_layer_nmse(reconstruction, target)
    summed = summed_latent_statistics(reconstruction, target)
    return {
        "per_layer_nmse": [float(value) for value in layer],
        "aggregate_layer_nmse": float(layer.mean()),
        "summed_latent_nmse": float(summed["nmse"]),
        "summed_latent_snr_db": float(summed["snr_db"]),
        "si_sdr_db": current["si_sdr_db"],
        "delta_si_sdr_db": current["si_sdr_db"] - baseline["si_sdr_db"],
        "waveform_snr_db": current["waveform_snr_db"],
        "delta_waveform_snr_db": current["waveform_snr_db"] - baseline["waveform_snr_db"],
        "stft_ratio": current["stft_l1"] / max(baseline["stft_l1"], 1e-12),
    }


def _write_csv(path: Path, rows: list[dict]):
    flattened = []
    for row in rows:
        item = dict(row)
        for key, value in list(item.items()):
            if isinstance(value, (list, dict)):
                item[key] = json.dumps(value, separators=(",", ":"))
        flattened.append(item)
    fields = sorted({key for row in flattened for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def _condition_summary(rows, bootstrap_seed, bootstrap_samples):
    output = {}
    for condition in CONDITIONS:
        output[condition] = {}
        for snr in (5.0, 10.0, 15.0):
            members = [row for row in rows
                       if row["condition"] == condition and row["nominal_snr_db"] == snr]
            output[condition][str(snr)] = {
                metric: summarize_distribution(
                    [float(row[metric]) for row in members],
                    higher_is_better=higher,
                    bootstrap_seed=bootstrap_seed + index + int(snr * 10),
                    bootstrap_samples=bootstrap_samples,
                )
                for index, (metric, higher) in enumerate(SUMMARY_METRICS.items())
            }
            for metric in (
                "transmit_symbol_power", "awgn_power",
                "channel_estimation_nmse", "pilot_evm",
                "equalized_symbol_correlation",
                "mean_h_power", "median_h_power", "minimum_h_power",
                "h_power_p05", "h_power_p10",
                "h_power_below_0_1_fraction", "h_power_below_0_01_fraction",
                "oracle_empirical_minus_theory_db",
                "noise_component_energy_ratio", "csi_distortion_energy_ratio",
                "total_residual_energy_ratio", "cross_term_energy_ratio",
            ):
                values = [float(row[metric]) for row in members if row.get(metric) is not None]
                if values:
                    output[condition][str(snr)][metric] = summarize_distribution(
                        values, higher_is_better=False,
                        bootstrap_seed=bootstrap_seed + 100 + int(snr * 10),
                        bootstrap_samples=bootstrap_samples,
                    )
    return output


def _paired(rows, bootstrap_seed, bootstrap_samples):
    indexed = {
        (row["condition"], row["utterance_index"], row["realization"],
         row["nominal_snr_db"]): row for row in rows
    }
    comparisons = []
    pairs = (
        ("A_to_B", "awgn_identity", "multipath_oracle_csi"),
        ("B_to_C", "multipath_oracle_csi", "multipath_estimated_csi"),
    )
    delta_metrics = (
        "post_eq_sinr_empirical_db", "equalized_symbol_nmse",
        "summed_latent_nmse", "si_sdr_db", "waveform_snr_db", "stft_ratio",
    )
    for label, before, after in pairs:
        for snr in (5.0, 10.0, 15.0):
            for utterance in range(64):
                for realization in range(2):
                    key = (utterance, realization, snr)
                    a = indexed.get((before, *key))
                    b = indexed.get((after, *key))
                    if a is None or b is None:
                        continue
                    comparisons.append({
                        "comparison": label, "utterance_index": utterance,
                        "realization": realization, "nominal_snr_db": snr,
                        **{f"delta_{metric}": float(b[metric]) - float(a[metric])
                           for metric in delta_metrics},
                    })
    summary = {}
    for label, _, _ in pairs:
        summary[label] = {}
        for snr in (5.0, 10.0, 15.0):
            members = [row for row in comparisons
                       if row["comparison"] == label and row["nominal_snr_db"] == snr]
            summary[label][str(snr)] = {
                key: summarize_distribution(
                    [float(row[key]) for row in members],
                    higher_is_better=True,
                    bootstrap_seed=bootstrap_seed + index + int(snr * 10),
                    bootstrap_samples=bootstrap_samples,
                )
                for index, key in enumerate(
                    f"delta_{metric}" for metric in delta_metrics
                )
            }
    return comparisons, summary


def main():
    args = parse_args()
    if args.dry_run:
        print(json.dumps({
            "dry_run": True, "checkpoint": args.checkpoint,
            "output_dir": args.output_dir, "utterances": args.utterances,
            "realizations": args.realizations, "snr_db": [5, 10, 15],
            "conditions": list(CONDITIONS), "training": False, "jammer": False,
        }, indent=2))
        return
    if (args.utterances > 2 or args.realizations > 1) and not args.allow_long_run:
        raise SystemExit("full oracle-CSI comparison requires --allow-long-run")
    settings = load_config(args.config)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    config["device"] = args.device
    config["model"]["grid_shape"] = [64, 32]
    config["channel"] = settings["channel"]
    device = resolve_device(args.device)
    codec, model = build_components(config, device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    codec.eval().requires_grad_(False)
    validate_cf2_contract(model, config)
    root = Path(args.output_dir)
    _prepare(root, args.overwrite)
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (root / "command.txt").write_text(" ".join(sys.argv) + "\n")
    _, paths = fixed_paths(config, int(settings["seed"]))
    paths = paths[:args.utterances]
    grid = paired_evaluation_grid(
        int(settings["seed"]), utterance_count=len(paths),
        realizations=args.realizations,
        snr_values=tuple(map(float, settings["snr_db"])),
    )
    (root / "evaluation_ids.json").write_text(json.dumps(
        {"ordered_ids": [str(path) for path in paths], "scenarios": grid,
         "suite_hash": suite_hash(grid)}, indent=2
    ))
    rows = []
    sample_rate = int(config["codec"]["sample_rate"])
    with torch.no_grad():
        for scenario in grid:
            path = paths[scenario["utterance_index"]]
            waveform = load_batch([path], config, device)
            target = codec.encode_waveform(waveform)
            encoder_state = target.new_zeros((1, model.encoder.channel_state_dim))
            symbols = model.encoder(target, encoder_state)
            allocation = allocate_resources(
                symbols, torch.ones_like(symbols.real),
                model.encoder.layer_channel_uses, mode="uniform"
            )
            pilot_mask = make_pilot_mask(
                (1, 64, 32), int(config["channel"]["pilot_spacing"]),
                time_spacing=int(config["channel"]["pilot_time_spacing"]), device=device
            )
            transmitted_grid, pilots = insert_data_and_pilots(
                allocation.symbols, pilot_mask
            )
            batch = generate_paired_evaluation_batch(
                codec, batch_size=1,
                waveform_samples=int(config["codec"]["waveform_samples"]),
                channel_shape=(64, 32), snr_db=float(scenario["snr_db"]),
                jsr_db=-120.0, jammer_type="none", jammed_fraction=0.0,
                pilot_spacing=int(config["channel"]["pilot_spacing"]),
                pilot_time_spacing=int(config["channel"]["pilot_time_spacing"]),
                target_power=float(config["model"]["target_power"]),
                seed=int(scenario["seed"]), device=device,
                fading="multipath_block",
                num_taps=int(config["channel"]["num_taps"]),
                pdp_decay=float(config["channel"]["pdp_decay"]),
                channel_estimator="dft_tap_ls",
                estimator_num_taps=int(config["channel"]["estimator_num_taps"]),
                estimator_ridge_lambda=float(config["channel"]["estimator_ridge_lambda"]),
                waveform=waveform, representation=target,
            )
            h, noise = batch.signal_fading, batch.noise
            received = h * transmitted_grid + noise
            h_hat = estimate_channel_ls(
                received, pilots, pilot_mask, fading="multipath_block",
                channel_estimator="dft_tap_ls",
                estimator_num_taps=int(config["channel"]["estimator_num_taps"]),
                estimator_ridge_lambda=float(config["channel"]["estimator_ridge_lambda"]),
            )
            clean = codec.decode_representation(target)
            common_hashes = {
                "transmitted_symbol_hash": tensor_hash(symbols),
                "channel_hash": tensor_hash(h),
                "noise_hash": tensor_hash(noise),
                "pilot_mask_hash": tensor_hash(pilot_mask),
                "resource_mapping_hash": tensor_hash(allocation.resource_to_source),
            }
            modes = {
                "awgn_identity": (
                    transmitted_grid + noise, torch.ones_like(h),
                    torch.ones_like(h),
                ),
                "multipath_oracle_csi": (received, h, h_hat),
                "multipath_estimated_csi": (received, h_hat, h_hat),
            }
            for condition, (condition_received, eq_channel, state_channel) in modes.items():
                equalized_grid = equalize_with_csi(condition_received, eq_channel)
                extracted = extract_data_resources(equalized_grid, pilot_mask)
                decoder_input = deallocate_resources(
                    extracted, allocation.resource_to_source
                )
                receiver_state = build_observable_receiver_state_v1(
                    condition_received, pilots, pilot_mask, state_channel
                )
                reconstruction = model.decoder(decoder_input, receiver_state)
                data_h = extract_data_resources(
                    torch.ones_like(h) if condition == "awgn_identity" else h,
                    pilot_mask,
                )
                requested_noise_power = float(config["model"]["target_power"]) / (
                    10.0 ** (float(scenario["snr_db"]) / 10.0)
                )
                symbol = empirical_symbol_metrics(
                    allocation.symbols, extracted, h=data_h,
                    requested_noise_power=requested_noise_power,
                    oracle=condition != "multipath_estimated_csi",
                )
                estimate_nmse = (
                    0.0 if condition != "multipath_estimated_csi"
                    else float(csi_nmse(h, h_hat))
                )
                condition_pilot_evm = float(pilot_evm(
                    condition_received, pilots, pilot_mask, state_channel
                ))
                row = {
                    "condition": condition, "utterance_id": str(path),
                    "utterance_index": scenario["utterance_index"],
                    "realization": scenario["realization"],
                    "seed": scenario["seed"],
                    "nominal_snr_db": scenario["snr_db"],
                    "awgn_power": float(noise.abs().square().mean()),
                    "channel_estimation_nmse": estimate_nmse,
                    "pilot_evm": condition_pilot_evm,
                    **common_hashes, **symbol,
                    **channel_power_statistics(data_h),
                    **_row_metrics(codec, reconstruction, target, waveform, clean,
                                   sample_rate),
                }
                if condition == "multipath_estimated_csi":
                    data_noise = extract_data_resources(noise, pilot_mask)
                    data_h_hat = extract_data_resources(h_hat, pilot_mask)
                    decomp = residual_decomposition(
                        allocation.symbols, data_h, data_h_hat, data_noise
                    )
                    row.update({
                        key: value for key, value in decomp.items()
                        if not torch.is_tensor(value)
                    })
                rows.append(row)
    for condition in CONDITIONS:
        members = [row for row in rows if row["condition"] == condition]
        _write_csv(root / condition / "per_sample_metrics.csv", members)
        (root / condition / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False)
        )
    summary = _condition_summary(
        rows, int(settings["seed"]), int(settings["bootstrap_samples"])
    )
    comparisons, paired_summary = _paired(
        rows, int(settings["seed"]) + 1000, int(settings["bootstrap_samples"])
    )
    _write_csv(root / "paired_comparison" / "paired_differences.csv", comparisons)
    (root / "paired_comparison" / "per_snr_summary.json").write_text(
        json.dumps(paired_summary, indent=2)
    )
    for condition in CONDITIONS:
        (root / condition / "per_snr_summary.json").write_text(
            json.dumps(summary[condition], indent=2)
        )
    final = {
        "checkpoint": args.checkpoint, "suite_hash": suite_hash(grid),
        "sample_count_per_condition": len(grid),
        "condition_summaries": summary, "paired_comparisons": paired_summary,
        "training_performed": False, "jammer_enabled": False,
    }
    (root / "final_summary.json").write_text(json.dumps(final, indent=2))
    lines = ["# Oracle CSI paired comparison", ""]
    for snr in ("5.0", "10.0", "15.0"):
        oracle = summary["multipath_oracle_csi"][snr]["post_eq_sinr_empirical_db"]["mean"]
        estimated = summary["multipath_estimated_csi"][snr]["post_eq_sinr_empirical_db"]["mean"]
        lines.append(
            f"- {snr} dB: oracle `{oracle:.3f}` dB; estimated `{estimated:.3f}` dB; "
            f"additional CSI loss `{oracle-estimated:.3f}` dB"
        )
    (root / "final_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
