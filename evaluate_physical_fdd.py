from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import torch
import yaml

from channels.multipath import exponential_pdp
from channels.physical_ofdm import (
    NR_LIKE_R2,
    NR_LIKE_R3,
    active_grid_masks,
    apply_tti_multipath,
    demodulate_tti,
    estimate_comb_dft_ls,
    insert_physical_pilots,
    modulate_tti,
)
from channels.temporal_multipath import (
    correlated_tap_trajectory,
    doppler_frequency_hz,
    jakes_slot_correlation,
    measured_lag1_correlation,
)
from models.observable_channel_state import build_observable_receiver_state_v1
from speech_jscc.config import resolve_device
from speech_jscc.diagnostics.physical_fdd import (
    PhysicalCSIReport,
    allocate_physical_resources,
    lmmse_source_estimate,
)
from speech_jscc.experiment import build_components
from speech_jscc.training.channel_free_revalidation import (
    per_layer_nmse,
    summed_latent_statistics,
)
from src.evaluation.waveform_metrics import waveform_metrics
from train_channel_free_conv_conformer import fixed_paths, load_batch


PROFILES = {"nr_like_r2": NR_LIKE_R2, "nr_like_r3": NR_LIKE_R3}
CHECKPOINT = "runs/waveform_aware_wireless/clean_channel_training/best_waveform_si_sdr.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ofdm_nr_like_r3.yaml")
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--utterances", type=int, default=64)
    parser.add_argument("--realizations", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-long-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _extract(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.stack([values[index][mask] for index in range(values.shape[0])])


def _write_csv(path: Path, rows: list[dict]) -> None:
    scalar = [{key: value for key, value in row.items() if not isinstance(value, list)} for row in rows]
    fields = sorted({key for row in scalar for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scalar)


def _mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _summary(rows: list[dict]) -> dict:
    keys = (
        "post_mmse_sinr_db", "aggregate_layer_nmse", "summed_latent_nmse",
        "summed_latent_snr_db", "si_sdr_db", "delta_si_sdr_db",
        "waveform_snr_db", "delta_waveform_snr_db", "stft_ratio",
        "csi_nmse", "pilot_evm", "interpolation_error",
        "source_symbol_power", "allocated_data_re_power", "time_domain_tx_power",
        "allocated_power_min", "allocated_power_max", "allocated_power_mean",
        "papr", "selected_reliability_mean",
    )
    return {key: _mean(rows, key) for key in keys}


def main() -> None:
    args = parse_args()
    spec = yaml.safe_load(Path(args.config).read_text())
    profile = PROFILES[spec["profile"]["name"]]
    if args.dry_run:
        print(json.dumps({
            "profile": profile.name, "checkpoint": args.checkpoint,
            "output_dir": args.output_dir, "utterances": args.utterances,
            "realizations": args.realizations, "snr_db": spec["channel"]["snr_db"],
            "training": False, "jammer": False,
        }, indent=2))
        return
    if args.utterances > 2 and not args.allow_long_run:
        raise SystemExit("full physical FDD evaluation requires --allow-long-run")
    output = Path(args.output_dir)
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"refusing existing output directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    config["device"] = args.device
    device = resolve_device(args.device)
    codec, model = build_components(config, device)
    model.load_state_dict(saved["model"], strict=True)
    codec.eval().requires_grad_(False)
    model.eval()
    _, available_paths = fixed_paths(config, int(config["seed"]))
    paths = available_paths[: args.utterances]
    masks = active_grid_masks(profile, device=device)
    candidate_mask = masks.candidate_data

    physical = spec["physical"]
    fd = doppler_frequency_hz(
        physical["user_speed_mps"], physical["carrier_frequency_hz"]
    )
    rho = jakes_slot_correlation(
        physical["user_speed_mps"],
        physical["carrier_frequency_hz"],
        profile.tti_duration_s,
    )
    channel = spec["channel"]
    pdp = exponential_pdp(channel["num_taps"], channel["pdp_decay"])
    rows: list[dict] = []
    trajectory_rows: list[dict] = []
    importance = spec["allocation"]["layer_importance_order"]

    for snr_db in map(float, channel["snr_db"]):
        for realization in range(args.realizations):
            seed = int(config["seed"]) + round(snr_db * 1000) + realization * 100_003
            permutation = torch.randperm(
                args.utterances, generator=torch.Generator().manual_seed(seed + 11)
            ).tolist()
            taps_cpu = correlated_tap_trajectory(
                slots=args.utterances, batch_size=1, pdp=pdp, rho=rho, seed=seed + 17
            )
            noise_generator = torch.Generator(device=device).manual_seed(seed + 23)
            report: PhysicalCSIReport | None = None
            for tti, utterance_index in enumerate(permutation):
                path = paths[utterance_index]
                waveform = load_batch([path], config, device)
                with torch.no_grad():
                    target = codec.encode_waveform(waveform)
                    tx_state = target.new_zeros((1, model.encoder.channel_state_dim))
                    source = model.encoder(target, tx_state)
                    clean_audio = codec.decode_representation(target)
                allocation = allocate_physical_resources(
                    profile=profile, tx_tti=tti, report=report,
                    layer_importance_order=importance,
                    alpha=spec["allocation"]["alpha"],
                    minimum_relative_power=spec["allocation"]["minimum_relative_power"],
                    maximum_relative_power=spec["allocation"]["maximum_relative_power"],
                )
                data_grid = allocation.place(source)
                tx_grid, pilots = insert_physical_pilots(data_grid, profile)
                tx_waveform = modulate_tti(tx_grid, profile)
                taps = taps_cpu[tti].to(device)
                faded = apply_tti_multipath(tx_waveform, taps, profile)
                source_power = float(source.abs().square().mean())
                noise_variance = source_power / 10 ** (snr_db / 10)
                noise = torch.complex(
                    torch.randn(faded.shape, generator=noise_generator, device=device),
                    torch.randn(faded.shape, generator=noise_generator, device=device),
                ) * math.sqrt(noise_variance / 2)
                received_waveform = faded + noise
                received_grid = demodulate_tti(received_waveform, profile)
                h_hat = estimate_comb_dft_ls(
                    received_grid, pilots, profile, num_taps=channel["num_taps"],
                    ridge_lambda=channel["estimator_ridge_lambda"],
                )
                h_true_fft = torch.fft.fft(taps, n=profile.n_fft)
                h_true = h_true_fft[:, list(profile.active_fft_bins), None].expand_as(h_hat)
                candidate_received = _extract(received_grid, candidate_mask)
                candidate_h_hat = _extract(h_hat, candidate_mask)
                selected = allocation.selected_candidate_indices.to(device)
                selected_received = candidate_received.index_select(-1, selected)
                selected_h_hat = candidate_h_hat.index_select(-1, selected)
                mapped_estimate, effective_gain = lmmse_source_estimate(
                    selected_received, selected_h_hat, allocation.relative_power,
                    noise_variance=noise_variance, source_power=source_power,
                )
                decoder_input = torch.empty_like(mapped_estimate)
                decoder_input[..., allocation.resource_to_source.to(device)] = mapped_estimate
                rx_state = build_observable_receiver_state_v1(
                    received_grid, pilots, masks.pilot, h_hat
                )
                with torch.no_grad():
                    reconstruction = model.decoder(decoder_input, rx_state)
                    decoded = codec.decode_representation(reconstruction)
                report = PhysicalCSIReport.from_reliability(
                    tti, _extract(h_hat.abs().square(), candidate_mask)[0].cpu()
                )
                residual = decoder_input - source
                post_sinr = source.abs().square().sum() / residual.abs().square().sum().clamp_min(1e-12)
                layer = per_layer_nmse(reconstruction, target)
                summed = summed_latent_statistics(reconstruction, target)
                clean_metrics = waveform_metrics(
                    waveform, clean_audio, int(config["codec"]["sample_rate"])
                )
                current_metrics = waveform_metrics(
                    waveform, decoded, int(config["codec"]["sample_rate"])
                )
                raw_pilot = received_grid[masks.pilot[None]] / pilots[masks.pilot[None]]
                estimated_pilot = h_hat[masks.pilot[None]]
                pilot_reference = (h_hat * pilots)[masks.pilot[None]]
                pilot_residual = (received_grid - h_hat * pilots)[masks.pilot[None]]
                selected_reliability = (
                    report.reliability[allocation.selected_candidate_indices.cpu()]
                )
                power = allocation.relative_power
                data_values = tx_grid[:, candidate_mask]
                row = {
                    "profile": profile.name, "snr_db": snr_db, "realization": realization,
                    "tti": tti, "bootstrap_uniform": tti == 0, "utterance_id": str(path),
                    "channel_hash": _hash(taps), "noise_hash": _hash(noise),
                    "allocation_hash": _hash(allocation.selected_candidate_indices),
                    "source_symbol_power": source_power,
                    "allocated_data_re_power": float(data_values.abs().square().sum() / 1920),
                    "fixed_data_energy_budget": float(
                        allocation.relative_power.sum() * source_power
                    ),
                    "realized_data_energy": float(data_values.abs().square().sum()),
                    "pilot_power": 1.0,
                    "total_data_energy": float(
                        allocation.relative_power.sum() * source_power
                    ),
                    "total_pilot_energy": float(pilots.abs().square().sum()),
                    "time_domain_tx_power": float(tx_waveform.abs().square().mean()),
                    "awgn_variance": noise_variance,
                    "allocated_power_min": float(power.min()),
                    "allocated_power_max": float(power.max()),
                    "allocated_power_mean": float(power.mean()),
                    "papr": float(data_values.abs().square().max() / data_values.abs().square().mean()),
                    "fraction_at_minimum": float((power <= spec["allocation"]["minimum_relative_power"] + 1e-6).float().mean()),
                    "fraction_at_maximum": float((power >= spec["allocation"]["maximum_relative_power"] - 1e-6).float().mean()),
                    "selected_reliability_mean": float(selected_reliability.mean()),
                    "power_to_worst_selected_decile": float(
                        power[
                            torch.topk(
                                selected_reliability,
                                max(1, selected_reliability.numel() // 10),
                                largest=False,
                            ).indices
                        ].mean()
                    ),
                    "effective_gain_mean": float(effective_gain.abs().mean()),
                    "post_mmse_sinr_db": float(10 * torch.log10(post_sinr)),
                    "aggregate_layer_nmse": float(layer.mean()),
                    "per_layer_nmse": [float(value) for value in layer],
                    "summed_latent_nmse": float(summed["nmse"]),
                    "summed_latent_snr_db": float(summed["snr_db"]),
                    "si_sdr_db": current_metrics["si_sdr_db"],
                    "delta_si_sdr_db": current_metrics["si_sdr_db"] - clean_metrics["si_sdr_db"],
                    "waveform_snr_db": current_metrics["waveform_snr_db"],
                    "delta_waveform_snr_db": current_metrics["waveform_snr_db"] - clean_metrics["waveform_snr_db"],
                    "stft_ratio": current_metrics["stft_l1"] / max(clean_metrics["stft_l1"], 1e-12),
                    "csi_nmse": float((h_hat - h_true).abs().square().mean() / h_true.abs().square().mean()),
                    "pilot_evm": float((pilot_residual.abs().square().sum() / pilot_reference.abs().square().sum()).sqrt()),
                    "interpolation_error": float((estimated_pilot - raw_pilot).abs().square().mean() / raw_pilot.abs().square().mean().clamp_min(1e-12)),
                }
                rows.append(row)
            trajectory_rows.append({
                "snr_db": snr_db, "realization": realization,
                "configured_rho": rho,
                "measured_rho": measured_lag1_correlation(taps_cpu),
            })

    by_snr = {}
    for snr_db in map(float, channel["snr_db"]):
        members = [row for row in rows if row["snr_db"] == snr_db and row["tti"] > 0]
        by_snr[str(snr_db)] = _summary(members)
    five = by_snr["5.0"]["post_mmse_sinr_db"]
    profile_report = {
        "profile_name": profile.name,
        "carrier_frequency_hz": physical["carrier_frequency_hz"],
        "subcarrier_spacing_hz": profile.subcarrier_spacing_hz,
        "fft_size": profile.n_fft,
        "sample_rate_hz": profile.sample_rate_hz,
        "active_bins": list(profile.active_fft_bins),
        "guard_bins": list(profile.guard_fft_bins),
        "dc_bin": 0,
        "occupied_bandwidth_hz": profile.occupied_bandwidth_hz,
        "cp_samples": profile.cp_samples,
        "cp_duration_s": profile.cp_duration_s,
        "useful_symbol_duration_s": profile.useful_symbol_duration_s,
        "ofdm_symbol_duration_s": profile.ofdm_symbol_duration_s,
        "symbols_per_tti": profile.n_ofdm_symbols,
        "tti_duration_s": profile.tti_duration_s,
        "doppler_frequency_hz": fd,
        "configured_lag1_correlation": rho,
        "measured_lag1_correlation": _mean(trajectory_rows, "measured_rho"),
        "feedback_delay_ttis": 1,
        "feedback_delay_seconds": profile.tti_duration_s,
        "pilot_symbol_indices": list(profile.pilot_symbol_indices),
        "pilot_re": profile.pilot_re,
        "candidate_data_re": profile.candidate_data_re,
        "active_source_re": 1920,
        "unused_candidate_re": profile.candidate_data_re - 1920,
        "guard_and_dc_re": (profile.n_fft - profile.active_subcarriers)
        * profile.n_ofdm_symbols,
        "total_null_re": (profile.n_fft - profile.active_subcarriers)
        * profile.n_ofdm_symbols
        + profile.candidate_data_re
        - 1920,
        "active_utilization_ratio": 1920 / profile.candidate_data_re,
        "pilot_overhead_ratio": profile.pilot_re / profile.total_active_re,
        "fixed_total_data_energy_budget": 1920.0,
        "pilot_energy_per_tti": float(profile.pilot_re),
    }
    summary = {
        "checkpoint": args.checkpoint,
        "profile": profile_report,
        "utterances": args.utterances,
        "realizations": args.realizations,
        "by_snr_excluding_tti0": by_snr,
        "five_db_target": {
            "threshold_db": 8.0, "measured_db": five,
            "passed": five >= 8.0, "remaining_gap_db": max(0.0, 8.0 - five),
        },
        "jammer_unblocked": False,
        "training_performed": False,
    }
    _write_csv(output / "per_sample_metrics.csv", rows)
    _write_csv(output / "trajectory_metrics.csv", trajectory_rows)
    (output / "physical_profile.json").write_text(json.dumps(profile_report, indent=2))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
