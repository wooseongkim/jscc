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
    NR_LIKE_R3,
    NR_LIKE_R4,
    active_grid_masks,
    apply_tti_multipath,
    demodulate_tti,
    estimate_comb_dft_ls,
    insert_physical_pilots,
    modulate_tti,
)
from channels.global_triplet_allocator import (
    GlobalTripletCSIReport,
    allocate_global_balanced_triplets,
)
from channels.repetition_mrc import (
    RepetitionCSIReport,
    allocate_repetition3,
    coherent_mrc,
    oracle_branch_sinr,
)
from channels.temporal_multipath import (
    correlated_tap_trajectory,
    delay_samples_for_rate,
    expand_taps_to_sample_delays,
    jakes_slot_correlation,
    measured_lag1_correlation,
)
from models.observable_channel_state import build_observable_receiver_state_v1
from speech_jscc.config import resolve_device
from speech_jscc.experiment import build_components
from speech_jscc.training.channel_free_revalidation import (
    per_layer_nmse,
    summed_latent_statistics,
)
from speech_jscc.training.r4_waveform_finetune import r4_physical_layer_forward
from src.evaluation.waveform_metrics import waveform_metrics
from train_channel_free_conv_conformer import fixed_paths, load_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ofdm_nr_like_r3_repetition3_mrc.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--utterances", type=int)
    parser.add_argument("--realizations", type=int)
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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double() - a.double().mean()
    b = b.double() - b.double().mean()
    return float((a * b).sum() / (a.square().sum() * b.square().sum()).sqrt().clamp_min(1e-12))


def _aggregate(rows: list[dict]) -> dict:
    keys = (
        "branch1_sinr_db", "branch2_sinr_db", "branch3_sinr_db",
        "theoretical_combined_sinr_db", "oracle_empirical_sinr_db",
        "post_combining_sinr_db", "oracle_empirical_theory_mismatch_db",
        "gain_over_strongest_branch_db", "gain_over_mean_branch_db",
        "aggregate_layer_nmse", "summed_latent_nmse", "summed_latent_snr_db",
        "si_sdr_db", "delta_si_sdr_db", "waveform_snr_db",
        "delta_waveform_snr_db", "stft_ratio", "csi_nmse", "pilot_evm",
        "delayed_current_csi_correlation", "weight_max_mean",
        "one_branch_over_0_8_fraction", "all_branches_over_0_1_fraction",
        "total_data_energy", "triplet_gain_min", "triplet_gain_p05",
        "triplet_gain_mean", "triplet_gain_worst_decile",
        "minimum_frequency_separation", "triplet_q_min", "triplet_q_max",
        "branch_fraction_min",
    )
    return {key: _mean(rows, key) for key in keys}


def _gain_stats(values: torch.Tensor) -> dict:
    ordered = torch.sort(values.double()).values
    return {
        "minimum": float(ordered[0]),
        "p05": float(torch.quantile(ordered, .05)),
        "median": float(torch.quantile(ordered, .5)),
        "mean": float(ordered.mean()),
        "standard_deviation": float(ordered.std()),
        "coefficient_of_variation": float(ordered.std() / ordered.mean().clamp_min(1e-12)),
        "worst_decile_mean": float(ordered[: max(1, ordered.numel() // 10)].mean()),
    }


def _write_r4_validation(output: Path, spec: dict, physical_spec: dict) -> dict:
    pdp = exponential_pdp(
        physical_spec["channel"]["num_taps"], physical_spec["channel"]["pdp_decay"]
    )
    coefficients = correlated_tap_trajectory(
        slots=2, batch_size=1, pdp=pdp, rho=.98, seed=27041
    )[0]
    delay_seconds = tuple(physical_spec["channel"]["tap_delay_seconds"])
    reports = {}
    for profile in (NR_LIKE_R3, NR_LIKE_R4):
        delays = delay_samples_for_rate(delay_seconds, profile.sample_rate_hz)
        sparse = expand_taps_to_sample_delays(coefficients, delays)
        response = torch.fft.fft(sparse, n=profile.n_fft)[
            :, list(profile.active_fft_bins)
        ].abs().square()[0, :, None].expand(
            profile.active_subcarriers, profile.n_ofdm_symbols
        )
        reports[profile.name] = response[active_grid_masks(profile).candidate_data]
    old = allocate_repetition3(
        profile=NR_LIKE_R3, tx_tti=1,
        report=RepetitionCSIReport.from_reliability(0, reports["nr_like_r3"]),
        layer_importance_order=spec["allocation"]["layer_importance_order"],
    )
    old_destination = reports["nr_like_r3"][old.selected_candidate_indices].sum(0)
    old_gain = torch.empty_like(old_destination)
    old_gain[old.resource_to_source] = old_destination
    new = allocate_global_balanced_triplets(
        profile=NR_LIKE_R4, tx_tti=1,
        report=GlobalTripletCSIReport.from_reliability(0, reports["nr_like_r4"]),
        layer_importance_order=spec["allocation"]["layer_importance_order"],
        min_selected_re_per_subcarrier=spec["allocation"]["min_selected_re_per_subcarrier"],
        max_selected_re_per_subcarrier=spec["allocation"]["max_selected_re_per_subcarrier"],
        minimum_frequency_separation_subcarriers=spec["allocation"]["minimum_frequency_separation_subcarriers"],
        q_min=spec["power"]["q_min"], q_max=spec["power"]["q_max"],
        branch_alpha=spec["power"]["branch_alpha"],
        branch_min_fraction=spec["power"]["branch_min_fraction"],
    )
    report = {
        "paired_coefficient_seed": 27041,
        "r3_fixed_three_bands": {
            "candidate_utilization": 5760 / NR_LIKE_R3.candidate_data_re,
            "triplet_gain": _gain_stats(old_gain),
            "unused_candidate_re": old.unused_candidate_re,
            "total_power": float(old.power_source_order.sum()),
        },
        "r4_global_balanced_triplets": {
            "candidate_utilization": 5760 / NR_LIKE_R4.candidate_data_re,
            "triplet_gain_before_refinement": _gain_stats(new.before_triplet_gain),
            "triplet_gain": _gain_stats(new.predicted_triplet_gain),
            "unused_candidate_re": new.unused_candidate_re,
            "minimum_frequency_separation": int(new.separation_levels.min()),
            "total_power": float(new.power_source_order.sum()),
            "triplet_power": _gain_stats(new.triplet_power_multiplier),
        },
    }
    report["validation_pass"] = (
        report["r4_global_balanced_triplets"]["triplet_gain"]["p05"]
        > report["r3_fixed_three_bands"]["triplet_gain"]["p05"]
        and abs(report["r4_global_balanced_triplets"]["total_power"] - 5760) < 2e-3
        and report["r4_global_balanced_triplets"]["minimum_frequency_separation"] >= 60
    )
    directory = output.parent / "allocation_validation"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(report, indent=2))
    if not report["validation_pass"]:
        raise RuntimeError("R4 allocation validation failed; refusing full evaluation")
    return report


def main() -> None:
    args = parse_args()
    spec = yaml.safe_load(Path(args.config).read_text())
    if args.checkpoint:
        spec["checkpoint"] = args.checkpoint
    physical_spec = yaml.safe_load(Path(spec["profile_config"]).read_text())
    utterances = args.utterances or int(spec["evaluation"]["utterances"])
    realizations = args.realizations or int(spec["evaluation"]["realizations"])
    if args.dry_run:
        print(json.dumps({
            "profile": physical_spec["profile"]["name"], "copies": 3, "combiner": "coherent_mrc",
            "energy_contract": spec.get("energy", {}).get(
                "contract", "fixed_power_per_copy_equivalent"
            ), "utterances": utterances,
            "realizations": realizations, "output_dir": args.output_dir,
            "training": False, "jammer": False,
        }, indent=2))
        return
    if utterances > 2 and not args.allow_long_run:
        raise SystemExit("full repetition/MRC evaluation requires --allow-long-run")
    output = Path(args.output_dir)
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"refusing existing output directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    allocation_validation = (
        _write_r4_validation(output, spec, physical_spec)
        if physical_spec["profile"]["name"] == "nr_like_r4"
        else None
    )

    saved = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    model_config = saved["config"]
    model_config["device"] = args.device
    device = resolve_device(args.device)
    codec, model = build_components(model_config, device)
    model.load_state_dict(saved["model"], strict=True)
    codec.eval().requires_grad_(False)
    model.eval()
    _, paths = fixed_paths(model_config, int(model_config["seed"]))
    paths = paths[:utterances]
    allocation_mode = spec.get("allocation", {}).get("mode", "fixed_three_bands")
    profile = NR_LIKE_R4 if physical_spec["profile"]["name"] == "nr_like_r4" else NR_LIKE_R3
    r4_mode = allocation_mode == "global_balanced_triplets"
    delay_seconds = tuple(
        physical_spec["channel"].get(
            "tap_delay_seconds",
            [index / profile.sample_rate_hz for index in range(physical_spec["channel"]["num_taps"])],
        )
    )
    delay_samples = delay_samples_for_rate(delay_seconds, profile.sample_rate_hz)
    if max(delay_samples) >= profile.cp_samples:
        raise AssertionError("physical channel delay exceeds R4 cyclic prefix")
    masks = active_grid_masks(profile, device=device)
    pdp = exponential_pdp(
        physical_spec["channel"]["num_taps"], physical_spec["channel"]["pdp_decay"]
    )
    rho = jakes_slot_correlation(
        physical_spec["physical"]["user_speed_mps"],
        physical_spec["physical"]["carrier_frequency_hz"],
        profile.tti_duration_s,
    )
    rows: list[dict] = []
    trajectories: list[dict] = []
    importance = spec["allocation"]["layer_importance_order"]

    for snr_db in map(float, spec["evaluation"]["snr_db"]):
        for realization in range(realizations):
            seed = int(model_config["seed"]) + round(snr_db * 1000) + realization * 100_003
            permutation = torch.randperm(
                utterances, generator=torch.Generator().manual_seed(seed + 11)
            ).tolist()
            taps_cpu = correlated_tap_trajectory(
                slots=utterances, batch_size=1, pdp=pdp, rho=rho, seed=seed + 17
            )
            noise_generator = torch.Generator(device=device).manual_seed(seed + 23)
            report: RepetitionCSIReport | GlobalTripletCSIReport | None = None
            for tti, utterance_index in enumerate(permutation):
                path = paths[utterance_index]
                waveform = load_batch([path], model_config, device)
                with torch.no_grad():
                    target = codec.encode_waveform(waveform)
                    state = target.new_zeros((1, model.encoder.channel_state_dim))
                    source = model.encoder(target, state)
                    clean = codec.decode_representation(target)
                allocation_report = report
                if r4_mode:
                    allocation = allocate_global_balanced_triplets(
                        profile=profile, tx_tti=tti, report=allocation_report,
                        layer_importance_order=importance,
                        min_selected_re_per_subcarrier=spec["allocation"]["min_selected_re_per_subcarrier"],
                        max_selected_re_per_subcarrier=spec["allocation"]["max_selected_re_per_subcarrier"],
                        minimum_frequency_separation_subcarriers=spec["allocation"]["minimum_frequency_separation_subcarriers"],
                        q_min=spec["power"]["q_min"], q_max=spec["power"]["q_max"],
                        branch_alpha=spec["power"]["branch_alpha"],
                        branch_min_fraction=spec["power"]["branch_min_fraction"],
                    )
                else:
                    allocation = allocate_repetition3(
                        profile=profile, tx_tti=tti, report=allocation_report,
                        layer_importance_order=importance,
                        energy_contract=spec["energy"]["contract"],
                        alpha=spec["allocation"]["alpha"],
                        minimum_relative_power=spec["allocation"]["minimum_relative_power"],
                        maximum_relative_power=spec["allocation"]["maximum_relative_power"],
                    )
                tap_coefficients = taps_cpu[tti].to(device)
                taps = expand_taps_to_sample_delays(tap_coefficients, delay_samples)
                if r4_mode:
                    physical_forward = r4_physical_layer_forward(
                        source, allocation, taps, snr_db=snr_db,
                        noise_generator=noise_generator,
                        tap_delay_samples=delay_samples,
                        estimator_num_taps=physical_spec["channel"]["num_taps"],
                        estimator_ridge_lambda=physical_spec["channel"]["estimator_ridge_lambda"],
                        epsilon=spec["repetition"]["epsilon"],
                    )
                    data_grid = physical_forward.data_grid
                    pilots = physical_forward.pilots
                    noise = physical_forward.noise
                    received_grid = physical_forward.received_grid
                    h_hat_grid = physical_forward.estimated_channel
                    h_true_grid = physical_forward.true_channel
                    h_true = physical_forward.true_channel_source_order
                    combined = physical_forward.combined
                    oracle = physical_forward.oracle_combined
                    rx_state = physical_forward.decoder_state
                    noise_variance = physical_forward.noise_variance
                    source_power = float(physical_forward.source_power)
                else:
                    data_grid = allocation.place(source)
                    tx_grid, pilots = insert_physical_pilots(data_grid, profile)
                    tx_waveform = modulate_tti(tx_grid, profile)
                    faded = apply_tti_multipath(tx_waveform, taps, profile)
                    source_power = float(source.abs().square().mean())
                    noise_variance = source_power / 10 ** (snr_db / 10)
                    noise = torch.complex(
                        torch.randn(faded.shape, generator=noise_generator, device=device),
                        torch.randn(faded.shape, generator=noise_generator, device=device),
                    ) * math.sqrt(noise_variance / 2)
                    received_grid = demodulate_tti(faded + noise, profile)
                    h_hat_grid = estimate_comb_dft_ls(
                        received_grid, pilots, profile,
                        num_taps=physical_spec["channel"]["num_taps"],
                        tap_delay_samples=delay_samples,
                        ridge_lambda=physical_spec["channel"]["estimator_ridge_lambda"],
                    )
                    h_true_fft = torch.fft.fft(taps, n=profile.n_fft)
                    h_true_grid = h_true_fft[:, list(profile.active_fft_bins), None].expand_as(h_hat_grid)
                    raw = allocation.extract_source_order(received_grid)
                    h_hat = allocation.extract_source_order(h_hat_grid)
                    h_true = allocation.extract_source_order(h_true_grid)
                    power = allocation.power_source_order
                    combined = coherent_mrc(
                        raw, h_hat, power, noise_variance,
                        source_power=source_power, epsilon=spec["repetition"]["epsilon"],
                    )
                    oracle = coherent_mrc(
                        raw, h_true, power, noise_variance,
                        source_power=source_power, epsilon=spec["repetition"]["epsilon"],
                    )
                    rx_state = build_observable_receiver_state_v1(
                        received_grid, pilots, masks.pilot, h_hat_grid
                    )
                power = allocation.power_source_order
                if combined.estimate.shape != source.shape:
                    raise AssertionError("MRC did not restore the 1920-symbol decoder interface")
                with torch.no_grad():
                    reconstruction = model.decoder(combined.estimate, rx_state)
                    decoded = codec.decode_representation(reconstruction)
                current_reliability = _extract(
                    h_hat_grid.abs().square(), masks.candidate_data
                )[0].cpu()
                delayed_corr = 1.0 if report is None else _corr(
                    report.reliability, current_reliability
                )
                report_type = GlobalTripletCSIReport if r4_mode else RepetitionCSIReport
                report = report_type.from_reliability(tti, current_reliability)
                branch_sinr = oracle_branch_sinr(
                    h_true, power, noise_variance, source_power=source_power
                )
                source_energy = source.abs().square().sum()
                branch_mean = torch.stack([
                    source_energy
                    / (float(source_power) / branch_sinr[:, branch]).sum()
                    for branch in range(3)
                ])
                theoretical = source_energy / (
                    float(source_power) / branch_sinr.sum(dim=1)
                ).sum()
                oracle_empirical = (
                    source.abs().square().sum()
                    / (oracle.estimate - source).abs().square().sum()
                )
                empirical = (
                    source.abs().square().sum()
                    / (combined.estimate - source).abs().square().sum()
                )
                layer = per_layer_nmse(reconstruction, target)
                summed = summed_latent_statistics(reconstruction, target)
                clean_metrics = waveform_metrics(
                    waveform, clean, int(model_config["codec"]["sample_rate"])
                )
                current_metrics = waveform_metrics(
                    waveform, decoded, int(model_config["codec"]["sample_rate"])
                )
                csi_error = (
                    (h_hat_grid - h_true_grid).abs().square().mean()
                    / h_true_grid.abs().square().mean()
                )
                pilot_residual = (received_grid - h_hat_grid * pilots)[masks.pilot[None]]
                pilot_reference = (h_hat_grid * pilots)[masks.pilot[None]]
                weights = combined.weights
                selected_delayed = torch.ones(3, 1920) if tti == 0 else torch.stack([
                    allocation_report.reliability[allocation.selected_candidate_indices[b].cpu()]
                    for b in range(3)
                ])
                transmitted_source_order = allocation.extract_source_order(data_grid)
                branch_energy = transmitted_source_order.abs().square().sum(dim=-1)[0]
                branch_papr = (
                    transmitted_source_order.abs().square().amax(dim=-1)
                    / transmitted_source_order.abs().square().mean(dim=-1)
                )[0]
                total_papr = float(
                    transmitted_source_order.abs().square().max()
                    / transmitted_source_order.abs().square().mean()
                )
                energy_by_layer = [
                    float(
                        transmitted_source_order[
                            ..., layer_index * 240 : (layer_index + 1) * 240
                        ].abs().square().sum()
                    )
                    for layer_index in range(8)
                ]
                delayed_source_order = torch.empty_like(selected_delayed)
                delayed_source_order[:, allocation.resource_to_source] = selected_delayed
                reliability_by_layer = [
                    float(
                        delayed_source_order[
                            :, layer_index * 240 : (layer_index + 1) * 240
                        ].mean()
                    )
                    for layer_index in range(8)
                ]
                row = {
                    "snr_db": snr_db, "realization": realization, "tti": tti,
                    "utterance_id": str(path), "bootstrap_uniform": tti == 0,
                    "channel_hash": _hash(taps), "noise_hash": _hash(noise),
                    "mapping_hash": _hash(allocation.selected_candidate_indices),
                    "branch1_sinr_db": float(10 * torch.log10(branch_mean[0])),
                    "branch2_sinr_db": float(10 * torch.log10(branch_mean[1])),
                    "branch3_sinr_db": float(10 * torch.log10(branch_mean[2])),
                    "theoretical_combined_sinr_db": float(10 * torch.log10(theoretical)),
                    "oracle_empirical_sinr_db": float(10 * torch.log10(oracle_empirical)),
                    "post_combining_sinr_db": float(10 * torch.log10(empirical)),
                    "oracle_empirical_theory_mismatch_db": float(
                        10 * torch.log10(oracle_empirical / theoretical)
                    ),
                    "gain_over_strongest_branch_db": float(
                        10 * torch.log10(empirical) - 10 * torch.log10(branch_mean.max())
                    ),
                    "gain_over_mean_branch_db": float(
                        10 * torch.log10(empirical) - 10 * torch.log10(branch_mean.mean())
                    ),
                    "csi_nmse": float(csi_error),
                    "pilot_evm": float(
                        (pilot_residual.abs().square().sum() / pilot_reference.abs().square().sum()).sqrt()
                    ),
                    "delayed_current_csi_correlation": delayed_corr,
                    "selected_reliability_copy1": float(selected_delayed[0].mean()),
                    "selected_reliability_copy2": float(selected_delayed[1].mean()),
                    "selected_reliability_copy3": float(selected_delayed[2].mean()),
                    "weight_copy1_mean": float(weights[:, 0].mean()),
                    "weight_copy2_mean": float(weights[:, 1].mean()),
                    "weight_copy3_mean": float(weights[:, 2].mean()),
                    "weight_max_mean": float(weights.max(dim=1).values.mean()),
                    "weight_p05": float(torch.quantile(weights, .05)),
                    "weight_p50": float(torch.quantile(weights, .50)),
                    "weight_p95": float(torch.quantile(weights, .95)),
                    "one_branch_over_0_8_fraction": float((weights.max(dim=1).values > .8).float().mean()),
                    "all_branches_over_0_1_fraction": float((weights.min(dim=1).values >= .1).float().mean()),
                    "copy1_power_min": float(power[0].min()), "copy1_power_max": float(power[0].max()),
                    "copy1_power_mean": float(power[0].mean()),
                    "copy2_power_min": float(power[1].min()), "copy2_power_max": float(power[1].max()),
                    "copy2_power_mean": float(power[1].mean()),
                    "copy3_power_min": float(power[2].min()), "copy3_power_max": float(power[2].max()),
                    "copy3_power_mean": float(power[2].mean()),
                    "copy1_power_at_min_fraction": float((power[0] <= .500001).float().mean()),
                    "copy1_power_at_max_fraction": float((power[0] >= 1.999999).float().mean()),
                    "copy2_power_at_min_fraction": float((power[1] <= .500001).float().mean()),
                    "copy2_power_at_max_fraction": float((power[1] >= 1.999999).float().mean()),
                    "copy3_power_at_min_fraction": float((power[2] <= .500001).float().mean()),
                    "copy3_power_at_max_fraction": float((power[2] >= 1.999999).float().mean()),
                    "copy1_papr": float(branch_papr[0]),
                    "copy2_papr": float(branch_papr[1]),
                    "copy3_papr": float(branch_papr[2]),
                    "total_packet_papr": total_papr,
                    "copy1_energy": float(branch_energy[0]),
                    "copy2_energy": float(branch_energy[1]),
                    "copy3_energy": float(branch_energy[2]),
                    "total_data_energy": float(power.sum() * source_power),
                    "energy_increase_db": 10 * math.log10(3),
                    "unused_candidate_re": allocation.unused_candidate_re,
                    "triplet_gain_min": float(allocation.predicted_triplet_gain.min()) if r4_mode else float(selected_delayed.sum(0).min()),
                    "triplet_gain_p05": float(torch.quantile(allocation.predicted_triplet_gain, .05)) if r4_mode else float(torch.quantile(selected_delayed.sum(0), .05)),
                    "triplet_gain_mean": float(allocation.predicted_triplet_gain.mean()) if r4_mode else float(selected_delayed.sum(0).mean()),
                    "triplet_gain_worst_decile": float(torch.sort(allocation.predicted_triplet_gain).values[:192].mean()) if r4_mode else float(torch.sort(selected_delayed.sum(0)).values[:192].mean()),
                    "minimum_frequency_separation": float(allocation.separation_levels.min()) if r4_mode else 72.0,
                    "triplet_q_min": float(allocation.triplet_power_multiplier.min()) if r4_mode else 1.0,
                    "triplet_q_max": float(allocation.triplet_power_multiplier.max()) if r4_mode else 1.0,
                    "branch_fraction_min": float(allocation.branch_power_fractions.min()) if r4_mode else 1 / 3,
                    "aggregate_layer_nmse": float(layer.mean()),
                    "per_layer_nmse": [float(value) for value in layer],
                    "energy_by_layer": energy_by_layer,
                    "selected_reliability_by_layer": reliability_by_layer,
                    "summed_latent_nmse": float(summed["nmse"]),
                    "summed_latent_snr_db": float(summed["snr_db"]),
                    "si_sdr_db": current_metrics["si_sdr_db"],
                    "delta_si_sdr_db": current_metrics["si_sdr_db"] - clean_metrics["si_sdr_db"],
                    "waveform_snr_db": current_metrics["waveform_snr_db"],
                    "delta_waveform_snr_db": current_metrics["waveform_snr_db"] - clean_metrics["waveform_snr_db"],
                    "stft_ratio": current_metrics["stft_l1"] / max(clean_metrics["stft_l1"], 1e-12),
                }
                if not all(
                    math.isfinite(float(value))
                    for value in row.values()
                    if isinstance(value, (float, int)) and not isinstance(value, bool)
                ):
                    raise FloatingPointError(f"nonfinite sample: {row}")
                rows.append(row)
            trajectories.append({
                "snr_db": snr_db, "realization": realization,
                "configured_rho": rho, "measured_rho": measured_lag1_correlation(taps_cpu),
            })
    by_snr = {}
    for snr_db in map(float, spec["evaluation"]["snr_db"]):
        members = [row for row in rows if row["snr_db"] == snr_db and row["tti"] > 0]
        by_snr[str(snr_db)] = _aggregate(members)
    five = by_snr["5.0"]
    gates = spec["gates"]
    oracle_ok = all(
        abs(row["oracle_empirical_theory_mismatch_db"])
        <= gates["oracle_empirical_theory_mismatch_max_db"]
        for row in rows
    )
    summary = {
        "checkpoint": spec["checkpoint"], "profile": profile.name,
        "allocation_mode": allocation_mode,
        "copies": 3, "candidate_data_re": profile.candidate_data_re,
        "active_data_re": 5760, "unused_candidate_re": profile.candidate_data_re - 5760,
        "energy_contract": spec.get("energy", {}).get("contract", "fixed_power_per_copy_equivalent"),
        "total_packet_energy": 5760, "energy_increase_db": 10 * math.log10(3),
        "tap_delay_seconds": delay_seconds, "tap_delay_samples": delay_samples,
        "allocation_validation": allocation_validation,
        "utterances": utterances, "realizations": realizations,
        "configured_rho": rho, "measured_rho": _mean(trajectories, "measured_rho"),
        "by_snr_excluding_tti0": by_snr,
        "oracle_validation_pass": oracle_ok,
        "five_db_gate": {
            "post_combining_sinr_pass": five["post_combining_sinr_db"]
            >= gates["five_db_post_combining_sinr_min_db"],
            "fallback_sinr_pass": five["post_combining_sinr_db"]
            >= gates.get(
                "fallback_post_combining_sinr_min_db",
                gates["five_db_post_combining_sinr_min_db"],
            ),
            "delta_si_sdr_pass": five["delta_si_sdr_db"] >= gates["delta_si_sdr_min_db"],
            "delta_waveform_snr_pass": five["delta_waveform_snr_db"]
            >= gates["delta_waveform_snr_min_db"],
            "stft_pass": five["stft_ratio"] <= gates["stft_ratio_max"],
        },
        "nonfinite_samples": 0, "jammer_unblocked": False, "training_performed": False,
    }
    summary["clean_channel_gate_pass"] = (
        oracle_ok and all(summary["five_db_gate"].values())
    )
    _write_csv(output / "per_sample_metrics.csv", rows)
    _write_jsonl(output / "per_sample_details.jsonl", rows)
    _write_csv(output / "trajectory_metrics.csv", trajectories)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
