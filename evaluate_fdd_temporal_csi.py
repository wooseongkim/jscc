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
from channels.pilot import (
    csi_nmse,
    estimate_channel_ls,
    extract_data_resources,
    insert_data_and_pilots,
    make_pilot_mask,
    pilot_evm,
)
from channels.temporal_multipath import (
    correlated_tap_trajectory,
    doppler_frequency_hz,
    iid_tap_trajectory,
    jakes_slot_correlation,
    measured_lag1_correlation,
    taps_to_slot_frequency_response,
)
from speech_jscc.config import resolve_device
from speech_jscc.diagnostics.fdd_temporal_csi import (
    CSIReport,
    DelayedCSIBuffer,
    allocate_from_current_oracle,
    allocate_from_report,
    apply_resource_map,
    deterministic_interleaver,
    invert_resource_map,
    mmse_equalize,
)
from speech_jscc.experiment import build_components
from speech_jscc.training.channel_free_revalidation import (
    per_layer_nmse,
    summed_latent_statistics,
)
from src.evaluation.waveform_metrics import waveform_metrics
from train_channel_free_conv_conformer import fixed_paths, load_batch


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fdd_temporal_csi.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--utterances", type=int)
    parser.add_argument("--realizations", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-long-run", action="store_true")
    return parser.parse_args()


def _hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _rank(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float64, device=values.device)
    return ranks


def _correlations(previous: torch.Tensor, current: torch.Tensor) -> tuple[float, float]:
    def corr(a: torch.Tensor, b: torch.Tensor) -> float:
        a, b = a.double() - a.double().mean(), b.double() - b.double().mean()
        return float((a * b).sum() / (a.square().sum() * b.square().sum()).sqrt().clamp_min(1e-12))
    return corr(previous, current), corr(_rank(previous), _rank(current))


def _top_overlap(previous: torch.Tensor, current: torch.Tensor, fraction: float) -> float:
    count = max(1, round(previous.numel() * fraction))
    a = set(torch.topk(previous, count).indices.tolist())
    b = set(torch.topk(current, count).indices.tolist())
    return len(a & b) / count


def _mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _summarize(rows: list[dict]) -> dict:
    keys = (
        "post_mmse_sinr_db", "aggregate_layer_nmse", "summed_latent_nmse",
        "summed_latent_snr_db", "si_sdr_db", "delta_si_sdr_db",
        "waveform_snr_db", "delta_waveform_snr_db", "stft_ratio",
        "csi_nmse", "pilot_evm", "allocation_reliability_metric",
    )
    return {key: _mean(rows, key) for key in keys}


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key, value in row.items() if not isinstance(value, list)})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: value for key, value in row.items() if key in fields} for row in rows])


def _extract_real_data(resources: torch.Tensor, pilot_mask: torch.Tensor) -> torch.Tensor:
    mask = torch.broadcast_to(pilot_mask.to(resources.device, torch.bool), resources.shape)
    return torch.stack([resources[index][~mask[index]] for index in range(resources.shape[0])])


def main() -> None:
    args = arguments()
    spec = yaml.safe_load(Path(args.config).read_text())
    checkpoint_path = args.checkpoint or spec["source_checkpoint"]
    output_root = Path(args.output_root or spec["output_root"])
    utterances = args.utterances or int(spec["evaluation"]["utterances"])
    realizations = args.realizations or int(spec["evaluation"]["realizations"])
    if args.dry_run:
        print(json.dumps({"checkpoint": checkpoint_path, "output_root": str(output_root),
                          "utterances": utterances, "realizations": realizations,
                          "modes": ["T0", "T1", "T2", "T3"]}, indent=2))
        return
    if utterances > 2 and not args.allow_long_run:
        raise SystemExit("full FDD evaluation requires --allow-long-run")
    if output_root.exists() and any((output_root / name).exists() for name in (
        "t0_iid_baseline", "t1_correlated_no_allocation",
        "t2_correlated_delayed_csi", "t3_oracle_current_csi_upper_bound",
    )):
        if not args.overwrite:
            raise SystemExit(f"refusing existing FDD evaluation outputs under: {output_root}")
        for name in ("t0_iid_baseline", "t1_correlated_no_allocation",
                     "t2_correlated_delayed_csi", "t3_oracle_current_csi_upper_bound",
                     "paired_temporal_comparison", "temporal_channel_validation"):
            shutil.rmtree(output_root / name, ignore_errors=True)
    output_root.mkdir(parents=True, exist_ok=True)

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = saved["config"]
    config["device"] = args.device
    device = resolve_device(args.device)
    codec, model = build_components(config, device)
    model.load_state_dict(saved["model"], strict=True)
    codec.eval().requires_grad_(False)
    model.eval()
    _, all_paths = fixed_paths(config, int(config["seed"]))
    paths = all_paths[:utterances]

    mobility = spec["mobility"]
    rho = jakes_slot_correlation(
        mobility["user_speed_mps"], mobility["carrier_frequency_hz"], mobility["slot_duration_s"]
    )
    fd = doppler_frequency_hz(mobility["user_speed_mps"], mobility["carrier_frequency_hz"])
    channel_spec = spec["channel"]
    pdp = exponential_pdp(channel_spec["num_taps"], channel_spec["pdp_decay"])
    pilot_mask_cpu = make_pilot_mask((1, 64, 32), 4, time_spacing=4)[0]
    base_map_cpu = deterministic_interleaver(pilot_mask_cpu)
    importance = list(spec["allocation"]["layer_importance_order"])
    all_rows: list[dict] = []
    validation_rows: list[dict] = []

    for snr_db in map(float, channel_spec["snr_db"]):
        noise_power = float(config["model"]["target_power"]) / 10 ** (snr_db / 10)
        for realization in range(realizations):
            seed = int(spec["seed"]) + int(snr_db * 1000) + realization * 100_003
            generator = torch.Generator().manual_seed(seed + 31)
            permutation = torch.randperm(utterances, generator=generator).tolist()
            correlated = correlated_tap_trajectory(
                slots=utterances, batch_size=1, pdp=pdp, rho=rho, seed=seed + 1
            )
            iid = iid_tap_trajectory(
                slots=utterances, batch_size=1, pdp=pdp, seed=seed + 2
            )
            correlated_h = taps_to_slot_frequency_response(
                correlated, subcarriers=64, ofdm_symbols=32
            ).to(device)
            iid_h = taps_to_slot_frequency_response(iid, subcarriers=64, ofdm_symbols=32).to(device)
            noise_gen = torch.Generator(device=device).manual_seed(seed + 3)
            noise = torch.complex(
                torch.randn((utterances, 1, 64, 32), generator=noise_gen, device=device),
                torch.randn((utterances, 1, 64, 32), generator=noise_gen, device=device),
            ) * math.sqrt(noise_power / 2)
            mode_buffers = {"T2": DelayedCSIBuffer(1)}
            delayed_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
            for slot, utterance_index in enumerate(permutation):
                path = paths[utterance_index]
                waveform = load_batch([path], config, device)
                with torch.no_grad():
                    target = codec.encode_waveform(waveform)
                    tx_state = target.new_zeros((1, model.encoder.channel_state_dim))
                    source = model.encoder(target, tx_state)
                    clean = codec.decode_representation(target)
                for mode in ("T0", "T1", "T2", "T3"):
                    h = iid_h[slot] if mode == "T0" else correlated_h[slot]
                    mapping = base_map_cpu
                    report = None
                    if slot > 0 and mode == "T2":
                        report = mode_buffers["T2"].available_for_tx(slot)
                        if report is None:
                            raise AssertionError("one-slot CSI report missing")
                        mapping = allocate_from_report(slot, report, base_map_cpu, importance)
                    elif slot > 0 and mode == "T3":
                        current_rel = _extract_real_data(
                            h.abs().square(), pilot_mask_cpu[None].to(device)
                        )[0].cpu()
                        mapping = allocate_from_current_oracle(
                            slot, current_rel, base_map_cpu, importance
                        )
                    mapped = apply_resource_map(source, mapping.to(device))
                    current_resource_reliability = _extract_real_data(
                        h.abs().square(), pilot_mask_cpu[None].to(device)
                    )[0]
                    importance_weights = torch.empty(8, device=device)
                    for rank, layer in enumerate(importance):
                        importance_weights[layer] = float(8 - rank)
                    destination_layers = mapping.to(device) // 240
                    allocation_metric = (
                        current_resource_reliability * importance_weights[destination_layers]
                    ).mean()
                    grid, pilots = insert_data_and_pilots(mapped, pilot_mask_cpu[None].to(device))
                    received = h * grid + noise[slot]
                    h_hat = estimate_channel_ls(
                        received, pilots, pilot_mask_cpu[None].to(device),
                        fading="multipath_block", channel_estimator="dft_tap_ls",
                        estimator_num_taps=int(channel_spec["num_taps"]),
                        estimator_ridge_lambda=float(channel_spec["estimator_ridge_lambda"]),
                    )
                    equalized = mmse_equalize(
                        received, h_hat, noise_power=noise_power,
                        signal_power=float(config["model"]["target_power"]),
                    )
                    recovered = extract_data_resources(equalized, pilot_mask_cpu[None].to(device))
                    decoder_input = invert_resource_map(recovered, mapping.to(device))
                    # T3 changes allocation only; all modes use current estimated CSI at RX.
                    from models.observable_channel_state import build_observable_receiver_state_v1
                    rx_state = build_observable_receiver_state_v1(
                        received, pilots, pilot_mask_cpu[None].to(device), h_hat
                    )
                    with torch.no_grad():
                        reconstruction = model.decoder(decoder_input, rx_state)
                        decoded = codec.decode_representation(reconstruction)
                    residual = decoder_input - source
                    post_sinr = source.abs().square().sum() / residual.abs().square().sum().clamp_min(1e-12)
                    layers = per_layer_nmse(reconstruction, target)
                    summed = summed_latent_statistics(reconstruction, target)
                    clean_metrics = waveform_metrics(waveform, clean, int(config["codec"]["sample_rate"]))
                    current_metrics = waveform_metrics(waveform, decoded, int(config["codec"]["sample_rate"]))
                    row = {
                        "mode": mode, "snr_db": snr_db, "realization": realization,
                        "slot": slot, "include_slot0": slot == 0, "utterance_id": str(path),
                        "utterance_permutation_index": utterance_index,
                        "channel_hash": _hash(h), "noise_hash": _hash(noise[slot]),
                        "mapping_hash": _hash(mapping), "bootstrap_uniform": slot == 0,
                        "source_symbol_hash": _hash(source),
                        "allocation_reliability_metric": float(allocation_metric),
                        "post_mmse_sinr_db": float(10 * torch.log10(post_sinr)),
                        "aggregate_layer_nmse": float(layers.mean()),
                        "per_layer_nmse": [float(x) for x in layers],
                        "summed_latent_nmse": float(summed["nmse"]),
                        "summed_latent_snr_db": float(summed["snr_db"]),
                        "csi_nmse": float(csi_nmse(h, h_hat).mean()),
                        "pilot_evm": float(pilot_evm(received, pilots, pilot_mask_cpu[None].to(device), h_hat).mean()),
                        "si_sdr_db": current_metrics["si_sdr_db"],
                        "delta_si_sdr_db": current_metrics["si_sdr_db"] - clean_metrics["si_sdr_db"],
                        "waveform_snr_db": current_metrics["waveform_snr_db"],
                        "delta_waveform_snr_db": current_metrics["waveform_snr_db"] - clean_metrics["waveform_snr_db"],
                        "stft_ratio": current_metrics["stft_l1"] / max(clean_metrics["stft_l1"], 1e-12),
                    }
                    all_rows.append(row)
                    if mode == "T2":
                        current_rel = _extract_real_data(
                            h.abs().square(), pilot_mask_cpu[None].to(device)
                        )[0].cpu()
                        if report is not None:
                            delayed_pairs.append((report.reliability, current_rel))
                        generated = CSIReport.from_reliability(
                            slot,
                            _extract_real_data(
                                h_hat.abs().square(), pilot_mask_cpu[None].to(device)
                            )[0].cpu(),
                        )
                        mode_buffers["T2"].submit(generated)
            pearson = [_correlations(a, b)[0] for a, b in delayed_pairs]
            spearman = [_correlations(a, b)[1] for a, b in delayed_pairs]
            validation_rows.append({
                "snr_db": snr_db, "realization": realization,
                "configured_rho": rho, "measured_rho": measured_lag1_correlation(correlated),
                "delayed_current_pearson": sum(pearson) / len(pearson),
                "delayed_current_spearman": sum(spearman) / len(spearman),
                **{f"top_{int(f*100)}_overlap": sum(_top_overlap(a, b, f) for a, b in delayed_pairs) / len(delayed_pairs)
                   for f in (.10, .25, .50)},
            })

    names = {
        "T0": "t0_iid_baseline", "T1": "t1_correlated_no_allocation",
        "T2": "t2_correlated_delayed_csi", "T3": "t3_oracle_current_csi_upper_bound",
    }
    summary: dict = {
        "checkpoint": checkpoint_path,
        "mobility": {**mobility, "doppler_frequency_hz": fd, "configured_rho_slot": rho,
                     "feedback_delay_slots": 1,
                     "feedback_delay_seconds": mobility["slot_duration_s"]},
        "layer_importance": {"order": importance, "source": spec["allocation"]["importance_source"],
                             "status": spec["allocation"]["importance_evidence_status"]},
        "slot0_policy": "uniform_bootstrap",
        "modes": {},
    }
    for mode, directory in names.items():
        mode_rows = [row for row in all_rows if row["mode"] == mode]
        path = output_root / directory
        path.mkdir(parents=True)
        _write_csv(path / "per_sample_metrics.csv", mode_rows)
        summary["modes"][mode] = {
            "including_slot0": _summarize(mode_rows),
            "excluding_slot0": _summarize([row for row in mode_rows if row["slot"] > 0]),
        }
        (path / "summary.json").write_text(json.dumps(summary["modes"][mode], indent=2))
    validation = output_root / "temporal_channel_validation"
    validation.mkdir(parents=True)
    _write_csv(validation / "trajectory_metrics.csv", validation_rows)
    comparison = output_root / "paired_temporal_comparison"
    comparison.mkdir(parents=True)
    paired = {}
    for left, right, label in (("T1", "T2", "deployable_delayed_csi_gain"),
                                ("T2", "T3", "stale_csi_penalty")):
        paired[label] = {}
        for snr in map(float, channel_spec["snr_db"]):
            a = [r for r in all_rows if r["mode"] == left and r["snr_db"] == snr and r["slot"] > 0]
            b = [r for r in all_rows if r["mode"] == right and r["snr_db"] == snr and r["slot"] > 0]
            paired[label][str(snr)] = {
                key: _mean(b, key) - _mean(a, key)
                for key in ("post_mmse_sinr_db", "aggregate_layer_nmse",
                            "summed_latent_nmse", "si_sdr_db", "waveform_snr_db",
                            "stft_ratio", "allocation_reliability_metric")
            }
    summary["temporal_validation"] = {
        key: _mean(validation_rows, key) for key in (
            "configured_rho", "measured_rho", "delayed_current_pearson",
            "delayed_current_spearman", "top_10_overlap", "top_25_overlap", "top_50_overlap"
        )
    }
    summary["paired_comparison"] = paired
    (comparison / "comparison.json").write_text(json.dumps(paired, indent=2))
    (output_root / "final_summary.json").write_text(json.dumps(summary, indent=2))
    (output_root / "final_summary.md").write_text(
        "# FDD temporal-CSI comparison\n\n"
        f"- Doppler: {fd:.6f} Hz\n"
        f"- Configured lag-1 correlation: {rho:.8f}\n"
        f"- Measured lag-1 correlation: {summary['temporal_validation']['measured_rho']:.8f}\n"
        f"- Delayed/current Spearman correlation: "
        f"{summary['temporal_validation']['delayed_current_spearman']:.8f}\n"
        "- Slot 0: uniform bootstrap (reported both included and excluded)\n"
        "- T2: previous-slot receiver-estimated CSI only\n"
        "- T3: current true CSI for allocation only; current estimated CSI for RX MMSE\n\n"
        "The configured layer order is smoke-derived, provisional, and not an "
        "accepted scientific ablation result.\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
