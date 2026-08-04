from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from scipy import stats

from channels.global_triplet_allocator import GlobalTripletCSIReport, allocate_global_balanced_triplets
from channels.multipath import exponential_pdp
from channels.physical_ofdm import NR_LIKE_R4, active_grid_masks
from channels.temporal_multipath import correlated_tap_trajectory, expand_taps_to_sample_delays, jakes_slot_correlation
from speech_jscc.config import resolve_device
from speech_jscc.evaluation.r4_stratified_allocation import (
    PRIMARY_PROFILES, allocation_profile, compose_source_power, destination_power_from_source_order, measured_average_power,
    measured_transmit_symbol_power, normalize_layer_power_weights, paired_bootstrap_summary,
    preserve_measured_transmit_power, reference_check, source_layer_indices,
    validate_paired_profile_rows,
)
from speech_jscc.experiment import build_components
from speech_jscc.training.channel_free_revalidation import per_layer_nmse
from speech_jscc.training.r4_waveform_finetune import (
    R4ForwardCondition, R4WaveformForward, freeze_codec_for_input_gradient,
    r4_physical_layer_forward,
)
from src.evaluation.waveform_metrics import waveform_metrics
from train_channel_free_conv_conformer import load_batch


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval_r4_stratified_allocation.yaml")
    parser.add_argument("--dataset-role", choices=("legacy_final", "expanded_selection"), default="legacy_final")
    parser.add_argument("--snr-db", type=float, default=5.0)
    parser.add_argument("--profiles", nargs="+", default=list(PRIMARY_PROFILES), choices=PRIMARY_PROFILES)
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--max-utterances", type=int)
    parser.add_argument("--max-realizations", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-long-run", action="store_true")
    return parser.parse_args()


def _jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def resolve_config_path(config_path: str | Path, value: str | Path) -> Path:
    """Resolve repository-config paths independently of the invoking directory."""
    path = Path(value)
    return path if path.is_absolute() else Path(__file__).resolve().parent / path


def resolve_checkpoint_config_paths(model_config: dict) -> None:
    """Make codec assets in a checkpoint portable across invocation directories."""
    codec = model_config.get("codec", {})
    for key in ("config_path", "checkpoint_path"):
        if codec.get(key):
            codec[key] = str(resolve_config_path(__file__, codec[key]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash(tensor: torch.Tensor | None) -> str | None:
    if tensor is None:
        return None
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row if not isinstance(row[key], (list, dict))})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields} for row in rows])


def _holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[name] = running
    return adjusted


def _statistical_tests(rows: list[dict], profiles: list[str]) -> dict:
    results = {}
    pvalues = {}
    for profile in profiles:
        if profile == "uniform":
            continue
        grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            grouped[row["utterance_id"]][row["profile"]].append(float(row["si_sdr_db"]))
        deltas = [sum(values[profile]) / len(values[profile]) - sum(values["uniform"]) / len(values["uniform"])
                  for values in grouped.values()]
        if len(deltas) >= 2 and any(value != 0 for value in deltas):
            wilcoxon = float(stats.wilcoxon(deltas).pvalue)
            ttest = float(stats.ttest_1samp(deltas, 0.0).pvalue)
        else:
            wilcoxon = ttest = 1.0
        pvalues[profile] = wilcoxon
        results[profile] = {"wilcoxon_signed_rank_pvalue": wilcoxon, "paired_t_test_pvalue_reference_only": ttest}
    adjusted = _holm(pvalues)
    for profile, value in results.items():
        value["wilcoxon_holm_adjusted_pvalue"] = adjusted[profile]
    return results


def _conditions(spec: dict, count: int, seed: int) -> list[torch.Tensor]:
    physical = yaml.safe_load(Path(spec["physical_profile_config"]).read_text())
    pdp = exponential_pdp(physical["channel"]["num_taps"], physical["channel"]["pdp_decay"])
    rho = jakes_slot_correlation(physical["physical"]["user_speed_mps"], physical["physical"]["carrier_frequency_hz"], NR_LIKE_R4.tti_duration_s)
    return correlated_tap_trajectory(slots=count, batch_size=1, pdp=pdp, rho=rho, seed=seed + 500)


def _profile_allocation(base, profile_name: str):
    normalized = normalize_layer_power_weights(allocation_profile(profile_name).raw_weights)
    source_power = compose_source_power(base.power_source_order, normalized, source_layer_indices())
    destination_power = destination_power_from_source_order(source_power, base.resource_to_source)
    return dataclasses.replace(base, power_per_resource=destination_power), normalized, source_power


def _report_from_physical(physical, tti: int) -> GlobalTripletCSIReport:
    masks = active_grid_masks(NR_LIKE_R4, device=physical.estimated_channel.device)
    reliability = physical.estimated_channel.abs().square().mean(0)[masks.candidate_data].detach().cpu()
    return GlobalTripletCSIReport.from_reliability(tti, reliability)


def _report_markdown(role: str, baseline: dict | None, summaries: dict) -> str:
    lines = ["# R4 stratified allocation exploratory report", "", "## Scope", "", "Fixed power-tier comparison only; no RE/repetition/CSI policy changes and no learned allocation.", "", "## Baseline reproduction", "", json.dumps(baseline, indent=2), "", "## Paired SI-SDR deltas", ""]
    for name, value in summaries.items():
        lines.append(f"- {name}: mean {value['mean']:.6f} dB, CI {value['bootstrap_95_ci']}, positive fraction {value['positive_gain_fraction']:.3f}")
    lines += ["", "## Interpretation limits", "", "Oracle ablation gain is not power-allocation gain. These fixed heuristic profiles do not establish an optimal or production policy. Results precede SI-SDR-aware fine-tuning; changed training objectives may change both importance and allocation response. Jammer evaluation remains out of scope."]
    return "\n".join(lines) + "\n"


def _tail_risk(rows: list[dict], profiles: list[str]) -> tuple[dict, list[dict]]:
    grouped: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[(row["utterance_id"], row["profile"])][row["profile"]].append(row)
    by_utterance: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_utterance[row["utterance_id"]][row["profile"]].append(row)
    summary, catastrophic = {}, []
    for profile in profiles:
        if profile == "uniform": continue
        members = []
        for utterance_id, values in by_utterance.items():
            uniform, candidate = values["uniform"], values[profile]
            delta = sum(x["si_sdr_db"] for x in candidate) / len(candidate) - sum(x["si_sdr_db"] for x in uniform) / len(uniform)
            members.append((delta, utterance_id, uniform, candidate))
        values = torch.tensor([item[0] for item in members], dtype=torch.float64)
        summary[profile] = {"worst_utterances": [{"utterance_id": item[1], "delta_si_sdr_vs_uniform_db": item[0]} for item in sorted(members)[:5]], "p1": float(torch.quantile(values, .01)), "p5": float(torch.quantile(values, .05)), "p10": float(torch.quantile(values, .10)), "minimum": float(values.min()), "catastrophic_failure_fraction": {str(threshold): float((values < threshold).double().mean()) for threshold in (-1.0, -3.0, -5.0, -10.0)}}
        for delta, utterance_id, uniform, candidate in members:
            if delta < -1.0:
                catastrophic.append({"profile": profile, "utterance_id": utterance_id, "speaker_id": candidate[0]["speaker_id"], "utterance_mean_delta_si_sdr_vs_uniform_db": delta, "uniform_rows": uniform, "profile_rows": candidate})
    return summary, catastrophic


def main() -> None:
    args = _args()
    config_path = Path(args.config).resolve()
    spec = yaml.safe_load(config_path.read_text())
    for key in ("selected_checkpoint", "initial_checkpoint", "output_root", "physical_profile_config"):
        spec[key] = str(resolve_config_path(config_path, spec[key]))
    for key, value in spec["data"].items():
        if key.endswith("_manifest"):
            spec["data"][key] = str(resolve_config_path(config_path, value))
    if args.profiles != list(PRIMARY_PROFILES):
        raise SystemExit("primary comparison requires uniform core_protection layer1_focused together")
    entries = _jsonl(spec["data"][f"{args.dataset_role}_manifest"])
    entries = entries[:args.max_utterances] if args.max_utterances else entries[:spec["evaluation"]["utterances"][args.dataset_role]]
    seeds = spec["evaluation"]["realization_seeds"][args.dataset_role]
    seeds = seeds[:args.max_realizations] if args.max_realizations else seeds
    output = Path(args.output_dir or Path(spec["output_root"]) / f"{args.dataset_role}_{int(args.snr_db)}db")
    dry = {"dataset_role": args.dataset_role, "utterances": len(entries), "realizations": len(seeds), "snr_db": args.snr_db, "profiles": args.profiles, "output_dir": str(output), "source_symbols": 1920, "total_data_re": 5760, "repetition": 3}
    if args.dry_run:
        print(json.dumps(dry, indent=2)); return
    if len(entries) * len(seeds) > 4 and not args.allow_long_run:
        raise SystemExit("evaluation beyond smoke size requires --allow-long-run")
    if output.exists():
        if not args.overwrite: raise SystemExit(f"refusing existing output directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    _write_json(output / "environment.json", {"python": platform.python_version(), "torch": torch.__version__, "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()})
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n")
    checkpoint_path = Path(spec["selected_checkpoint"]); (output / "checkpoint_sha256.txt").write_text(_sha256(checkpoint_path) + "\n")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = copy.deepcopy(payload["config"]); resolve_checkpoint_config_paths(model_config); model_config["device"] = args.device or spec["device"]
    device = resolve_device(model_config["device"]); codec, model = build_components(model_config, device)
    model.load_state_dict(payload["model"], strict=True); model.eval().requires_grad_(False); freeze_codec_for_input_gradient(codec)
    initial_payload = torch.load(spec["initial_checkpoint"], map_location="cpu", weights_only=False)
    initial_model = copy.deepcopy(model); initial_model.load_state_dict(initial_payload["model"], strict=True); initial_model.eval().requires_grad_(False)
    layer_weights = {name: [float(value) for value in normalize_layer_power_weights(allocation_profile(name).raw_weights)] for name in args.profiles}
    _write_json(output / "allocation_profiles.json", {name: dataclasses.asdict(allocation_profile(name)) for name in args.profiles})
    _write_json(output / "normalized_power_weights.json", {"symbol_counts": [240] * 8, "weights": layer_weights, "normalization": "sum(N_l*w_l)/sum(N_l)=1"})
    rows: list[dict] = []; sample_rate = int(model_config["codec"]["sample_rate"]); physical_config = yaml.safe_load(Path(spec["physical_profile_config"]).read_text())
    delays = tuple(range(0, 2 * physical_config["channel"]["num_taps"], 2))
    with torch.no_grad():
        for realization, seed in enumerate(seeds):
            trajectory = _conditions(spec, len(entries), int(seed)); delayed = None
            initial_engine = R4WaveformForward(codec, initial_model, estimator_ridge_lambda=float(spec["physical"]["estimator_ridge_lambda"]), epsilon=float(spec["physical"]["epsilon"]))
            for tti, entry in enumerate(entries):
                waveform = load_batch([Path(entry["source_path"])], model_config, device); target = codec.encode_waveform(waveform)
                state = target.new_zeros((1, model.encoder.channel_state_dim)); source = model.encoder(target, state); clean = codec.decode_representation(target)
                base = allocate_global_balanced_triplets(profile=NR_LIKE_R4, tx_tti=tti, report=delayed, layer_importance_order=[1, 0, 2, 5, 3, 4, 6, 7])
                taps = expand_taps_to_sample_delays(trajectory[tti].to(device), delays)
                condition = R4ForwardCondition(snr_db=args.snr_db, tti=tti, tap_coefficients=trajectory[tti], tap_delay_samples=delays, noise_seed=int(seed) + tti + round(args.snr_db * 1000), noise_variance_override=10.0 ** (-args.snr_db / 10.0))
                initial_result = initial_engine.forward(target, waveform, condition, delayed, training=False)
                shared = {}
                for name in args.profiles:
                    allocation, normalized, source_power = _profile_allocation(base, name)
                    source_power = preserve_measured_transmit_power(source, base.power_source_order, source_power)
                    destination_power = destination_power_from_source_order(source_power, base.resource_to_source).to(base.power_per_resource.device)
                    allocation = dataclasses.replace(base, power_per_resource=destination_power)
                    generator = torch.Generator(device=device).manual_seed(int(seed) + tti + round(args.snr_db * 1000))
                    physical = r4_physical_layer_forward(source, allocation, taps, snr_db=args.snr_db, noise_generator=generator, tap_delay_samples=delays, estimator_num_taps=physical_config["channel"]["num_taps"], estimator_ridge_lambda=float(spec["physical"]["estimator_ridge_lambda"]), epsilon=float(spec["physical"]["epsilon"]), noise_variance_override=10.0 ** (-args.snr_db / 10.0))
                    reconstruction = model.decoder(physical.combined.estimate, physical.decoder_state); decoded = codec.decode_representation(reconstruction)
                    metrics = waveform_metrics(waveform, decoded, sample_rate); clean_metrics = waveform_metrics(waveform, clean, sample_rate)
                    layer = per_layer_nmse(reconstruction, target); received = source_power.mean(0).reshape(8, 240).mean(1)
                    post = (physical.estimated_channel_source_order.abs().square() * source_power.to(physical.estimated_channel_source_order.device)[None] / physical.noise_variance).sum(1).reshape(1, 8, 240).mean((0, 2))
                    effective = source.abs().square().sum() / (physical.combined.estimate - source).abs().square().sum().clamp_min(1e-12)
                    rms = waveform.square().mean().sqrt(); low_energy = float((waveform.abs() < rms * .01).double().mean())
                    row = {"utterance_id": entry["utterance_id"], "speaker_id": entry.get("speaker_id", ""), "realization": realization, "tti": tti, "profile": name, "snr_db": args.snr_db, "channel_seed": int(seed), "noise_seed": int(seed) + tti + round(args.snr_db * 1000), "mapping_hash": _hash(base.selected_candidate_indices), "input_delayed_csi_hash": _hash(None if delayed is None else delayed.reliability), "si_sdr_db": float(metrics["si_sdr_db"]), "clean_codec_si_sdr_db": float(clean_metrics["si_sdr_db"]), "waveform_rms": float(rms), "low_energy_fraction_1pct_rms": low_energy, "waveform_snr_db": float(metrics["waveform_snr_db"]), "multires_stft": float(metrics["multi_resolution_stft_distance"]), "stft_l1": float(metrics["stft_l1"]), "delta_si_sdr_vs_clean_codec_db": float(metrics["si_sdr_db"] - clean_metrics["si_sdr_db"]), "delta_waveform_snr_vs_clean_codec_db": float(metrics["waveform_snr_db"] - clean_metrics["waveform_snr_db"]), "stft_ratio_vs_clean_codec": float(metrics["stft_l1"] / max(clean_metrics["stft_l1"], 1e-12)), "aggregate_latent_nmse": float(layer.mean()), "per_layer_nmse": [float(x) for x in layer], "effective_sinr_db": float(10 * torch.log10(effective)), "per_layer_received_power": [float(x) for x in received], "per_layer_post_combining_snr_db": [float(10 * torch.log10(x)) for x in post], "normalized_layer_power_weights": [float(x) for x in normalized], "measured_total_average_power": measured_average_power(source_power), "measured_source_weighted_transmit_power": float(measured_transmit_symbol_power(source, source_power))}
                    rows.append(row); shared[name] = physical
                delayed = initial_result.next_delayed_csi
    validate_paired_profile_rows(rows)
    baseline_rows = [row for row in rows if row["profile"] == "uniform"]
    mean_baseline = {key: sum(float(row[key]) for row in baseline_rows) / len(baseline_rows) for key in ("si_sdr_db", "waveform_snr_db", "stft_l1")}
    baseline = None if len(entries) < 64 or len(seeds) < 2 or args.dataset_role != "legacy_final" else reference_check(mean_baseline, spec["reference"]["legacy_final"], si_sdr_tolerance=spec["reference"]["si_sdr_tolerance"], waveform_snr_tolerance=spec["reference"]["waveform_snr_tolerance"], stft_tolerance=spec["reference"]["stft_tolerance"])
    _write_json(output / "baseline_reproduction.json", {"uniform_mean": mean_baseline, "reference_check": baseline, "smoke": len(entries) < int(spec["evaluation"]["utterances"][args.dataset_role])})
    if baseline is not None and not baseline["passed"]: raise SystemExit("uniform baseline reference check failed; comparison aborted")
    summaries = {name: paired_bootstrap_summary(rows, profile=name, metric="si_sdr_db", samples=int(spec["statistics"]["bootstrap_samples"]), seed=int(spec["statistics"]["bootstrap_seed"])) for name in args.profiles if name != "uniform"}
    tail_summary, catastrophic = _tail_risk(rows, args.profiles)
    _write_jsonl(output / "per_sample_profile_comparison.jsonl", rows); _write_json(output / "paired_delta_summary.json", summaries); _write_json(output / "statistical_tests.json", _statistical_tests(rows, args.profiles)); _write_json(output / "tail_risk_summary.json", tail_summary); _write_json(output / "catastrophic_samples.json", catastrophic)
    profile_summary = {name: {key: sum(float(row[key]) for row in rows if row["profile"] == name) / len(baseline_rows) for key in ("si_sdr_db", "waveform_snr_db", "stft_l1", "aggregate_latent_nmse", "effective_sinr_db", "measured_total_average_power", "measured_source_weighted_transmit_power")} for name in args.profiles}
    _write_json(output / "profile_summary.json", profile_summary); _write_json(output / "per_layer_nmse_summary.json", {name: [sum(row["per_layer_nmse"][layer] for row in rows if row["profile"] == name) / len(baseline_rows) for layer in range(8)] for name in args.profiles}); _write_json(output / "per_layer_received_power_summary.json", {name: [sum(row["per_layer_received_power"][layer] for row in rows if row["profile"] == name) / len(baseline_rows) for layer in range(8)] for name in args.profiles})
    legacy = Path(spec["output_root"]) / "legacy_final_5db" / "paired_delta_summary.json"
    split = {"status": "single_split_run", "dataset_role": args.dataset_role}
    if args.dataset_role == "expanded_selection" and legacy.exists():
        legacy_summary = json.loads(legacy.read_text())
        split = {"status": "computed", "legacy_final": legacy_summary, "expanded_selection": summaries, "mean_direction_agreement": {name: (legacy_summary[name]["mean"] > 0) == (summaries[name]["mean"] > 0) for name in summaries}}
    _write_json(output / "split_comparison.json", split); _write_json(output / "profile_ranking.json", {"ranking_by_mean_si_sdr": sorted(args.profiles, key=lambda name: profile_summary[name]["si_sdr_db"], reverse=True), "exploratory_only": True})
    _write_csv(output / "results_table.csv", [{"profile": name, **values} for name, values in profile_summary.items()]); _write_csv(output / "si_sdr_by_profile.csv", rows); _write_csv(output / "per_layer_nmse_by_profile.csv", [{"profile": row["profile"], "utterance_id": row["utterance_id"], "realization": row["realization"], "layer": layer, "nmse": value} for row in rows for layer, value in enumerate(row["per_layer_nmse"])]); _write_csv(output / "paired_si_sdr_delta_distribution.csv", [{"profile": name, **summary} for name, summary in summaries.items()]); _write_csv(output / "tail_risk_by_profile.csv", [{"profile": name, "threshold_db": threshold, "fraction": fraction} for name, summary in tail_summary.items() for threshold, fraction in summary["catastrophic_failure_fraction"].items()])
    (output / "report.md").write_text(_report_markdown(args.dataset_role, baseline, summaries)); print(json.dumps({"output": str(output), "baseline": baseline, "profile_summary": profile_summary}, indent=2))


if __name__ == "__main__": main()
