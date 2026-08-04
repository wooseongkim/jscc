"""Paired source-versus-candidate R4 waveform smoke evaluator.

This intentionally narrow CLI reuses the expanded-selection R4 physical forward.
It is not a checkpoint ranking or selection tool.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from evaluate_r4_expanded_validation import (
    _cache_codec_inputs,
    _conditions,
    _make_engine,
    _report_hash,
    _tensor_hash,
)
from speech_jscc.config import resolve_device
from speech_jscc.evaluation.expanded_validation import ManifestEntry, file_sha256
from speech_jscc.experiment import build_components
from speech_jscc.training.channel_free_revalidation import per_layer_nmse
from speech_jscc.training.r4_waveform_finetune import freeze_codec_for_input_gradient
from src.evaluation.waveform_metrics import waveform_metrics
from speech_jscc.evaluation.r4_si_sdr_checkpoint_selection import (
    discover_checkpoints, full_constraints, model_state_hash, paired_statistics, rank_experiments, smoke_constraints, statistical_tests,
    write_selection_artifact,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent


def _repository_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _checkpoint_payload(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"checkpoint has no model state: {path}")
    return payload


def _condition_hash(*, condition, input_report, mapping_indices, codec_hash: str) -> str:
    """Hash every channel/input value that must be shared by the pair."""
    digest = hashlib.sha256()
    digest.update(str(condition.snr_db).encode())
    digest.update(str(condition.tti).encode())
    digest.update(str(condition.noise_seed).encode())
    digest.update(str(condition.noise_variance_override).encode())
    digest.update(_tensor_hash(condition.tap_coefficients).encode())
    digest.update((_report_hash(input_report) or "none").encode())
    digest.update(_tensor_hash(mapping_indices).encode())
    digest.update(codec_hash.encode())
    return digest.hexdigest()


def _finite_metrics(metrics: dict) -> bool:
    return all(math.isfinite(float(metrics[name])) for name in ("si_sdr_db", "waveform_snr_db", "stft_l1"))


def _source_rows(*, codec, model, cache, conditions, specification, device):
    """Expanded-validation source trajectory, with values retained for pairing."""
    engine = _make_engine(codec, model, specification)
    sample_rate = int(model._expanded_validation_config["codec"]["sample_rate"])
    report = None
    rows = []
    with torch.no_grad():
        for cached, condition in zip(cache, conditions, strict=True):
            waveform = cached["waveform"].to(device)
            target = cached["target"].to(device)
            input_report = report
            result = engine.forward(target, waveform, condition, input_report, training=False)
            metrics = waveform_metrics(waveform, result.decoded_waveform, sample_rate)
            if not _finite_metrics(metrics):
                raise FloatingPointError(
                    f"nonfinite source metric utterance={cached['entry'].utterance_id} seed={condition.noise_seed}"
                )
            rows.append({
                "metrics": metrics,
                "input_report": input_report,
                "input_report_hash": _report_hash(input_report),
                "next_report": result.next_delayed_csi,
                "mapping_hash": _tensor_hash(result.mapping_indices),
                "mapping_indices": result.mapping_indices.detach().cpu(),
                "latent_nmse": float(per_layer_nmse(result.reconstruction, target).mean()),
                "per_layer_nmse": [float(x) for x in per_layer_nmse(result.reconstruction, target)],
                "effective_sinr_db": float(10.0 * torch.log10(result.effective_sinr)),
            })
            report = result.next_delayed_csi
    return rows


def _candidate_rows(*, codec, model, cache, conditions, source_rows, specification, device):
    """Evaluate candidate against the fixed source delayed-CSI trajectory."""
    engine = _make_engine(codec, model, specification)
    sample_rate = int(model._expanded_validation_config["codec"]["sample_rate"])
    tolerance = float(specification["pairing"]["csi_report_tolerance"])
    rows = []
    with torch.no_grad():
        for cached, condition, source in zip(cache, conditions, source_rows, strict=True):
            waveform = cached["waveform"].to(device)
            target = cached["target"].to(device)
            result = engine.forward(target, waveform, condition, source["input_report"], training=False)
            metrics = waveform_metrics(waveform, result.decoded_waveform, sample_rate)
            if not _finite_metrics(metrics):
                raise FloatingPointError(
                    f"nonfinite candidate metric utterance={cached['entry'].utterance_id} seed={condition.noise_seed}"
                )
            if _report_hash(source["input_report"]) != source["input_report_hash"]:
                raise AssertionError("candidate input delayed-CSI report differs from source")
            if _tensor_hash(result.mapping_indices) != source["mapping_hash"]:
                raise AssertionError("candidate allocation mapping differs from source")
            report_difference = float((result.next_delayed_csi.reliability - source["next_report"].reliability).abs().max())
            if report_difference > tolerance:
                raise AssertionError(f"candidate delayed-CSI observation mismatch: {report_difference}")
            rows.append({
                "metrics": metrics,
                "latent_nmse": float(per_layer_nmse(result.reconstruction, target).mean()),
                "per_layer_nmse": [float(x) for x in per_layer_nmse(result.reconstruction, target)],
            })
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval_r4_si_sdr_finetune_checkpoints.yaml")
    parser.add_argument("--candidate-checkpoint")
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--stage", choices=("waveform_smoke", "selection_smoke", "full_selection"), default="waveform_smoke")
    parser.add_argument("--allow-long-run", action="store_true")
    parser.add_argument("--max-utterances", type=int, default=2)
    parser.add_argument("--max-realizations", type=int, default=1)
    parser.add_argument("--snr-db", type=float, default=5.0)
    parser.add_argument("--manifest")
    parser.add_argument("--realization-seeds")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _selection_main(args, config, source_path: Path) -> None:
    """Repeat the verified single-pair CLI and aggregate only expanded rows."""
    if args.stage == "full_selection" and not args.allow_long_run:
        raise SystemExit("--stage full_selection requires --allow-long-run")
    default_output = (
        config["selection_output_root"]
        if args.stage == "selection_smoke"
        else config["output_root"]
    )
    output = _repository_path(args.output_dir or default_output)
    experiments = args.experiments or list(config["experiments"])
    weights = {name: float(value["target_si_sdr_weight"]) for name, value in config["experiments"].items()}
    records, representatives = discover_checkpoints(_repository_path(config["training_root"]), experiments, weights)
    if args.dry_run:
        print(json.dumps({"stage":args.stage,"experiments":experiments,"discovered":len(records),"deduplicated":len(representatives),"selection_split":"expanded_selection"}, indent=2)); return
    if output.exists():
        if not args.overwrite: raise SystemExit(f"refusing existing output directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    _write_json(output/"discovered_checkpoints.json", records)
    _write_json(output/"checkpoint_hashes.json", [{key:item.get(key) for key in ("checkpoint_id","sha256","model_state_hash")} for item in records])
    _write_json(output/"deduplicated_checkpoints.json", representatives)
    if not representatives:
        raise SystemExit("no loadable fine-tuning checkpoints discovered")
    rows=[]; summaries=[]; constraints=[]
    temp = output / ".pair_runs"
    snrs = [args.snr_db] if args.stage == "selection_smoke" else [5.0,10.0,15.0]
    utterances = args.max_utterances if args.stage == "selection_smoke" else 48
    realizations = args.max_realizations if args.stage == "selection_smoke" else 2
    for inventory in representatives:
        candidate_rows=[]
        for snr in snrs:
            child = temp / inventory["checkpoint_id"].replace(":","_") / str(snr)
            command=[sys.executable, str(REPOSITORY_ROOT/"evaluate_r4_si_sdr_finetune_checkpoints.py"), "--config", str(_repository_path(args.config)), "--candidate-checkpoint", inventory["checkpoint_path"], "--max-utterances", str(utterances), "--max-realizations", str(realizations), "--snr-db", str(snr), "--output-dir", str(child), "--device", args.device, "--stage", "waveform_smoke", "--overwrite"]
            completed=subprocess.run(command, cwd=REPOSITORY_ROOT, text=True, capture_output=True)
            if completed.returncode:
                raise RuntimeError(f"waveform evaluation failed for {inventory['checkpoint_id']}: {completed.stderr}\n{completed.stdout}")
            child_rows=[json.loads(line) for line in (child/"per_sample_selection_results.jsonl").read_text().splitlines()]
            for row in child_rows:
                row.update({"source_checkpoint_id":"source_step5750","source_checkpoint_sha256":file_sha256(source_path),"candidate_checkpoint_id":inventory["checkpoint_id"],"candidate_checkpoint_sha256":inventory["sha256"],"experiment":inventory["experiment"],"local_step":inventory["local_step"],"target_si_sdr_weight":inventory["target_si_sdr_weight"],"delta_si_sdr_vs_clean_codec_db":row["candidate_si_sdr_db"]-row["clean_codec_si_sdr_db"],"delta_waveform_snr_vs_source_db":row["candidate_waveform_snr_db"]-row["source_waveform_snr_db"],"delta_stft_l1_vs_source":row["candidate_stft_l1"]-row["source_stft_l1"],"delta_latent_nmse_vs_source":row["candidate_latent_nmse"]-row["source_latent_nmse"]})
            candidate_rows.extend(child_rows); rows.extend(child_rows)
        expected=utterances*realizations*len(snrs)
        keys=[(r["utterance_id"],r["realization_index"],r["snr_db"]) for r in candidate_rows]
        if len(candidate_rows)!=expected or len(set(keys))!=expected: raise AssertionError(f"paired row completeness failed for {inventory['checkpoint_id']}")
        by_snr={}
        for snr in snrs:
            members=[r for r in candidate_rows if r["snr_db"]==float(snr)]
            utterance={}
            for row in members: utterance.setdefault(row["utterance_id"],[]).append(row)
            deltas=[sum(v["delta_si_sdr_vs_source_db"] for v in values)/len(values) for values in utterance.values()]
            stats=paired_statistics(deltas,samples=int(config["bootstrap_samples"]),seed=int(config["bootstrap_seed"])+int(snr*100))
            by_snr[str(float(snr))]={"si_sdr_mean":sum(r["candidate_si_sdr_db"] for r in members)/len(members),"source_si_sdr_mean":sum(r["source_si_sdr_db"] for r in members)/len(members),"waveform_snr_mean":sum(r["candidate_waveform_snr_db"] for r in members)/len(members),"source_waveform_snr_mean":sum(r["source_waveform_snr_db"] for r in members)/len(members),"stft_mean":sum(r["candidate_stft_l1"] for r in members)/len(members),"source_stft_mean":sum(r["source_stft_l1"] for r in members)/len(members),"latent_nmse_mean":sum(r["candidate_latent_nmse"] for r in members)/len(members),"statistics":stats}
        constraint=smoke_constraints(rows=candidate_rows) if args.stage=="selection_smoke" else full_constraints(rows=candidate_rows, by_snr=by_snr, catastrophic_3_max=float(config["constraints"]["catastrophic_lt_minus_3_fraction_max"]), catastrophic_5_max=float(config["constraints"]["catastrophic_lt_minus_5_fraction_max"]))
        metric5=by_snr["5.0"]
        metrics={"si_sdr_5db":metric5["si_sdr_mean"],"delta_vs_source_5db":metric5["statistics"]["mean"],"positive_gain_fraction_5db":metric5["statistics"]["positive_gain_fraction"],"p5_delta":metric5["statistics"]["p5"],"catastrophic_lt_minus_3":metric5["statistics"]["catastrophic_fractions"]["-3.0"],"catastrophic_lt_minus_5":metric5["statistics"]["catastrophic_fractions"]["-5.0"],"stft_ratio":metric5["stft_mean"]/max(metric5["source_stft_mean"],1e-12),"waveform_snr":metric5["waveform_snr_mean"],"average_si_sdr":sum(item["si_sdr_mean"] for item in by_snr.values())/len(by_snr)}
        summary={**inventory,"by_snr":by_snr,"metrics":metrics,"constraint_pass":constraint["pass"]}; summaries.append(summary); constraints.append({"checkpoint_id":inventory["checkpoint_id"],**constraint})
    with (output/"per_sample_selection_results.jsonl").open("w") as handle:
        for row in rows: handle.write(json.dumps(row,sort_keys=True)+"\n")
    _write_json(output/"paired_statistics.json", summaries); _write_json(output/"checkpoint_constraints.json", constraints)
    ranking=rank_experiments(summaries,tolerance=float(config["ranking_tolerance"])); _write_json(output/"checkpoint_ranking.json",ranking)
    selected={name: {**ranking.get(name, {"status":"no_passing_candidate", "selected_checkpoint":None}), "experiment":name} for name in experiments}
    source={"path":str(source_path),"sha256":file_sha256(source_path),"model_state_hash":model_state_hash(_checkpoint_payload(source_path)["model"]),"global_step":5750}
    write_selection_artifact(output/"experiment_best_checkpoints.json",source=source,selected=selected,constraint_mode="smoke" if args.stage=="selection_smoke" else "full_selection")
    test_groups={item["checkpoint_id"]:[row["delta_si_sdr_vs_source_db"] for row in rows if row["candidate_checkpoint_id"]==item["checkpoint_id"] and row["snr_db"]==5.0] for item in summaries}
    _write_json(output/"statistical_tests.json", statistical_tests(test_groups))
    _write_json(output/"tail_risk_summary.json", {item["checkpoint_id"]:item["by_snr"]["5.0"]["statistics"] for item in summaries})
    _write_json(output/"worst_samples.json", {item["checkpoint_id"]:sorted([r for r in rows if r["candidate_checkpoint_id"]==item["checkpoint_id"] and r["snr_db"]==5.0],key=lambda r:r["delta_si_sdr_vs_source_db"])[:5] for item in summaries})
    for name in ("control_vs_source","low_vs_source","medium_vs_source","low_vs_control","medium_vs_control"): _write_json(output/f"{name}.json",{"status":"unavailable","reason":"comparison aggregation is not available in this selection stage"})
    (output/"per_checkpoint_summary.csv").write_text("checkpoint_id,experiment,si_sdr_5db,delta_vs_source_5db\n"+"".join(f"{x['checkpoint_id']},{x['experiment']},{x['metrics']['si_sdr_5db']},{x['metrics']['delta_vs_source_5db']}\n" for x in summaries))
    (output/"resolved_config.yaml").write_text(yaml.safe_dump(config)); (output/"command.txt").write_text(" ".join(sys.argv)+"\n"); (output/"environment.json").write_text(json.dumps({"python":sys.version,"device":args.device,"selection_split":"expanded_selection"},indent=2)); (output/"selection_report.md").write_text("# Expanded selection checkpoint selection\n\nOnly expanded_selection waveform rows were used.\n")
    print(json.dumps({"evaluated_candidates":len(summaries),"output":str(output)},indent=2))


def main() -> None:
    args = _parse_args()
    if args.max_utterances <= 0 or args.max_realizations <= 0:
        raise SystemExit("max utterances and realizations must be positive")
    config_path = _repository_path(args.config)
    config = yaml.safe_load(config_path.read_text())
    source_path = _repository_path(config["source_checkpoint"])
    if args.stage != "waveform_smoke":
        _selection_main(args, config, source_path)
        return
    if not args.candidate_checkpoint and not args.dry_run:
        raise SystemExit("--candidate-checkpoint is required for --stage waveform_smoke")
    candidate_path = _repository_path(args.candidate_checkpoint) if args.candidate_checkpoint else None
    manifest_path = _repository_path(args.manifest or config["expanded_selection_manifest"])
    output = _repository_path(args.output_dir or config["output_root"])
    if args.dry_run:
        print(json.dumps({"source_checkpoint": str(source_path), "candidate_checkpoint": str(candidate_path) if candidate_path else None, "manifest": str(manifest_path), "scope": {"utterances": args.max_utterances, "realizations": args.max_realizations, "snr_db": args.snr_db}}, indent=2))
        return
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"refusing existing output directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    source_payload = _checkpoint_payload(source_path)
    candidate_payload = _checkpoint_payload(candidate_path)
    model_config = copy.deepcopy(source_payload["config"])
    for key in ("config_path", "checkpoint_path"):
        codec_path = Path(model_config["codec"][key])
        if not codec_path.is_absolute():
            model_config["codec"][key] = str(_repository_path(codec_path))
    model_config["device"] = args.device
    device = resolve_device(args.device)
    codec, source_model = build_components(model_config, device)
    source_model.load_state_dict(source_payload["model"], strict=True)
    candidate_model = copy.deepcopy(source_model)
    candidate_model.load_state_dict(candidate_payload["model"], strict=True)
    freeze_codec_for_input_gradient(codec)
    source_model.eval().requires_grad_(False)
    candidate_model.eval().requires_grad_(False)
    source_model._expanded_validation_config = model_config
    candidate_model._expanded_validation_config = model_config
    entries = [ManifestEntry(**json.loads(line)) for line in manifest_path.read_text().splitlines() if line.strip()]
    entries = entries[:args.max_utterances]
    if len(entries) != args.max_utterances:
        raise ValueError(f"manifest has {len(entries)} entries, requested {args.max_utterances}")
    specification = {
        "physical_profile_config": str(_repository_path(config["physical_profile_config"])),
        "physical": {"estimator_ridge_lambda": float(config["physical"]["estimator_ridge_lambda"]), "epsilon": float(config["physical"]["epsilon"])},
        "pairing": {"csi_report_tolerance": float(config["pairing"]["csi_report_tolerance"])},
    }
    cache = _cache_codec_inputs(codec, entries, model_config, device)
    seeds = ([int(x) for x in args.realization_seeds.split(",")] if args.realization_seeds else list(config["realization_seeds"]))[:args.max_realizations]
    if len(seeds) != args.max_realizations:
        raise ValueError("not enough configured realization seeds")
    rows = []
    for realization_index, realization_seed in enumerate(seeds):
        conditions = _conditions(specification, _make_engine(codec, source_model, specification), snr=args.snr_db, seed=int(realization_seed), count=len(cache))
        source_rows = _source_rows(codec=codec, model=source_model, cache=cache, conditions=conditions, specification=specification, device=device)
        candidate_rows = _candidate_rows(codec=codec, model=candidate_model, cache=cache, conditions=conditions, source_rows=source_rows, specification=specification, device=device)
        for cached, condition, source, candidate in zip(cache, conditions, source_rows, candidate_rows, strict=True):
            condition_digest = _condition_hash(condition=condition, input_report=source["input_report"], mapping_indices=source["mapping_indices"], codec_hash=cached["codec_baseline_hash"])
            fields = {
                "split": "expanded_selection", "utterance_id": cached["entry"].utterance_id, "speaker_id": cached["entry"].speaker_id,
                "realization_index": realization_index, "realization_seed": int(realization_seed), "snr_db": float(args.snr_db),
                "source_checkpoint_path": str(source_path), "candidate_checkpoint_path": str(candidate_path), "condition_hash": condition_digest,
                "mapping_hash": source["mapping_hash"],
                # Candidate evaluation has already asserted the same delayed-CSI
                # input and mapping; retain both hashes for the paired audit.
                "source_condition_hash": condition_digest, "candidate_condition_hash": condition_digest,
                "clean_codec_si_sdr_db": float(cached["clean_metrics"]["si_sdr_db"]),
                "source_si_sdr_db": float(source["metrics"]["si_sdr_db"]), "candidate_si_sdr_db": float(candidate["metrics"]["si_sdr_db"]),
                "source_waveform_snr_db": float(source["metrics"]["waveform_snr_db"]), "candidate_waveform_snr_db": float(candidate["metrics"]["waveform_snr_db"]),
                "source_stft_l1": float(source["metrics"]["stft_l1"]), "candidate_stft_l1": float(candidate["metrics"]["stft_l1"]),
                "source_latent_nmse": source["latent_nmse"], "candidate_latent_nmse": candidate["latent_nmse"],
                "source_per_layer_nmse": source["per_layer_nmse"], "candidate_per_layer_nmse": candidate["per_layer_nmse"],
                "effective_sinr_db": source["effective_sinr_db"],
                "waveform_rms": float(waveform_metrics(cached["waveform"].to(device), cached["waveform"].to(device), int(model_config["codec"]["sample_rate"]))["output_rms"]),
                "silence_ratio": float((cached["waveform"].abs() < 1e-4).float().mean()),
            }
            fields["delta_si_sdr_vs_source_db"] = fields["candidate_si_sdr_db"] - fields["source_si_sdr_db"]
            fields["finite"] = all(math.isfinite(float(value)) for value in (fields["clean_codec_si_sdr_db"], fields["source_si_sdr_db"], fields["candidate_si_sdr_db"], fields["source_waveform_snr_db"], fields["candidate_waveform_snr_db"], fields["source_stft_l1"], fields["candidate_stft_l1"], fields["source_latent_nmse"], fields["candidate_latent_nmse"], fields["effective_sinr_db"]))
            if not fields["finite"]:
                raise FloatingPointError(f"nonfinite paired row: {fields['utterance_id']}")
            rows.append(fields)
    with (output / "per_sample_selection_results.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    mean = lambda name: sum(float(row[name]) for row in rows) / len(rows)
    summary = {
        "evaluated_utterance_count": len(entries), "realization_count": len(seeds), "paired_row_count": len(rows),
        "source_mean_si_sdr_db": mean("source_si_sdr_db"), "candidate_mean_si_sdr_db": mean("candidate_si_sdr_db"),
        "mean_paired_delta_si_sdr_db": mean("delta_si_sdr_vs_source_db"),
        "condition_hash_equality": all(row["source_condition_hash"] == row["candidate_condition_hash"] for row in rows),
        "has_nan_or_inf": not all(row["finite"] for row in rows),
        "source_checkpoint_sha256": file_sha256(source_path), "candidate_checkpoint_sha256": file_sha256(candidate_path),
    }
    (output / "waveform_smoke_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n")
    (output / "report.md").write_text("# R4 SI-SDR checkpoint waveform smoke\n\n" + json.dumps(summary, indent=2) + "\n\nUses the existing expanded-validation R4 waveform forward without alignment correction or active-speech masking.\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
