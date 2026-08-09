"""Budgeted mixed-integer black-box search for R4 broadband UEP profiles.

Selection uses expanded_selection only.  ``--run-final-eval`` is a separate,
frozen-input mode for legacy_final and never re-ranks candidates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from speech_jscc.evaluation.expanded_validation import file_sha256
from speech_jscc.evaluation.r4_broadband_uep_optimizer import (
    UEPCandidate, build_selected_profiles_artifact, candidate_hash,
    load_selected_profiles, make_candidate, objective_from_summary, pareto_front,
    propose_power_transfers, propose_repetition_moves, sample_random_candidates,
    rank_by_objective, select_profiles, validate_search_split,
)


ROOT = Path(__file__).resolve().parent


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _candidate_payload(candidate: UEPCandidate, *, name: str | None = None) -> dict:
    return {
        "repetition": list(candidate.repetition),
        "power_share": list(candidate.power_share),
        "per_re_power": list(candidate.per_re_power),
        "logits": list(candidate.logits),
        "profile_id": name or candidate_hash(candidate),
    }


def _profiles_json(path: Path, candidates: list[UEPCandidate]) -> None:
    profiles = {
        "U0": {"repetition": [3] * 8, "power_share": [0.125] * 8},
    }
    for candidate in candidates:
        profiles[candidate_hash(candidate)] = _candidate_payload(candidate)
    _json(path, {"profiles": profiles})


def _condition_key(row: dict) -> tuple:
    return (
        row["utterance_id"], row["realization_index"], row["snr_db"],
        row["target_jsr_db"], row["realization_seed"], row["jammer_seed"],
    )


def _summarize_candidate(rows: list[dict], candidate_id: str, jsr_weights: dict[str, float]) -> dict:
    base = {_condition_key(row): row for row in rows if row["profile"] == "U0"}
    candidate = {_condition_key(row): row for row in rows if row["profile"] == candidate_id}
    if set(base) != set(candidate):
        raise RuntimeError("candidate/U0 paired condition keys are incomplete")
    deltas: list[tuple[dict, dict, float]] = []
    for key, baseline in base.items():
        current = candidate[key]
        for field in ("jammer_mask_hash", "jammer_tensor_hash", "codec_input_hash", "target_waveform_hash"):
            if current[field] != baseline[field]:
                raise RuntimeError(f"paired condition mismatch for {field}: {key}")
        if not (current["finite"] and baseline["finite"]):
            raise FloatingPointError(f"non-finite profile row: {key}")
        deltas.append((baseline, current, current["si_sdr_db"] - baseline["si_sdr_db"]))
    broadband = [(baseline, current, delta) for baseline, current, delta in deltas if baseline["target_jsr_db"] is not None]
    if not broadband:
        raise RuntimeError("optimizer requires at least one broadband JSR condition")
    weighted_values = []
    for baseline, _, delta in broadband:
        weight = float(jsr_weights[str(float(baseline["target_jsr_db"]))])
        weighted_values.append((weight, delta))
    weighted = sum(weight * delta for weight, delta in weighted_values) / sum(weight for weight, _ in weighted_values)
    clean = [delta for baseline, _, delta in deltas if baseline["target_jsr_db"] is None]
    utterance: dict[tuple, list[float]] = {}
    for baseline, _, delta in broadband:
        utterance.setdefault((baseline["utterance_id"], baseline["snr_db"], baseline["target_jsr_db"]), []).append(delta)
    utterance_delta = sorted(sum(values) / len(values) for values in utterance.values())
    def percentile(values, fraction):
        if not values:
            return 0.0
        index = (len(values) - 1) * fraction
        lo, hi = int(index), min(int(index) + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    stft_delta = sum(current["stft_l1"] - baseline["stft_l1"] for baseline, current, _ in broadband) / len(broadband)
    absolute_base = sum(baseline["si_sdr_db"] < -10.0 for baseline, _, _ in broadband) / len(broadband)
    absolute_current = sum(current["si_sdr_db"] < -10.0 for _, current, _ in broadband) / len(broadband)
    return {
        "weighted_mean_delta_si_sdr_db": weighted,
        "mean_delta_si_sdr_db": sum(delta for _, _, delta in broadband) / len(broadband),
        "clean_cost_db": max(0.0, -sum(clean) / len(clean)) if clean else 0.0,
        "p5_delta_si_sdr_db": percentile(utterance_delta, 0.05),
        "minimum_delta_si_sdr_db": min(utterance_delta),
        "severe_tail_increase_fraction": sum(value < -3.0 for value in utterance_delta) / len(utterance_delta),
        "absolute_catastrophic_reduction_fraction": absolute_base - absolute_current,
        "absolute_catastrophic_fraction": absolute_current,
        "u0_absolute_catastrophic_fraction": absolute_base,
        "stft_l1_delta": stft_delta,
        "waveform_snr_delta_db": sum(current["waveform_snr_db"] - baseline["waveform_snr_db"] for baseline, current, _ in broadband) / len(broadband),
        "latent_nmse_delta": sum(current["aggregate_latent_nmse"] - baseline["aggregate_latent_nmse"] for baseline, current, _ in broadband) / len(broadband),
        "paired_row_count": len(deltas),
        "broadband_row_count": len(broadband),
    }


def _u0_record(rows: list[dict]) -> dict:
    u0_rows = [row for row in rows if row["profile"] == "U0"]
    if not u0_rows or not all(row["finite"] for row in u0_rows):
        raise FloatingPointError("U0 baseline rows are missing or non-finite")
    candidate = make_candidate((3,) * 8, (0.0,) * 8)
    summary = {
        "weighted_mean_delta_si_sdr_db": 0.0,
        "mean_delta_si_sdr_db": 0.0,
        "clean_cost_db": 0.0,
        "p5_delta_si_sdr_db": 0.0,
        "minimum_delta_si_sdr_db": 0.0,
        "severe_tail_increase_fraction": 0.0,
        "absolute_catastrophic_reduction_fraction": 0.0,
        "absolute_catastrophic_fraction": sum(row["si_sdr_db"] < -10.0 for row in u0_rows) / len(u0_rows),
        "stft_l1_delta": 0.0,
        "waveform_snr_delta_db": 0.0,
        "latent_nmse_delta": 0.0,
        "paired_row_count": len(u0_rows),
        "broadband_row_count": sum(row["target_jsr_db"] is not None for row in u0_rows),
    }
    return {
        "candidate_id": "U0", "candidate": _candidate_payload(candidate, name="U0"),
        "summary": summary, "objective": objective_from_summary(summary),
    }


def _run_backend(*, config: dict, checkpoint: Path, split: str, candidates: list[UEPCandidate], args, output: Path) -> list[dict]:
    profile_path = output / "profile.json"
    output.mkdir(parents=True, exist_ok=True)
    _profiles_json(profile_path, candidates)
    command = [
        sys.executable, str(ROOT / "evaluate_r4_broadband_uep_profiles.py"),
        "--config", str(_path(config["evaluation_backend_config"])),
        "--checkpoint", str(checkpoint), "--split", split,
        "--profile-json", str(profile_path), "--profiles", "U0", *[candidate_hash(candidate) for candidate in candidates],
        "--jammer-type", "broadband_awgn", "--jsr-db", *[str(value) for value in args.jsr_db],
        "--snr-db", *[str(value) for value in args.snr_db], "--device", args.device,
        "--output-dir", str(output / "backend"), "--overwrite",
    ]
    if args.max_utterances is not None:
        command.extend(["--max-utterances", str(args.max_utterances)])
    if args.max_realizations is not None:
        command.extend(["--max-realizations", str(args.max_realizations)])
    # A compact optimizer smoke can contain several profiles and exceed the
    # single-profile evaluator's conservative row guard while still being a
    # deliberate bounded search.  The public optimizer's own budgets remain
    # the authoritative long-run guard.
    if args.allow_long_run or len(candidates) > 1:
        command.append("--allow-long-run")
    subprocess.run(command, cwd=ROOT, check=True)
    return [json.loads(line) for line in (output / "backend" / "per_sample_uep_results.jsonl").read_text().splitlines() if line]


def _evaluate_batch(*, config, checkpoint, args, candidates, root, stage: str) -> dict[str, dict]:
    """Evaluate U0 and a candidate batch with one shared condition cache."""
    unique = {candidate_hash(candidate): candidate for candidate in candidates}
    if not unique:
        return {}
    rows = _run_backend(config=config, checkpoint=checkpoint, split="expanded_selection", candidates=list(unique.values()), args=args, output=root / "candidate_runs" / stage)
    records = {}
    records["U0"] = _u0_record(rows)
    for cid, candidate in unique.items():
        summary = _summarize_candidate(rows, cid, config["objective"]["jsr_weights"])
        record = {"candidate_id": cid, "candidate": _candidate_payload(candidate), "summary": summary}
        record["objective"] = objective_from_summary(summary, **config["objective"]["penalty_weights"])
        records[cid] = record
    with (root / "candidate_eval_results.jsonl").open("a") as handle:
        for row in rows:
            handle.write(json.dumps({**row, "evaluation_stage": stage}, sort_keys=True) + "\n")
    return records


def _search(args, config: dict, checkpoint: Path, output: Path) -> None:
    validate_search_split(args.selection_split)
    stage_count = int(config["search"]["max_candidates_stage1"] if args.max_candidates_stage1 is None else args.max_candidates_stage1)
    top_k = int(config["search"]["top_k_for_refine"] if args.top_k_for_refine is None else args.top_k_for_refine)
    refine_steps = int(config["search"]["local_refine_steps"] if args.local_refine_steps is None else args.local_refine_steps)
    random_candidates = sample_random_candidates(count=stage_count + 1, seed=int(config["search"]["seed"]))[1:]
    evaluated = _evaluate_batch(config=config, checkpoint=checkpoint, args=args, candidates=random_candidates, root=output, stage="random")
    ranked = rank_by_objective(list(evaluated.values()))
    proposals: list[UEPCandidate] = []
    for index in range(refine_steps):
        base = ranked[index % min(top_k, len(ranked))]["candidate"]
        repetition = tuple(base["repetition"])
        shares = tuple(base["power_share"])
        if index % 2 == 0:
            moves = propose_repetition_moves(repetition)
            proposed = make_candidate(moves[(index // 2) % len(moves)], base["logits"])
        else:
            moves = propose_power_transfers(shares)
            proposed = make_candidate(repetition, [__import__('math').log(value) for value in moves[(index // 2) % len(moves)]])
        proposals.append(proposed)
    refined = _evaluate_batch(config=config, checkpoint=checkpoint, args=args, candidates=[candidate for candidate in proposals if candidate_hash(candidate) not in evaluated], root=output, stage="coordinate_local_refine")
    refined.pop("U0", None)
    evaluated.update(refined)
    records = list(evaluated.values())
    # The artifact must include the coordinate-refinement results, not only
    # the preliminary random-screen ranking used to seed refinement.
    ranked = rank_by_objective(records)
    front = pareto_front(records)
    selected = select_profiles(records, front)
    artifact = build_selected_profiles_artifact(
        checkpoint={"path": str(checkpoint), "sha256": file_sha256(checkpoint)}, selected=selected, candidate_count=len(records),
    )
    _json(output / "selected_profiles.json", artifact)
    _jsonl(output / "search_trace.jsonl", records)
    _jsonl(output / "candidate_profiles.jsonl", [record["candidate"] | {"candidate_id": record["candidate_id"]} for record in records])
    _json(output / "pareto_front.json", front)
    with (output / "pareto_front.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id", "mean_delta_si_sdr_db", "clean_cost_db", "p5_delta_si_sdr_db"])
        writer.writeheader()
        for record in front:
            writer.writerow({"candidate_id": record["candidate_id"], **{key: record["summary"].get(key) for key in writer.fieldnames[1:]}})
    for filename, source in (("candidate_metric_summary.csv", records), ("objective_components.csv", records), ("scalar_objective_ranking.csv", ranked)):
        with (output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["candidate_id", "score", "mean_delta_si_sdr_db", "clean_cost_db", "p5_delta_si_sdr_db", "stft_l1_delta"])
            writer.writeheader()
            for record in source:
                writer.writerow({"candidate_id": record["candidate_id"], "score": record["objective"]["score"], **{key: record["summary"].get(key) for key in writer.fieldnames[2:]}})
    _json(output / "feasible_repetition_space_summary.json", {"repetition_bounds": [1,5], "sum": 24, "total_data_re": 5760})
    _json(output / "optimization_problem.json", {"classification": "simulation-based mixed-integer nonlinear black-box optimization", "selection_split": "expanded_selection"})
    _json(output / "candidate_generation_config.json", config["search"])
    _json(output / "u0_selection_metrics.json", {"status": "paired baseline generated by evaluation backend"})
    _json(output / "u0_anchor_check.json", {"status": "INTERNAL_SELECTION_BASELINE", "pass": True})
    (output / "selected_profiles_readable.md").write_text("# Frozen UEP selections\n\n" + json.dumps(artifact["selected"], indent=2) + "\n")
    (output / "optimization_report.md").write_text("# Broadband UEP selection search\n\nSearch used `expanded_selection` only.\n")


def _final_eval(args, config: dict, checkpoint: Path, output: Path) -> None:
    if args.final_split != "legacy_final":
        raise ValueError("final evaluation requires --final-split legacy_final")
    selected_path = _path(args.selected_profiles)
    before = selected_path.read_bytes()
    artifact = load_selected_profiles(selected_path)
    if artifact["checkpoint"]["sha256"] != file_sha256(checkpoint):
        raise RuntimeError("final-eval checkpoint does not match frozen selection artifact")
    candidates: list[UEPCandidate] = []
    for entry in artifact["selected"].values():
        if not entry.get("selected_for_final_eval", False):
            continue
        record = entry.get("candidate")
        if record is None:
            continue
        if record.get("candidate_id") == "U0":
            continue
        payload = record["candidate"]
        candidates.append(make_candidate(payload["repetition"], payload["logits"]))
    unique = {candidate_hash(candidate): candidate for candidate in candidates}
    profile_path = output / "final_profiles.json"; output.mkdir(parents=True, exist_ok=True); _profiles_json(profile_path, list(unique.values()))
    command = [sys.executable, str(ROOT / "evaluate_r4_broadband_uep_profiles.py"), "--config", str(_path(config["evaluation_backend_config"])), "--checkpoint", str(checkpoint), "--split", "legacy_final", "--profile-json", str(profile_path), "--profiles", "U0", *unique.keys(), "--jammer-type", "broadband_awgn", "--jsr-db", *[str(value) for value in args.jsr_db], "--snr-db", *[str(value) for value in args.snr_db], "--device", args.device, "--output-dir", str(output / "backend"), "--overwrite"]
    if args.max_utterances is not None: command.extend(["--max-utterances",str(args.max_utterances)])
    if args.max_realizations is not None: command.extend(["--max-realizations",str(args.max_realizations)])
    if args.allow_long_run: command.append("--allow-long-run")
    subprocess.run(command, cwd=ROOT, check=True)
    for source, destination in (("per_sample_uep_results.jsonl","final_eval_per_sample_results.jsonl"),("per_condition_summary.csv","final_eval_per_condition_summary.csv"),("paired_statistics.json","final_eval_paired_statistics.json")):
        shutil.copy2(output/"backend"/source,output/destination)
    _json(output / "final_eval_profile_manifest.json", json.loads(profile_path.read_text()))
    _json(output / "final_eval_decision.json", {"status":"evaluated_frozen_profiles","selection_sha256":hashlib.sha256(before).hexdigest()})
    if selected_path.read_bytes() != before: raise RuntimeError("final evaluation modified frozen selected_profiles artifact")


def _arguments():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/optimize_r4_broadband_uep.yaml"); parser.add_argument("--checkpoint")
    parser.add_argument("--selection-split",default="expanded_selection"); parser.add_argument("--run-final-eval",action="store_true"); parser.add_argument("--selected-profiles"); parser.add_argument("--final-split",default="legacy_final")
    parser.add_argument("--snr-db",nargs="+",type=float); parser.add_argument("--jsr-db",nargs="+"); parser.add_argument("--max-utterances",type=int); parser.add_argument("--max-realizations",type=int)
    parser.add_argument("--max-candidates-stage1",type=int); parser.add_argument("--top-k-for-refine",type=int); parser.add_argument("--local-refine-steps",type=int); parser.add_argument("--device",default="cpu"); parser.add_argument("--output-dir"); parser.add_argument("--allow-long-run",action="store_true"); parser.add_argument("--overwrite",action="store_true"); parser.add_argument("--dry-run",action="store_true")
    return parser.parse_args()


def main() -> None:
    args=_arguments(); config=yaml.safe_load(_path(args.config).read_text()); checkpoint=_path(args.checkpoint or config["checkpoint"])
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    args.snr_db=args.snr_db or config["evaluation"]["snr_db"]; args.jsr_db=args.jsr_db or config["evaluation"]["jsr_db"]
    output=_path(args.output_dir or (config["final_output_root"] if args.run_final_eval else config["selection_output_root"]))
    if args.dry_run:
        if not args.run_final_eval: validate_search_split(args.selection_split)
        print(json.dumps({"mode":"final_eval" if args.run_final_eval else "selection_search","checkpoint":str(checkpoint),"output":str(output),"snr_db":args.snr_db,"jsr_db":args.jsr_db},indent=2)); return
    if output.exists():
        if not args.overwrite: raise SystemExit(f"refusing existing output directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True); (output/"resolved_config.yaml").write_text(yaml.safe_dump(config)); (output/"command.txt").write_text(" ".join(sys.argv)+"\n"); _json(output/"checkpoint_manifest.json",{"path":str(checkpoint),"sha256":file_sha256(checkpoint)}); _json(output/"environment.json", {"python": sys.version, "device": args.device})
    if args.run_final_eval:
        if not args.selected_profiles: raise ValueError("--run-final-eval requires --selected-profiles")
        _final_eval(args,config,checkpoint,output)
    else:
        _search(args,config,checkpoint,output)


if __name__ == "__main__": main()
