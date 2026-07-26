from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class ManifestEntry:
    source_path: str
    speaker_id: str
    utterance_id: str
    source_split: str
    assigned_evaluation_role: str
    crop_start_sample: int
    crop_num_samples: int
    source_sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_source_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    repository_root = manifest_path.resolve().parents[2]
    return repository_root / path


def build_selection_manifest(
    manifest_path: str | Path,
    *,
    count: int,
    seed: int,
    crop_num_samples: int,
    hash_files: bool = True,
) -> list[ManifestEntry]:
    """Select test-clean utterances round-robin by speaker for validation only."""
    manifest_path = Path(manifest_path)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        grouped[str(row["speaker_id"])].append(row)
    if not grouped:
        raise ValueError(f"empty selection source manifest: {manifest_path}")

    rng = random.Random(int(seed))
    speakers = sorted(grouped)
    rng.shuffle(speakers)
    for speaker, values in grouped.items():
        values.sort(key=lambda item: str(item["utt_id"]))
        random.Random(
            int(seed)
            ^ int(hashlib.sha256(speaker.encode()).hexdigest()[:8], 16)
        ).shuffle(values)

    selected: list[dict] = []
    offsets = {speaker: 0 for speaker in speakers}
    while len(selected) < int(count):
        progressed = False
        for speaker in speakers:
            index = offsets[speaker]
            if index < len(grouped[speaker]):
                selected.append(grouped[speaker][index])
                offsets[speaker] += 1
                progressed = True
                if len(selected) == int(count):
                    break
        if not progressed:
            break
    if len(selected) != int(count):
        raise ValueError(
            f"selection manifest has only {len(selected)} usable rows; requested {count}"
        )

    output = []
    for row in selected:
        source = _resolve_source_path(manifest_path, str(row["audio_path"]))
        if hash_files and not source.is_file():
            raise FileNotFoundError(source)
        source_rate = int(row.get("sample_rate", 16000))
        source_samples = round(
            int(row.get("num_samples", crop_num_samples)) * 16000 / source_rate
        )
        output.append(
            ManifestEntry(
                source_path=str(source),
                speaker_id=str(row["speaker_id"]),
                utterance_id=str(row["utt_id"]),
                source_split="test-clean",
                assigned_evaluation_role="selection_validation",
                crop_start_sample=max((source_samples - int(crop_num_samples)) // 2, 0),
                crop_num_samples=int(crop_num_samples),
                source_sha256=file_sha256(source) if hash_files else "not_computed",
            )
        )
    return output


def entries_from_paths(
    paths: Iterable[str | Path],
    *,
    source_split: str,
    role: str,
    crop_num_samples: int,
    hash_files: bool = True,
) -> list[ManifestEntry]:
    output = []
    for value in paths:
        path = Path(value).resolve()
        parts = path.stem.split("-")
        speaker = parts[0] if len(parts) >= 3 else "unknown"
        output.append(
            ManifestEntry(
                source_path=str(path),
                speaker_id=speaker,
                utterance_id=path.stem,
                source_split=source_split,
                assigned_evaluation_role=role,
                crop_start_sample=0,
                crop_num_samples=int(crop_num_samples),
                source_sha256=file_sha256(path) if hash_files else "not_computed",
            )
        )
    return output


def _overlap(left: Sequence[ManifestEntry], right: Sequence[ManifestEntry], field: str):
    return sorted(
        {getattr(entry, field) for entry in left}
        & {getattr(entry, field) for entry in right}
    )


def audit_protocol_overlap(
    train: Sequence[ManifestEntry],
    selection: Sequence[ManifestEntry],
    final: Sequence[ManifestEntry],
) -> dict:
    groups = {"train": train, "selection": selection, "final": final}
    speaker_overlaps = {}
    utterance_overlaps = {}
    duplicate_counts = {}
    for name, values in groups.items():
        ids = [entry.utterance_id for entry in values]
        duplicate_counts[name] = len(ids) - len(set(ids))
        if duplicate_counts[name]:
            raise ValueError(f"{name} contains duplicate utterances")
    for left, right in (
        ("train", "selection"),
        ("train", "final"),
        ("selection", "final"),
    ):
        name = f"{left}_{right}"
        speakers = _overlap(groups[left], groups[right], "speaker_id")
        utterances = _overlap(groups[left], groups[right], "utterance_id")
        speaker_overlaps[name] = len(speakers)
        utterance_overlaps[name] = len(utterances)
        if speakers or utterances:
            raise ValueError(
                f"{name} overlap: speakers={speakers}, utterances={utterances}"
            )
    return {
        "passed": True,
        "speaker_overlap_counts": speaker_overlaps,
        "utterance_overlap_counts": utterance_overlaps,
        "duplicate_utterance_counts": duplicate_counts,
        "speaker_counts": {
            name: len({entry.speaker_id for entry in values})
            for name, values in groups.items()
        },
        "utterance_counts": {name: len(values) for name, values in groups.items()},
    }


def explicit_metric_row(*, candidate: dict, initial: dict, clean: dict) -> dict:
    clean_stft = max(float(clean["stft_l1"]), 1e-12)
    initial_stft = max(float(initial["stft_l1"]), 1e-12)
    return {
        "si_sdr_absolute_db": float(candidate["si_sdr_db"]),
        "delta_si_sdr_vs_clean_codec_db": float(candidate["si_sdr_db"])
        - float(clean["si_sdr_db"]),
        "delta_si_sdr_vs_initial_r4_db": float(candidate["si_sdr_db"])
        - float(initial["si_sdr_db"]),
        "waveform_snr_absolute_db": float(candidate["waveform_snr_db"]),
        "delta_waveform_snr_vs_clean_codec_db": float(
            candidate["waveform_snr_db"]
        )
        - float(clean["waveform_snr_db"]),
        "delta_waveform_snr_vs_initial_r4_db": float(
            candidate["waveform_snr_db"]
        )
        - float(initial["waveform_snr_db"]),
        "stft_l1_absolute": float(candidate["stft_l1"]),
        "stft_ratio_vs_clean_codec": float(candidate["stft_l1"]) / clean_stft,
        "delta_stft_ratio_vs_initial_r4": float(candidate["stft_l1"])
        / initial_stft
        - 1.0,
    }


def paired_statistics(
    values: Sequence[float], *, bootstrap_samples: int, bootstrap_seed: int
) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("paired statistics require nonempty finite values")
    ddof = 1 if array.size > 1 else 0
    generator = np.random.default_rng(int(bootstrap_seed))
    indices = generator.integers(
        0, array.size, size=(int(bootstrap_samples), array.size)
    )
    bootstrap_means = array[indices].mean(axis=1)
    return {
        "sample_count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "standard_deviation": float(array.std(ddof=ddof)),
        "standard_error": float(array.std(ddof=ddof) / math.sqrt(array.size)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "improved_fraction": float(np.mean(array > 0)),
        "improved_by_at_least_0_5_fraction": float(np.mean(array >= 0.5)),
        "degraded_by_at_least_0_5_fraction": float(np.mean(array <= -0.5)),
        "paired_mean_bootstrap_ci95": [
            float(np.percentile(bootstrap_means, 2.5)),
            float(np.percentile(bootstrap_means, 97.5)),
        ],
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
    }


def build_seed_manifest(
    *,
    utterance_ids: Sequence[str],
    snr_db: Sequence[float],
    realization_seeds: Sequence[int],
) -> dict:
    conditions = []
    for snr in map(float, snr_db):
        for realization, base_seed in enumerate(realization_seeds):
            channel_seed = int(base_seed) + round(snr * 100)
            for tti, utterance_id in enumerate(utterance_ids):
                noise_seed = int(base_seed) + tti + round(snr * 1000)
                identity = (
                    f"snr={snr:g}|realization={realization}|tti={tti}|"
                    f"utterance={utterance_id}"
                )
                conditions.append(
                    {
                        "condition_id": hashlib.sha256(identity.encode()).hexdigest(),
                        "utterance_id": str(utterance_id),
                        "snr_db": snr,
                        "realization": realization,
                        "tti": tti,
                        "channel_seed": channel_seed,
                        "noise_seed": noise_seed,
                        "pilot_seed": 0,
                        "bootstrap_policy": "uniform_tti0",
                    }
                )
    return {
        "schema_version": "r4_expanded_validation_seed_v1",
        "candidate_invariant": True,
        "conditions": conditions,
    }


def checkpoint_gate(by_snr: dict[str, dict]) -> dict:
    five = by_snr["5.0"]
    margins = {
        "5db_si_sdr_vs_clean_codec": (
            float(five["delta_si_sdr_vs_clean_codec_db"]) + 1.0
        )
        / 1.0,
        "5db_waveform_snr_vs_clean_codec": (
            float(five["delta_waveform_snr_vs_clean_codec_db"]) + 1.0
        )
        / 1.0,
        "5db_stft_ratio_vs_clean_codec": (
            1.20 - float(five["stft_ratio_vs_clean_codec"])
        )
        / 0.20,
    }
    for snr in ("10.0", "15.0"):
        values = by_snr[snr]
        margins[f"{snr}_si_sdr_vs_initial"] = (
            float(values["delta_si_sdr_vs_initial_r4_db"]) + 0.5
        ) / 0.5
        margins[f"{snr}_waveform_snr_vs_initial"] = (
            float(values["delta_waveform_snr_vs_initial_r4_db"]) + 0.5
        ) / 0.5
        margins[f"{snr}_stft_ratio_vs_initial"] = (
            0.05 - float(values["delta_stft_ratio_vs_initial_r4"])
        ) / 0.05
    minimum = min(margins.values())
    return {
        "passed": minimum >= 0,
        "normalized_margins": margins,
        "normalized_minimum_margin": minimum,
    }


def prepare_output_directory(path: str | Path, *, overwrite: bool) -> Path:
    output = Path(path)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"refusing existing output directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    return output


def prepare_final_test_directory(
    selection_output: str | Path, *, overwrite: bool
) -> Path:
    output = Path(selection_output)
    required = (
        output / "selection_decision.json",
        output / "best_expanded_validation.pt",
        output / "final_test_manifest_reference.jsonl",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "final test requires completed expanded selection artifacts: "
            + ", ".join(missing)
        )
    final = output / "final_test"
    if final.exists():
        if not overwrite:
            raise FileExistsError(f"refusing existing output directory: {final}")
        shutil.rmtree(final)
    final.mkdir()
    return final


def utterance_level_rows(
    rows: Sequence[dict], *, metric_keys: Sequence[str]
) -> list[dict]:
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["utterance_id"]), float(row["snr_db"]))].append(row)
    output = []
    for utterance_id, snr in sorted(grouped):
        members = grouped[(utterance_id, snr)]
        output.append(
            {
                "utterance_id": utterance_id,
                "snr_db": snr,
                **{
                    key: sum(float(row[key]) for row in members) / len(members)
                    for key in metric_keys
                },
            }
        )
    return output


def _checkpoint_step(path: Path) -> int | None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    value = payload.get("global_step")
    return int(value) if value is not None else None


def discover_candidates(run_dir: str | Path, *, top_k: int = 10) -> dict:
    run_dir = Path(run_dir)
    named = (
        "best_5db_si_sdr.pt",
        "best_clean_gate.pt",
        "best_validation_average.pt",
        "last.pt",
    )
    included_paths: list[Path] = [run_dir / name for name in named if (run_dir / name).is_file()]
    included_paths.extend(sorted(run_dir.glob("checkpoint_step_*.pt")))
    included_paths.extend(sorted(run_dir.glob("step_*.pt")))
    unique_paths = list(dict.fromkeys(path.resolve() for path in included_paths))
    included = []
    existing_steps = set()
    for path in unique_paths:
        step = _checkpoint_step(path)
        if step is not None:
            existing_steps.add(step)
        included.append(
            {
                "checkpoint_path": str(path),
                "global_step": step,
                "training_stage": torch.load(
                    path, map_location="cpu", weights_only=False
                ).get("curriculum_stage"),
                "candidate_source": (
                    "named" if path.name in named else "intermediate"
                ),
            }
        )

    validation_scores = []
    for path in sorted(run_dir.glob("validation_step_*.json")):
        payload = json.loads(path.read_text())
        step = int(path.stem.rsplit("_", 1)[-1])
        validation_scores.append(
            (float(payload.get("5db_delta_si_sdr_vs_initial_r4", -math.inf)), step)
        )
    requested = {step for _, step in sorted(validation_scores, reverse=True)[:top_k]}
    requested.update(range(1000, 20001, 1000))
    requested.update((4000, 12000, 20000, 5750))
    return {
        "included": included,
        "missing_historical_steps": sorted(requested - existing_steps),
        "historical_checkpoint_limitation": (
            "validation JSON does not contain model weights; missing steps cannot be "
            "reconstructed"
        ),
    }


def _five_metric(summary: dict) -> dict:
    return summary["by_snr"]["5.0"]["utterance_level"][
        "delta_si_sdr_vs_initial_r4_db"
    ]


def rank_candidates(summaries: Sequence[dict]) -> list[dict]:
    if not summaries:
        raise ValueError("no checkpoint summaries to rank")
    any_passing = any(bool(row["clean_gate_pass"]) for row in summaries)

    def key(row: dict):
        five = _five_metric(row)
        average = sum(
            float(
                row["by_snr"][str(snr)]["utterance_level"][
                    "delta_si_sdr_vs_initial_r4_db"
                ]["mean"]
            )
            for snr in (5.0, 10.0, 15.0)
        ) / 3
        if any_passing:
            return (
                bool(row["clean_gate_pass"]),
                float(five["mean"]),
                float(five["p10"]),
                average,
                -float(five["standard_deviation"]),
            )
        return (
            float(row["gate_normalized_minimum_margin"]),
            float(five["mean"]),
            float(five["p10"]),
        )

    ranked = [dict(row) for row in sorted(summaries, key=key, reverse=True)]
    for index, row in enumerate(ranked, 1):
        row["rank"] = index
        row["selection_status"] = (
            "passing_candidate"
            if row["clean_gate_pass"]
            else "best_nonpassing_candidate"
        )
    return ranked


def write_selected_checkpoint(
    source: str | Path, destination: str | Path, decision: dict
) -> None:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    metadata = dict(decision)
    metadata["clean_gate_pass"] = bool(decision["clean_gate_pass"])
    if not metadata["clean_gate_pass"]:
        metadata["selection_status"] = "best_nonpassing_candidate"
        payload.pop("passing_checkpoint", None)
    payload["expanded_validation"] = metadata
    torch.save(payload, destination)


def shared_report_for_candidate(precomputed_report, *, candidate_generated_report):
    """Return the immutable initial-model report, never candidate state."""
    del candidate_generated_report
    return precomputed_report


__all__ = [
    "ManifestEntry",
    "audit_protocol_overlap",
    "build_seed_manifest",
    "build_selection_manifest",
    "checkpoint_gate",
    "discover_candidates",
    "entries_from_paths",
    "explicit_metric_row",
    "file_sha256",
    "paired_statistics",
    "prepare_final_test_directory",
    "prepare_output_directory",
    "rank_candidates",
    "shared_report_for_candidate",
    "utterance_level_rows",
    "write_selected_checkpoint",
]
