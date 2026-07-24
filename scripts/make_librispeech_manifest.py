from __future__ import annotations

import argparse
import json
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SPLITS = {
    "train": "train-clean-5",
    "valid": "dev-clean-2",
    "test": "test-clean",
}
_MISSING_FLAC_METADATA_WARNING_EMITTED = False


@dataclass(frozen=True)
class ManifestStats:
    utterances: int
    total_duration_sec: float
    average_duration_sec: float
    output_path: Path


def parse_transcript_file(path: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value:
            continue
        parts = value.split(maxsplit=1)
        utt_id = parts[0]
        transcripts[utt_id] = parts[1] if len(parts) > 1 else ""
    return transcripts


def _audio_info(path: Path) -> tuple[int | None, int | None, float | None]:
    try:
        import soundfile as sf
    except ImportError:
        sf = None
    if sf is not None:
        info = sf.info(str(path))
        frames = int(info.frames)
        sample_rate = int(info.samplerate)
        return sample_rate, frames, frames / sample_rate if sample_rate > 0 else None
    try:
        import torchaudio
    except ImportError:
        torchaudio = None
    if torchaudio is not None:
        info = torchaudio.info(str(path))
        frames = int(info.num_frames)
        sample_rate = int(info.sample_rate)
        return sample_rate, frames, frames / sample_rate if sample_rate > 0 else None
    if path.suffix.lower() != ".wav":
        global _MISSING_FLAC_METADATA_WARNING_EMITTED
        if not _MISSING_FLAC_METADATA_WARNING_EMITTED:
            print(
                "warning: soundfile/torchaudio is not installed; "
                "duration/sample_rate unavailable for FLAC files",
                file=sys.stderr,
            )
            _MISSING_FLAC_METADATA_WARNING_EMITTED = True
        return None, None, None
    with wave.open(str(path), "rb") as handle:
        sample_rate = int(handle.getframerate())
        frames = int(handle.getnframes())
    return sample_rate, frames, frames / sample_rate if sample_rate > 0 else None


def _display_path(path: Path, repo_root: Path, absolute_paths: bool) -> str:
    resolved = path.resolve()
    if absolute_paths:
        return str(resolved)
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _chapter_transcripts(split_dir: Path) -> dict[Path, dict[str, str]]:
    return {
        path.parent: parse_transcript_file(path)
        for path in sorted(split_dir.glob("*/*/*.trans.txt"))
    }


def build_manifest_for_split(
    librispeech_root: Path,
    out_dir: Path,
    *,
    split: str,
    split_dir_name: str,
    repo_root: Path,
    absolute_paths: bool = False,
) -> ManifestStats:
    split_dir = librispeech_root / split_dir_name
    if not split_dir.exists():
        raise FileNotFoundError(f"LibriSpeech split directory not found: {split_dir}")
    audio_paths = sorted(
        path
        for suffix in ("*.flac", "*.wav")
        for path in split_dir.glob(f"*/*/{suffix}")
        if path.is_file()
    )
    if not audio_paths:
        raise FileNotFoundError(f"no .flac or .wav files found under {split_dir}")

    transcripts_by_chapter = _chapter_transcripts(split_dir)
    seen_utt_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    total_duration = 0.0
    duplicate_count = 0
    missing_transcript_count = 0
    sample_rate_warnings = 0

    for audio_path in audio_paths:
        utt_id = audio_path.stem
        parts = utt_id.split("-")
        speaker_id = parts[0] if len(parts) >= 1 else ""
        chapter_id = parts[1] if len(parts) >= 2 else ""
        if utt_id in seen_utt_ids:
            duplicate_count += 1
            print(f"warning: duplicate utt_id {utt_id}", file=sys.stderr)
        seen_utt_ids.add(utt_id)

        chapter_transcripts = transcripts_by_chapter.get(audio_path.parent, {})
        text = chapter_transcripts.get(utt_id)
        if text is None:
            missing_transcript_count += 1
            print(f"warning: transcript missing for {utt_id}", file=sys.stderr)
            text = ""

        sample_rate, num_samples, duration_sec = _audio_info(audio_path)
        if sample_rate is not None and sample_rate != 16000:
            sample_rate_warnings += 1
            print(
                f"warning: {audio_path} sample_rate={sample_rate}, expected 16000",
                file=sys.stderr,
            )
        if duration_sec is not None:
            total_duration += float(duration_sec)

        records.append(
            {
                "utt_id": utt_id,
                "audio_path": _display_path(audio_path, repo_root, absolute_paths),
                "speaker_id": speaker_id,
                "chapter_id": chapter_id,
                "split": split,
                "text": text,
                "duration_sec": duration_sec,
                "sample_rate": sample_rate,
                "num_samples": num_samples,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{split}.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    if duplicate_count:
        print(f"warning: {split} contains {duplicate_count} duplicate utt_id entries", file=sys.stderr)
    if missing_transcript_count:
        print(
            f"warning: {split} contains {missing_transcript_count} records without transcripts",
            file=sys.stderr,
        )
    if sample_rate_warnings:
        print(
            f"warning: {split} contains {sample_rate_warnings} files not sampled at 16 kHz",
            file=sys.stderr,
        )
    average_duration = total_duration / len(records) if records else 0.0
    return ManifestStats(
        utterances=len(records),
        total_duration_sec=total_duration,
        average_duration_sec=average_duration,
        output_path=output_path,
    )


def build_manifests(
    librispeech_root: str | Path,
    out_dir: str | Path,
    *,
    repo_root: str | Path = ".",
    absolute_paths: bool = False,
) -> dict[str, ManifestStats]:
    root = Path(librispeech_root)
    destination = Path(out_dir)
    repo = Path(repo_root)
    for split_dir_name in SPLITS.values():
        split_dir = root / split_dir_name
        if not split_dir.exists():
            raise FileNotFoundError(f"LibriSpeech split directory not found: {split_dir}")
    return {
        split: build_manifest_for_split(
            root,
            destination,
            split=split,
            split_dir_name=split_dir_name,
            repo_root=repo,
            absolute_paths=absolute_paths,
        )
        for split, split_dir_name in SPLITS.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create JSONL manifests from Mini LibriSpeech directories."
    )
    parser.add_argument("--root", default="data/mini_librispeech/LibriSpeech")
    parser.add_argument("--out_dir", default="manifests/mini_librispeech")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute audio_path values instead of repo-root relative paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_manifests(
        args.root,
        args.out_dir,
        repo_root=args.repo_root,
        absolute_paths=args.absolute_paths,
    )
    for split, values in stats.items():
        print(
            f"{split}: utterances={values.utterances} "
            f"total_duration_sec={values.total_duration_sec:.2f} "
            f"average_duration_sec={values.average_duration_sec:.2f} "
            f"manifest={values.output_path}"
        )


if __name__ == "__main__":
    main()
