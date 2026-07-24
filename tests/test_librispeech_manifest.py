import json
import wave
from pathlib import Path

import pytest

from speech_jscc.data import load_waveform_segment, resolve_waveform_splits


def _write_wav(path: Path, sample_rate: int = 16000, frames: int = 1600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)


def _build_librispeech_fixture(root: Path) -> None:
    for split_dir, utt_id, text in [
        ("train-clean-5", "1089-134686-0000", "CHAPTER ONE"),
        ("dev-clean-2", "1272-135031-0000", "VALID TEXT"),
        ("test-clean", "121-123852-0000", "TEST TEXT"),
    ]:
        speaker_id, chapter_id, _ = utt_id.split("-")
        chapter = root / split_dir / speaker_id / chapter_id
        _write_wav(chapter / f"{utt_id}.wav")
        (chapter / f"{speaker_id}-{chapter_id}.trans.txt").write_text(
            f"{utt_id} {text}\n", encoding="utf-8"
        )


def test_manifest_generation_matches_librispeech_transcripts(tmp_path):
    from scripts.make_librispeech_manifest import build_manifests

    librispeech_root = tmp_path / "data" / "mini_librispeech" / "LibriSpeech"
    output_dir = tmp_path / "manifests" / "mini_librispeech"
    _build_librispeech_fixture(librispeech_root)

    stats = build_manifests(librispeech_root, output_dir, repo_root=tmp_path)

    assert set(stats) == {"train", "valid", "test"}
    for split, expected_text in [
        ("train", "CHAPTER ONE"),
        ("valid", "VALID TEXT"),
        ("test", "TEST TEXT"),
    ]:
        manifest_path = output_dir / f"{split}.jsonl"
        records = [
            json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
        ]
        assert len(records) == 1
        record = records[0]
        assert {
            "utt_id",
            "audio_path",
            "speaker_id",
            "chapter_id",
            "split",
            "text",
            "duration_sec",
            "sample_rate",
            "num_samples",
        }.issubset(record)
        assert record["split"] == split
        assert record["text"] == expected_text
        assert record["audio_path"].startswith("data/mini_librispeech/LibriSpeech/")
        assert record["sample_rate"] == 16000
        assert record["num_samples"] == 1600
        assert record["duration_sec"] == pytest.approx(0.1)


def test_manifest_generation_errors_for_missing_split(tmp_path):
    from scripts.make_librispeech_manifest import build_manifests

    librispeech_root = tmp_path / "LibriSpeech"
    (librispeech_root / "train-clean-5").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="dev-clean-2"):
        build_manifests(librispeech_root, tmp_path / "manifests", repo_root=tmp_path)


def test_jsonl_manifests_are_resolved_by_audio_path(tmp_path):
    audio = tmp_path / "audio.wav"
    _write_wav(audio)
    train_manifest = tmp_path / "train.jsonl"
    valid_manifest = tmp_path / "valid.jsonl"
    train_manifest.write_text(
        json.dumps({"utt_id": "a", "audio_path": "audio.wav"}) + "\n", encoding="utf-8"
    )
    valid_manifest.write_text(
        json.dumps({"utt_id": "b", "audio_path": str(audio)}) + "\n", encoding="utf-8"
    )

    train_paths, valid_paths = resolve_waveform_splits(
        {"train_manifest": train_manifest, "val_manifest": valid_manifest}, seed=0
    )

    assert train_paths == [audio]
    assert valid_paths == [audio]


def test_soundfile_flac_manifest_audio_can_be_loaded(tmp_path):
    sf = pytest.importorskip("soundfile")
    audio = tmp_path / "audio.flac"
    sf.write(str(audio), [0.0] * 1600, 16000)

    waveform = load_waveform_segment(audio, sample_rate=16000, waveform_samples=800)

    assert waveform.shape == (800,)
