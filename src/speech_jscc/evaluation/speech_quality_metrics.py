"""Strict, optional objective speech metrics for fixed R4 evaluations.

No metric in this module has a surrogate fallback: a requested backend that is
not installed is an evaluation error rather than a silently different score.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf


class OptionalMetricDependencyError(RuntimeError):
    """Raised when a configured metric backend is unavailable."""


def _import_pystoi():
    try:
        return importlib.import_module("pystoi")
    except ModuleNotFoundError:
        return None


def _normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def compute_wer(reference_text: str, hypothesis_text: str) -> float:
    """Case/punctuation-normalized word error rate using Levenshtein distance."""
    reference = _normalized_words(reference_text)
    hypothesis = _normalized_words(hypothesis_text)
    if not reference:
        raise ValueError("WER reference transcript must contain at least one word")
    previous = list(range(len(hypothesis) + 1))
    for index, word in enumerate(reference, start=1):
        current = [index]
        for hyp_index, hyp_word in enumerate(hypothesis, start=1):
            current.append(min(
                previous[hyp_index] + 1,
                current[hyp_index - 1] + 1,
                previous[hyp_index - 1] + (word != hyp_word),
            ))
        previous = current
    return float(previous[-1] / len(reference))


def compute_estoi(reference: Sequence[float], estimate: Sequence[float], sample_rate: int) -> float:
    """Compute extended STOI; require the canonical pystoi implementation."""
    pystoi = _import_pystoi()
    if pystoi is None:
        raise OptionalMetricDependencyError(
            "ESTOI was requested but pystoi is unavailable; install speech-jscc[audio-metrics]"
        )
    value = float(pystoi.stoi(
        np.asarray(reference, dtype=np.float64).reshape(-1),
        np.asarray(estimate, dtype=np.float64).reshape(-1),
        int(sample_rate), extended=True,
    ))
    if not -1.0 <= value <= 1.0:
        raise FloatingPointError(f"nonfinite or out-of-range ESTOI: {value}")
    return value


def load_librispeech_transcript(sample_id: str | Path) -> str:
    """Read the exact transcript label corresponding to a LibriSpeech audio path."""
    sample = Path(sample_id)
    parts = sample.stem.split("-")
    if len(parts) < 3:
        raise ValueError(f"cannot infer LibriSpeech utterance id from {sample}")
    utterance_id = "-".join(parts[:3])
    transcript_path = sample.parent / f"{parts[0]}-{parts[1]}.trans.txt"
    if not transcript_path.is_file():
        raise FileNotFoundError(f"missing LibriSpeech transcript: {transcript_path}")
    for line in transcript_path.read_text().splitlines():
        identifier, separator, transcript = line.partition(" ")
        if identifier == utterance_id and separator:
            return transcript
    raise KeyError(f"transcript {utterance_id} is absent from {transcript_path}")


class FrozenWhisperTranscriber:
    """Frozen English ASR backend for a reproducible WER diagnostic."""

    def __init__(self, model_name: str, device: str) -> None:
        try:
            whisper = importlib.import_module("whisper")
        except ModuleNotFoundError as error:
            raise OptionalMetricDependencyError(
                "WER was requested but openai-whisper is unavailable; install speech-jscc[evaluation-metrics]"
            ) from error
        self._model = whisper.load_model(model_name, device=device)
        self.model_name = model_name
        self.device = device

    def transcribe(self, waveform: Sequence[float], sample_rate: int) -> str:
        if int(sample_rate) != 16_000:
            raise ValueError("Whisper WER backend requires 16 kHz input")
        result = self._model.transcribe(
            np.asarray(waveform, dtype=np.float32).reshape(-1),
            language="en", fp16=False, verbose=False, condition_on_previous_text=False,
        )
        return str(result["text"])


def compute_visqol(reference_path: Path, estimate_path: Path, binary: str) -> float:
    """Run the configured ViSQOL speech backend and parse MOS-LQO.

    ViSQOL's own speech-quality alignment is part of ViSQOL.  It does not
    change the project's explicitly unaligned SI-SDR policy.
    """
    try:
        result = subprocess.run(
            [binary, "--reference", str(reference_path), "--degraded", str(estimate_path), "--speech_mode"],
            check=True, capture_output=True, text=True,
        )
    except FileNotFoundError as error:
        raise OptionalMetricDependencyError(
            f"ViSQOL was requested but its binary is unavailable: {binary}"
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"ViSQOL failed: {error.stderr.strip()}") from error
    match = re.search(r"MOS-LQO:\s*([-+]?\d+(?:\.\d+)?)", result.stdout + "\n" + result.stderr)
    if match is None:
        raise RuntimeError("could not parse MOS-LQO from ViSQOL output")
    value = float(match.group(1))
    if not 1.0 <= value <= 5.0:
        raise FloatingPointError(f"invalid ViSQOL MOS-LQO: {value}")
    return value


@dataclass(frozen=True)
class MetricOptions:
    """Explicitly opt in to costly optional metrics; raw SI-SDR remains default."""

    estoi_enabled: bool = False
    wer_enabled: bool = False
    visqol_enabled: bool = False
    asr_model: str = "small.en"
    visqol_binary: str = "visqol"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MetricOptions":
        metrics = dict(config.get("speech_metrics", {}))
        return cls(
            estoi_enabled=bool(metrics.get("estoi_enabled", False)),
            wer_enabled=bool(metrics.get("wer_enabled", False)),
            visqol_enabled=bool(metrics.get("visqol_enabled", False)),
            asr_model=str(metrics.get("asr_model", "small.en")),
            visqol_binary=str(metrics.get("visqol_binary", "visqol")),
        )


def metric_backend_metadata(options: MetricOptions) -> dict[str, Any]:
    """Record exactly which optional metric backends a result used."""
    def version(distribution: str) -> str | None:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "raw_waveform_only": True,
        "estoi": {"enabled": options.estoi_enabled, "backend": "pystoi", "version": version("pystoi")},
        "wer": {
            "enabled": options.wer_enabled,
            "backend": "openai-whisper",
            "version": version("openai-whisper"),
            "model": options.asr_model,
            "frozen": True,
        },
        "visqol": {
            "enabled": options.visqol_enabled,
            "backend": options.visqol_binary,
            "package": "visqol-python",
            "version": version("visqol-python"),
            "mode": "speech_mode",
        },
    }


class RawSpeechMetricComputer:
    """Compute configured metrics strictly on a raw decoded waveform."""

    def __init__(self, options: MetricOptions, *, device: str) -> None:
        self.options = options
        self.transcriber = FrozenWhisperTranscriber(options.asr_model, device) if options.wer_enabled else None

    def evaluate(
        self,
        *, reference: Sequence[float], estimate: Sequence[float], sample_rate: int,
        sample_id: str, workspace: Path,
    ) -> dict[str, float]:
        reference_array = np.asarray(reference, dtype=np.float32).reshape(-1)
        estimate_array = np.asarray(estimate, dtype=np.float32).reshape(-1)
        if reference_array.shape != estimate_array.shape:
            raise ValueError("speech metrics require equal-length reference and estimate")
        result: dict[str, float] = {}
        if self.options.estoi_enabled:
            result["raw_estoi"] = compute_estoi(reference_array, estimate_array, sample_rate)
        if self.transcriber is not None:
            result["raw_wer"] = compute_wer(
                load_librispeech_transcript(sample_id),
                self.transcriber.transcribe(estimate_array, sample_rate),
            )
        if self.options.visqol_enabled:
            workspace.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=workspace) as temporary:
                root = Path(temporary)
                reference_path, estimate_path = root / "reference.wav", root / "estimate.wav"
                sf.write(reference_path, reference_array, sample_rate, subtype="FLOAT")
                sf.write(estimate_path, estimate_array, sample_rate, subtype="FLOAT")
                result["raw_visqol_mos_lqo"] = compute_visqol(reference_path, estimate_path, self.options.visqol_binary)
        if not all(np.isfinite(value) for value in result.values()):
            raise FloatingPointError(f"nonfinite speech metric: {result}")
        return result
