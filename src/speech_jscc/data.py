from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def synthetic_waveforms(
    batch_size: int,
    samples: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Generate speech-like harmonic mixtures for a dependency-free smoke run."""
    time = torch.linspace(0.0, 1.0, samples, device=device)[None, :]
    f0 = torch.empty(batch_size, 1, device=device).uniform_(2.0, 8.0, generator=generator)
    phase = torch.empty(batch_size, 1, device=device).uniform_(
        0.0, 2.0 * math.pi, generator=generator
    )
    waveform = torch.sin(2.0 * math.pi * f0 * time + phase)
    waveform += 0.35 * torch.sin(4.0 * math.pi * f0 * time + 0.5 * phase)
    return waveform / waveform.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)


def _read_manifest(path: str | Path) -> list[Path]:
    manifest = Path(path)
    paths: list[Path] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        item = Path(value)
        paths.append(item if item.is_absolute() else manifest.parent / item)
    return paths


def resolve_waveform_splits(data_config: dict[str, Any], seed: int) -> tuple[list[Path], list[Path]]:
    """Resolve explicit manifests or make a deterministic directory split."""
    train_manifest = data_config.get("train_manifest")
    val_manifest = data_config.get("val_manifest")
    if train_manifest or val_manifest:
        if not train_manifest or not val_manifest:
            raise ValueError("data.train_manifest and data.val_manifest must be provided together")
        return _read_manifest(train_manifest), _read_manifest(val_manifest)

    configured = data_config.get("waveform_paths")
    if configured is not None:
        paths = [Path(value) for value in configured]
    elif data_config.get("waveform_dir"):
        root = Path(data_config["waveform_dir"])
        pattern = data_config.get("waveform_glob", "**/*.wav")
        paths = sorted(path for path in root.glob(pattern) if path.is_file())
    else:
        return [], []
    if not paths:
        raise ValueError("no waveform files matched the configured data source")

    fraction = float(data_config.get("val_fraction", 0.1))
    if not 0.0 < fraction < 1.0:
        raise ValueError("data.val_fraction must be between 0 and 1")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(paths), generator=generator).tolist()
    shuffled = [paths[index] for index in order]
    val_count = max(1, round(len(paths) * fraction))
    if val_count >= len(paths):
        raise ValueError("waveform split needs at least two files")
    return shuffled[val_count:], shuffled[:val_count]


def codec_cache_namespace(config: dict[str, Any], codec) -> str:
    """Identify cache entries by codec type, layer count, and checkpoint revision."""
    codec_config = config.get("codec", {})
    checkpoint_value = codec_config.get("checkpoint_path")
    checkpoint_identity = "builtin"
    if checkpoint_value:
        checkpoint_path = Path(checkpoint_value)
        checkpoint_identity = str(checkpoint_path.resolve())
        if checkpoint_path.exists():
            stat = checkpoint_path.stat()
            checkpoint_identity += f"-{stat.st_size}-{stat.st_mtime_ns}"
    return (
        f"{codec_config.get('type', 'mock')}-"
        f"{codec_config.get('n_q', codec.representation_shape[0])}-"
        f"{checkpoint_identity}"
    )


def load_waveform_segment(
    path: str | Path,
    sample_rate: int,
    waveform_samples: int,
) -> Tensor:
    """Load mono audio and deterministically center crop/pad one training segment."""
    path = Path(path)
    if path.suffix.lower() in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict):
            waveform = payload["waveform"]
            source_rate = int(payload.get("sample_rate", sample_rate))
        else:
            waveform, source_rate = payload, sample_rate
        waveform = torch.as_tensor(waveform, dtype=torch.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
    else:
        try:
            import torchaudio
        except ImportError:
            payload = path.read_bytes()
            if payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
                raise ValueError(f"{path} is not a RIFF/WAVE file")
            offset, fmt, frame_bytes = 12, None, None
            while offset + 8 <= len(payload):
                chunk_id = payload[offset : offset + 4]
                chunk_size = struct.unpack_from("<I", payload, offset + 4)[0]
                chunk = payload[offset + 8 : offset + 8 + chunk_size]
                if chunk_id == b"fmt ":
                    fmt = struct.unpack_from("<HHIIHH", chunk)
                elif chunk_id == b"data":
                    frame_bytes = bytearray(chunk)
                offset += 8 + chunk_size + (chunk_size % 2)
            if fmt is None or frame_bytes is None:
                raise ValueError(f"{path} is missing WAV fmt or data")
            audio_format, channels, source_rate, _, _, bits = fmt
            if audio_format == 1 and bits == 8:
                waveform = (
                    torch.frombuffer(frame_bytes, dtype=torch.uint8).float() - 128.0
                ) / 128.0
            elif audio_format == 1 and bits == 16:
                waveform = torch.frombuffer(frame_bytes, dtype=torch.int16).float() / 32768.0
            elif audio_format == 1 and bits == 32:
                waveform = torch.frombuffer(frame_bytes, dtype=torch.int32).float() / 2147483648.0
            elif audio_format == 3 and bits == 32:
                waveform = torch.frombuffer(frame_bytes, dtype=torch.float32).float()
            else:
                raise ValueError(
                    f"unsupported WAV format={audio_format}, bits={bits} in {path}"
                )
            waveform = waveform.reshape(-1, channels).mean(dim=1)
        else:
            waveform, source_rate = torchaudio.load(str(path))
            waveform = waveform.float().mean(dim=0)
    if waveform.ndim != 1:
        raise ValueError(f"waveform {path} must resolve to one dimension")
    if source_rate != sample_rate:
        target_length = round(waveform.numel() * sample_rate / source_rate)
        waveform = F.interpolate(
            waveform.reshape(1, 1, -1),
            size=target_length,
            mode="linear",
            align_corners=False,
        ).reshape(-1)
    if waveform.numel() > waveform_samples:
        start = (waveform.numel() - waveform_samples) // 2
        waveform = waveform[start : start + waveform_samples]
    elif waveform.numel() < waveform_samples:
        waveform = F.pad(waveform, (0, waveform_samples - waveform.numel()))
    return waveform.contiguous()


class CachedCodecDataset:
    """Waveform-to-continuous-latent dataset with an optional disk cache."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        codec,
        *,
        sample_rate: int,
        waveform_samples: int,
        device: torch.device,
        split: str,
        cache_dir: str | Path | None = None,
        cache_namespace: str = "codec",
    ) -> None:
        self.paths = [Path(path) for path in paths]
        self.codec = codec
        self.sample_rate = int(sample_rate)
        self.waveform_samples = int(waveform_samples)
        self.device = device
        self.split = split
        self.cache_dir = Path(cache_dir) / split if cache_dir else None
        self.cache_namespace = cache_namespace
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.paths)

    def _cache_path(self, index: int) -> Path | None:
        if self.cache_dir is None:
            return None
        identity = (
            f"{self.paths[index].resolve()}|{self.paths[index].stat().st_mtime_ns}|"
            f"{self.cache_namespace}|{self.sample_rate}|{self.waveform_samples}"
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{digest}.pt"

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        waveform = load_waveform_segment(
            self.paths[index], self.sample_rate, self.waveform_samples
        )
        cache_path = self._cache_path(index)
        if cache_path is not None and cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            representation = payload["representation"]
        else:
            with torch.no_grad():
                representation = self.codec.encode_waveform(
                    waveform.to(self.device).unsqueeze(0)
                )[0].cpu()
            if cache_path is not None:
                torch.save(
                    {
                        "representation": representation,
                        "source": str(self.paths[index]),
                        "sample_rate": self.sample_rate,
                        "waveform_samples": self.waveform_samples,
                        "cache_namespace": self.cache_namespace,
                    },
                    cache_path,
                )
        return representation.to(self.device), waveform.to(self.device)
