from __future__ import annotations

import argparse
import json
import struct
import time
import wave
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from speech_jscc.codecs import SpeechTokenizerWrapper


def load_wave(path: Path) -> tuple[torch.Tensor, int]:
    payload = path.read_bytes()
    if payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise ValueError("input is not a RIFF/WAVE file")
    offset, fmt, frames = 12, None, None
    while offset + 8 <= len(payload):
        chunk_id = payload[offset:offset + 4]
        chunk_size = struct.unpack_from("<I", payload, offset + 4)[0]
        chunk = payload[offset + 8:offset + 8 + chunk_size]
        if chunk_id == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", chunk)
        elif chunk_id == b"data":
            frames = chunk
        offset += 8 + chunk_size + (chunk_size % 2)
    if fmt is None or frames is None:
        raise ValueError("WAV file is missing fmt or data chunk")
    audio_format, channels, sample_rate, _, _, bits = fmt
    if audio_format == 1 and bits == 16:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif audio_format == 3 and bits == 32:
        samples = np.frombuffer(frames, dtype="<f4").astype(np.float32)
    else:
        raise ValueError(f"unsupported WAV format={audio_format}, bits={bits}")
    samples = samples.reshape(-1, channels)[:, 0]
    return torch.from_numpy(samples.copy()), sample_rate


def write_pcm_wave(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (waveform.detach().cpu().clamp(-1.0, 1.0).numpy() * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate real SpeechTokenizer continuous embeddings")
    parser.add_argument(
        "--config-path",
        default="artifacts/codecs/speechtokenizer/speechtokenizer_hubert_avg/config.json",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="artifacts/codecs/speechtokenizer/speechtokenizer_hubert_avg/SpeechTokenizer.pt",
    )
    parser.add_argument("--audio", default="SpeechTokenizer/samples/example_input.wav")
    parser.add_argument(
        "--output", default="artifacts/codecs/speechtokenizer/reconstruction.wav"
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    waveform, sample_rate = load_wave(Path(args.audio))
    device = torch.device(args.device)
    with Path(args.config_path).open("r", encoding="utf-8") as handle:
        model_sample_rate = int(json.load(handle)["sample_rate"])
    start = time.perf_counter()
    if sample_rate != model_sample_rate:
        target_length = round(waveform.numel() * model_sample_rate / sample_rate)
        waveform = F.interpolate(
            waveform.reshape(1, 1, -1), size=target_length, mode="linear", align_corners=False
        ).reshape(-1)
    codec = SpeechTokenizerWrapper(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        waveform_samples=waveform.numel(),
        fallback_to_mock=False,
    ).to(device)
    load_seconds = time.perf_counter() - start
    batch = waveform.to(device).unsqueeze(0)

    with torch.inference_mode():
        started = time.perf_counter()
        representation = codec.encode_waveform(batch)
        encode_seconds = time.perf_counter() - started
        started = time.perf_counter()
        reconstruction = codec.decode_representation(representation)
        decode_seconds = time.perf_counter() - started
        codes = codec.model.encode(batch.unsqueeze(1), n_q=codec.n_q)
        official_reconstruction = codec.model.decode(codes).squeeze(1)

    error = batch - reconstruction
    mse = error.square().mean()
    reconstruction_snr = 10.0 * torch.log10(
        batch.square().sum() / error.square().sum().clamp_min(1e-12)
    )
    metrics = {
        "sample_rate": codec.sample_rate,
        "duration_seconds": batch.shape[-1] / codec.sample_rate,
        "representation_shape": list(representation.shape),
        "code_shape": list(codes.shape),
        "waveform_shape": list(reconstruction.shape),
        "mse": mse.item(),
        "reconstruction_snr_db": reconstruction_snr.item(),
        "continuous_vs_official_max_abs_diff": (
            reconstruction - official_reconstruction[..., : reconstruction.shape[-1]]
        ).abs().max().item(),
        "model_load_seconds": load_seconds,
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
    }
    write_pcm_wave(Path(args.output), reconstruction[0], codec.sample_rate)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
