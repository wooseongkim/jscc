from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import time
import wave
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from torch import Tensor

from speech_jscc.codecs import MockContinuousCodec, SpeechTokenizerWrapper
from speech_jscc.config import resolve_device
from speech_jscc.data import load_waveform_segment


CLEAN_METRIC_KEYS = ("waveform_mse", "waveform_l1", "waveform_snr_db", "si_sdr_db", "stft_l1")
BASELINE_PROTOCOLS = (
    ("A", "none", False),
    ("B", "none", True),
    ("C", "peak_xcorr", False),
    ("D", "peak_xcorr", True),
)
BASELINE_TABLE_METRICS = (
    "waveform_snr_db",
    "si_sdr_db",
    "stft_l1",
    "best_lag_samples",
    "input_rms",
    "output_rms",
)


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def read_manifest(path: str | Path, max_items: int | None = None) -> list[dict[str, Any]]:
    manifest = Path(path)
    records: list[dict[str, Any]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        if manifest.suffix.lower() == ".jsonl":
            record = json.loads(value)
        else:
            record = {"audio_path": value}
        if "audio_path" not in record:
            raise ValueError(f"{manifest} record is missing audio_path")
        audio_path = Path(str(record["audio_path"]))
        if not audio_path.is_absolute():
            manifest_relative = manifest.parent / audio_path
            cwd_relative = Path.cwd() / audio_path
            audio_path = manifest_relative if manifest_relative.exists() else cwd_relative
        record = dict(record)
        record["audio_path"] = str(audio_path)
        records.append(record)
        if max_items is not None and len(records) >= max_items:
            break
    if not records:
        raise ValueError(f"no records found in {manifest}")
    missing = [record["audio_path"] for record in records if not Path(record["audio_path"]).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest audio files do not exist; first={missing[0]}")
    return records


def build_codec_only(config: dict[str, Any], device: torch.device):
    codec_cfg = config["codec"]
    model_cfg = config.get("model", {})
    codec_type = codec_cfg.get("type", "mock").lower()
    if codec_type == "mock":
        codec = MockContinuousCodec(
            int(model_cfg.get("layers", codec_cfg.get("layers", 4))),
            int(model_cfg.get("frames", codec_cfg.get("frames", 12))),
            int(model_cfg.get("latent_dim", codec_cfg.get("latent_dim", 8))),
            int(codec_cfg["waveform_samples"]),
            seed=int(codec_cfg.get("seed", 0)),
        ).to(device)
    elif codec_type == "speechtokenizer":
        config_path = Path(codec_cfg["config_path"])
        checkpoint_path = Path(codec_cfg["checkpoint_path"])
        missing = [str(path) for path in (config_path, checkpoint_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "SpeechTokenizer codec-only evaluation requires existing codec files: "
                + ", ".join(missing)
            )
        codec = SpeechTokenizerWrapper(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            waveform_samples=int(codec_cfg["waveform_samples"]),
            n_q=codec_cfg.get("n_q"),
            fallback_to_mock=False,
            freeze=codec_cfg.get("freeze", True),
        ).to(device)
    else:
        raise ValueError(f"unsupported codec type: {codec_type}")
    codec.eval()
    return codec


def write_pcm_wave(path: str | Path, waveform: Tensor, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vector = waveform.detach().cpu().flatten().clamp(-1.0, 1.0)
    pcm = (vector * 32767.0).round().to(torch.int16).numpy().astype("<i2", copy=False)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())


def waveform_snr_db(reference: Tensor, estimate: Tensor, eps: float = 1e-8) -> Tensor:
    noise = reference - estimate
    signal_power = reference.square().mean(dim=-1).clamp_min(eps)
    noise_power = noise.square().mean(dim=-1).clamp_min(eps)
    return 10.0 * torch.log10(signal_power / noise_power)


def _as_batch(waveform: Tensor) -> Tensor:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim == 3 and waveform.shape[1] == 1:
        waveform = waveform.squeeze(1)
    if waveform.ndim != 2:
        raise ValueError(f"waveform must have shape [S], [B,S], or [B,1,S], got {tuple(waveform.shape)}")
    if not torch.isfinite(waveform).all():
        raise ValueError("waveform contains NaN or Inf")
    return waveform.float()


def _overlap_for_lag(reference: Tensor, estimate: Tensor, lag: int) -> tuple[Tensor, Tensor]:
    if lag > 0:
        length = min(reference.shape[-1] - lag, estimate.shape[-1])
        return reference[..., lag : lag + length], estimate[..., :length]
    if lag < 0:
        offset = -lag
        length = min(reference.shape[-1], estimate.shape[-1] - offset)
        return reference[..., :length], estimate[..., offset : offset + length]
    length = min(reference.shape[-1], estimate.shape[-1])
    return reference[..., :length], estimate[..., :length]


def align_for_metrics(
    reference: Tensor,
    estimate: Tensor,
    *,
    metric_align: str = "none",
    max_lag_samples: int = 1000,
) -> tuple[Tensor, Tensor, Tensor]:
    reference = _as_batch(reference)
    estimate = _as_batch(estimate)
    if reference.shape[0] != estimate.shape[0]:
        raise ValueError("reference and estimate batch sizes must match")
    if metric_align not in {"none", "peak_xcorr"}:
        raise ValueError("metric_align must be 'none' or 'peak_xcorr'")
    if metric_align == "none":
        length = min(reference.shape[-1], estimate.shape[-1])
        return reference[..., :length], estimate[..., :length], torch.zeros(reference.shape[0], dtype=torch.long)

    max_lag = min(int(max_lag_samples), reference.shape[-1] - 1, estimate.shape[-1] - 1)
    aligned_refs: list[Tensor] = []
    aligned_ests: list[Tensor] = []
    lags: list[int] = []
    lengths: list[int] = []
    for index in range(reference.shape[0]):
        best_lag = 0
        best_score: float | None = None
        ref_i = reference[index]
        est_i = estimate[index]
        for lag in range(-max_lag, max_lag + 1):
            ref_overlap, est_overlap = _overlap_for_lag(ref_i, est_i, lag)
            if ref_overlap.numel() == 0:
                continue
            score = float((ref_overlap * est_overlap).sum().detach().cpu().item())
            if best_score is None or score > best_score:
                best_score = score
                best_lag = lag
        ref_aligned, est_aligned = _overlap_for_lag(ref_i, est_i, best_lag)
        aligned_refs.append(ref_aligned)
        aligned_ests.append(est_aligned)
        lags.append(best_lag)
        lengths.append(ref_aligned.shape[-1])
    common_length = min(lengths)
    return (
        torch.stack([value[..., :common_length] for value in aligned_refs], dim=0),
        torch.stack([value[..., :common_length] for value in aligned_ests], dim=0),
        torch.tensor(lags, dtype=torch.long),
    )


def _scale_match_estimate(reference: Tensor, estimate: Tensor, eps: float = 1e-8) -> Tensor:
    alpha = (estimate * reference).sum(dim=-1, keepdim=True) / estimate.square().sum(dim=-1, keepdim=True).clamp_min(eps)
    return estimate * alpha


def compute_si_sdr_configurable(
    reference: Tensor,
    estimate: Tensor,
    *,
    zero_mean: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    reference = _as_batch(reference)
    estimate = _as_batch(estimate)
    length = min(reference.shape[-1], estimate.shape[-1])
    reference = reference[..., :length]
    estimate = estimate[..., :length]
    if zero_mean:
        reference = reference - reference.mean(dim=-1, keepdim=True)
        estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    projection = (estimate * reference).sum(dim=-1, keepdim=True) / reference.square().sum(dim=-1, keepdim=True).clamp_min(eps)
    target = projection * reference
    noise = estimate - target
    ratio = target.square().sum(dim=-1).clamp_min(eps) / noise.square().sum(dim=-1).clamp_min(eps)
    return 10.0 * torch.log10(ratio.clamp_min(eps))


def stft_l1(reference: Tensor, estimate: Tensor) -> float:
    if reference.ndim == 1:
        reference = reference.unsqueeze(0)
    if estimate.ndim == 1:
        estimate = estimate.unsqueeze(0)
    length = min(reference.shape[-1], estimate.shape[-1])
    reference = reference[..., :length]
    estimate = estimate[..., :length]
    n_fft = min(512, max(2, length))
    n_fft = 2 ** int(math.floor(math.log2(n_fft))) if n_fft > 2 else n_fft
    hop_length = max(1, n_fft // 4)
    window = torch.hann_window(n_fft, device=reference.device, dtype=reference.dtype)
    if length < n_fft:
        pad = n_fft - length
        reference = torch.nn.functional.pad(reference, (0, pad))
        estimate = torch.nn.functional.pad(estimate, (0, pad))
    ref_spec = torch.stft(
        reference,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
        center=True,
    ).abs()
    est_spec = torch.stft(
        estimate,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
        center=True,
    ).abs()
    return float((ref_spec - est_spec).abs().mean().detach().cpu().item())


def waveform_metrics(
    reference: Tensor,
    estimate: Tensor,
    *,
    metric_align: str = "none",
    max_lag_samples: int = 1000,
    snr_scale_match: bool = False,
    metric_zero_mean: bool = True,
) -> dict[str, float]:
    reference, estimate, lags = align_for_metrics(
        reference,
        estimate,
        metric_align=metric_align,
        max_lag_samples=max_lag_samples,
    )
    snr_estimate = _scale_match_estimate(reference, estimate) if snr_scale_match else estimate
    return {
        "waveform_mse": float((reference - estimate).square().mean().detach().cpu().item()),
        "waveform_l1": float((reference - estimate).abs().mean().detach().cpu().item()),
        "waveform_snr_db": float(waveform_snr_db(reference, snr_estimate).mean().detach().cpu().item()),
        "si_sdr_db": float(
            compute_si_sdr_configurable(reference, estimate, zero_mean=metric_zero_mean)
            .mean()
            .detach()
            .cpu()
            .item()
        ),
        "stft_l1": stft_l1(reference, estimate),
        "best_lag_samples": float(lags.float().mean().detach().cpu().item()),
    }


def signal_activity_features(waveform: Tensor, *, threshold: float) -> dict[str, float]:
    waveform = _as_batch(waveform)
    active = waveform.abs() > threshold
    ratios = []
    leading = []
    trailing = []
    for item in active:
        item_cpu = item.detach().cpu()
        active_indices = torch.nonzero(item_cpu, as_tuple=False).flatten()
        ratios.append(float(item_cpu.float().mean().item()))
        if active_indices.numel() == 0:
            leading.append(1.0)
            trailing.append(1.0)
        else:
            leading.append(float(active_indices[0].item() / item_cpu.numel()))
            trailing.append(float((item_cpu.numel() - 1 - active_indices[-1].item()) / item_cpu.numel()))
    return {
        "speech_activity_ratio": float(sum(ratios) / len(ratios)),
        "leading_silence_ratio": float(sum(leading) / len(leading)),
        "trailing_silence_ratio": float(sum(trailing) / len(trailing)),
    }


def crop_metadata(record: dict[str, Any], waveform_samples: int) -> dict[str, int]:
    original_samples = int(record.get("num_samples") or waveform_samples)
    if original_samples > waveform_samples:
        crop_start = (original_samples - waveform_samples) // 2
        crop_end = crop_start + waveform_samples
    else:
        crop_start = 0
        crop_end = original_samples
    return {
        "duration_before_crop": original_samples,
        "crop_start_sample": crop_start,
        "crop_end_sample": crop_end,
    }


def export_outlier_artifacts(
    *,
    output_dir: str | Path,
    utt_id: str,
    original: Tensor,
    official_recon: Tensor | None,
    continuous_sum_recon: Tensor,
    sample_rate: int,
    max_lag_samples: int = 1000,
) -> None:
    out_dir = Path(output_dir) / "outliers" / utt_id
    out_dir.mkdir(parents=True, exist_ok=True)
    recon = official_recon if official_recon is not None else continuous_sum_recon
    aligned_original, aligned_recon, _ = align_for_metrics(
        original,
        recon,
        metric_align="peak_xcorr",
        max_lag_samples=max_lag_samples,
    )
    error = aligned_original - aligned_recon
    write_pcm_wave(out_dir / "original.wav", original, sample_rate)
    if official_recon is not None:
        write_pcm_wave(out_dir / "official_recon.wav", official_recon, sample_rate)
    write_pcm_wave(out_dir / "continuous_sum_recon.wav", continuous_sum_recon, sample_rate)
    write_pcm_wave(out_dir / "aligned_original.wav", aligned_original[0], sample_rate)
    write_pcm_wave(out_dir / "aligned_recon.wav", aligned_recon[0], sample_rate)
    write_pcm_wave(out_dir / "error_waveform.wav", error[0], sample_rate)
    _save_outlier_plots(
        out_dir=out_dir,
        original=aligned_original[0],
        recon=aligned_recon[0],
        error=error[0],
        sample_rate=sample_rate,
    )


def _save_outlier_plots(
    *,
    out_dir: Path,
    original: Tensor,
    recon: Tensor,
    error: Tensor,
    sample_rate: int,
) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".matplotlib"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        time_axis = torch.arange(original.numel()).float().numpy() / float(sample_rate)
        fig, axis = plt.subplots(figsize=(8, 3))
        axis.plot(time_axis, original.detach().cpu().numpy(), label="original", linewidth=1.0)
        axis.plot(time_axis, recon.detach().cpu().numpy(), label="recon", linewidth=1.0, alpha=0.8)
        axis.set(xlabel="Time (s)", ylabel="Amplitude", title="Waveform overlay")
        axis.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / "waveform_overlay.png", dpi=150)
        plt.close(fig)

        for tensor, filename, title in (
            (original, "spectrogram_original.png", "Original spectrogram"),
            (recon, "spectrogram_recon.png", "Reconstruction spectrogram"),
            (error, "spectrogram_error.png", "Error spectrogram"),
        ):
            fig, axis = plt.subplots(figsize=(8, 3))
            axis.specgram(tensor.detach().cpu().numpy(), Fs=sample_rate, NFFT=256, noverlap=128)
            axis.set(xlabel="Time (s)", ylabel="Frequency (Hz)", title=title)
            fig.tight_layout()
            fig.savefig(out_dir / filename, dpi=150)
            plt.close(fig)
    except Exception as error_obj:
        print(f"warning: outlier plot generation failed for {out_dir}: {error_obj}")


def _json_list(values: Iterable[float | int]) -> str:
    return json.dumps(list(values), separators=(",", ":"))


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return float(math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)))


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _distribution(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": _mean(values),
        "std": _std(values),
        "median": _percentile(values, 0.5),
        "p10": _percentile(values, 0.1),
        "p90": _percentile(values, 0.9),
        "min": float(min(values)) if values else None,
        "max": float(max(values)) if values else None,
    }


def _save_plots(
    clean_rows: list[dict[str, Any]],
    noise_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(plot_dir / ".matplotlib"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for key, filename, title in (
            ("waveform_snr_db", "waveform_snr_hist.png", "Clean codec waveform SNR"),
            ("si_sdr_db", "si_sdr_hist.png", "Clean codec SI-SDR"),
        ):
            values = [float(row[key]) for row in clean_rows]
            fig, axis = plt.subplots(figsize=(6, 4))
            axis.hist(values, bins=min(20, max(3, len(values))))
            axis.set(xlabel=key, ylabel="Utterances", title=title)
            fig.tight_layout()
            fig.savefig(plot_dir / filename, dpi=150)
            plt.close(fig)

        if clean_rows:
            layer_values = [json.loads(str(row["per_layer_energy"])) for row in clean_rows]
            layer_count = len(layer_values[0])
            means = [
                sum(values[layer] for values in layer_values) / len(layer_values)
                for layer in range(layer_count)
            ]
            fig, axis = plt.subplots(figsize=(6, 4))
            axis.bar(range(1, layer_count + 1), means)
            axis.set(xlabel="Latent layer", ylabel="Energy", title="Mean per-layer latent energy")
            fig.tight_layout()
            fig.savefig(plot_dir / "per_layer_energy.png", dpi=150)
            plt.close(fig)

        if noise_rows:
            by_snr: dict[float, list[dict[str, Any]]] = {}
            for row in noise_rows:
                by_snr.setdefault(float(row["latent_noise_snr_db"]), []).append(row)
            snrs = sorted(by_snr, reverse=True)
            for key, filename, ylabel in (
                ("si_sdr_db", "latent_noise_snr_vs_si_sdr.png", "SI-SDR (dB)"),
                ("waveform_snr_db", "latent_noise_snr_vs_waveform_snr.png", "Waveform SNR (dB)"),
                ("stft_l1", "latent_noise_snr_vs_stft_l1.png", "STFT L1"),
            ):
                values = [
                    sum(float(row[key]) for row in by_snr[snr]) / len(by_snr[snr])
                    for snr in snrs
                ]
                fig, axis = plt.subplots(figsize=(6, 4))
                axis.plot(snrs, values, marker="o")
                axis.invert_xaxis()
                axis.set(xlabel="Latent noise SNR (dB)", ylabel=ylabel, title=ylabel)
                axis.grid(alpha=0.25)
                fig.tight_layout()
                fig.savefig(plot_dir / filename, dpi=150)
                plt.close(fig)
    except Exception as error:
        print(f"warning: plot generation failed: {error}")


def _record_metadata(record: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        "utt_id": record.get("utt_id", Path(str(record["audio_path"])).stem),
        "speaker_id": record.get("speaker_id", ""),
        "chapter_id": record.get("chapter_id", ""),
        "split": record.get("split", split),
        "audio_path": record["audio_path"],
        "text": record.get("text", ""),
        "duration_sec": record.get("duration_sec"),
        "sample_rate": record.get("sample_rate"),
        "num_samples": record.get("num_samples"),
    }


def _summary(
    *,
    clean_rows: list[dict[str, Any]],
    noise_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    worst_rows: list[dict[str, Any]],
    manifest_path: Path,
    split: str,
    codec_type: str,
    representation_shape: tuple[int, int, int],
    sample_rate: int,
    waveform_samples: int,
    latent_noise_sweep_used: bool,
    decode_mode: str,
    metric_align: str,
    max_lag_samples: int,
    snr_scale_match: bool,
    metric_zero_mean: bool,
) -> dict[str, Any]:
    metric_values = {
        key: [float(row[key]) for row in clean_rows]
        for key in CLEAN_METRIC_KEYS
    }
    runtime_values = {
        key: [float(row[key]) for row in clean_rows]
        for key in ("encode_time_sec", "decode_time_sec", "real_time_factor")
    }
    sweep: dict[str, Any] = {}
    if noise_rows:
        snrs = sorted({float(row["latent_noise_snr_db"]) for row in noise_rows}, reverse=True)
        for snr in snrs:
            selected = [row for row in noise_rows if float(row["latent_noise_snr_db"]) == snr]
            sweep[str(int(snr) if snr.is_integer() else snr)] = {
                "mean_waveform_snr_db": _mean([float(row["waveform_snr_db"]) for row in selected]),
                "mean_si_sdr_db": _mean([float(row["si_sdr_db"]) for row in selected]),
                "mean_stft_l1": _mean([float(row["stft_l1"]) for row in selected]),
                "mean_delta_waveform_mse_vs_clean": _mean(
                    [float(row["delta_waveform_mse_vs_clean"]) for row in selected]
                ),
                "mean_delta_waveform_snr_db_vs_clean": _mean(
                    [float(row["delta_waveform_snr_db_vs_clean"]) for row in selected]
                ),
                "mean_delta_si_sdr_db_vs_clean": _mean(
                    [float(row["delta_si_sdr_db_vs_clean"]) for row in selected]
                ),
                "mean_delta_stft_l1_vs_clean": _mean(
                    [float(row["delta_stft_l1_vs_clean"]) for row in selected]
                ),
            }
    official_values = {
        key: [float(row[f"official_{key}"]) for row in clean_rows if f"official_{key}" in row]
        for key in CLEAN_METRIC_KEYS
    }
    continuous_values = {
        key: [float(row[f"continuous_sum_{key}"]) for row in clean_rows if f"continuous_sum_{key}" in row]
        for key in CLEAN_METRIC_KEYS
    }
    gap_values = {
        key: [
            float(row[f"delta_official_minus_continuous_sum_{key}"])
            for row in clean_rows
            if f"delta_official_minus_continuous_sum_{key}" in row
        ]
        for key in CLEAN_METRIC_KEYS
    }
    alignment_gap_values = {
        key: [
            float(row[f"delta_metric_align_vs_none_{key}"])
            for row in clean_rows
            if f"delta_metric_align_vs_none_{key}" in row
        ]
        for key in CLEAN_METRIC_KEYS
    }
    return {
        "mode": "codec_only_with_latent_noise" if latent_noise_sweep_used else "codec_only",
        "wireless_channel_used": False,
        "jscc_used": False,
        "latent_noise_sweep_used": latent_noise_sweep_used,
        "decode_mode": decode_mode,
        "manifest": str(manifest_path),
        "split": split,
        "num_items_evaluated": len(clean_rows),
        "codec_type": codec_type,
        "representation_shape": list(representation_shape),
        "sample_rate": sample_rate,
        "waveform_samples": waveform_samples,
        "clean_codec_metrics_mean": {key: _mean(values) for key, values in metric_values.items()},
        "clean_codec_metrics_std": {key: _std(values) for key, values in metric_values.items()},
        "clean_codec_metrics_distribution": {key: _distribution(values) for key, values in metric_values.items()},
        "official_metrics_mean": {key: _mean(values) for key, values in official_values.items()},
        "official_metrics_distribution": {key: _distribution(values) for key, values in official_values.items()},
        "continuous_sum_metrics_mean": {key: _mean(values) for key, values in continuous_values.items()},
        "continuous_sum_metrics_distribution": {key: _distribution(values) for key, values in continuous_values.items()},
        "official_vs_continuous_sum_metric_gap": {
            key: _distribution(values) for key, values in gap_values.items()
        },
        "alignment_off_vs_on_metric_gap": {
            key: _distribution(values) for key, values in alignment_gap_values.items()
        },
        "alignment": {
            "metric_align": metric_align,
            "max_lag_samples": max_lag_samples,
            "snr_scale_match": snr_scale_match,
            "metric_zero_mean": metric_zero_mean,
        },
        "latent_noise_sweep": sweep,
        "runtime": {
            "mean_encode_time_sec": _mean(runtime_values["encode_time_sec"]),
            "mean_decode_time_sec": _mean(runtime_values["decode_time_sec"]),
            "mean_real_time_factor": _mean(runtime_values["real_time_factor"]),
        },
        "mean_per_layer_energy": _mean_per_layer_energy(clean_rows),
        "worst_samples": worst_rows,
        "si_sdr_outliers_below_minus_10db": [
            {
                "utt_id": row["utt_id"],
                "audio_path": row["audio_path"],
                "si_sdr_db": float(row["clean_si_sdr"]),
                "used_decode_mode": row["used_decode_mode"],
            }
            for row in clean_rows
            if float(row["clean_si_sdr"]) <= -10.0
        ],
        "decode_comparison_rows": len(comparison_rows),
    }


def _mean_per_layer_energy(clean_rows: list[dict[str, Any]]) -> list[float]:
    if not clean_rows:
        return []
    values = [json.loads(str(row["per_layer_energy"])) for row in clean_rows]
    return [
        float(sum(item[layer] for item in values) / len(values))
        for layer in range(len(values[0]))
    ]


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _metric_gaps(prefix: str, left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": left[key] - right[key] for key in CLEAN_METRIC_KEYS}


def _write_report_markdown(path: Path, summary: dict[str, Any]) -> None:
    def table(title: str, values: dict[str, Any]) -> list[str]:
        lines = [f"## {title}", "", "| metric | mean | median | p10 | p90 | min | max |", "|---|---:|---:|---:|---:|---:|---:|"]
        for key, dist in values.items():
            if not isinstance(dist, dict):
                continue
            lines.append(
                "| {key} | {mean} | {median} | {p10} | {p90} | {minv} | {maxv} |".format(
                    key=key,
                    mean=_format_optional(dist.get("mean")),
                    median=_format_optional(dist.get("median")),
                    p10=_format_optional(dist.get("p10")),
                    p90=_format_optional(dist.get("p90")),
                    minv=_format_optional(dist.get("min")),
                    maxv=_format_optional(dist.get("max")),
                )
            )
        lines.append("")
        return lines

    lines = [
        "# Codec-only SpeechTokenizer Baseline Diagnostics",
        "",
        f"- manifest: `{summary['manifest']}`",
        f"- split: `{summary['split']}`",
        f"- decode_mode: `{summary['decode_mode']}`",
        f"- metric_align: `{summary['alignment']['metric_align']}`",
        "",
    ]
    lines.extend(table("Clean official reconstruction summary", summary["official_metrics_distribution"]))
    lines.extend(table("Clean continuous_sum reconstruction summary", summary["continuous_sum_metrics_distribution"]))
    lines.extend(table("Official vs continuous_sum metric gap", summary["official_vs_continuous_sum_metric_gap"]))
    lines.extend(table("Alignment off vs alignment on metric gap", summary["alignment_off_vs_on_metric_gap"]))
    lines.append("## Worst sample list")
    lines.append("")
    lines.append("| utt_id | used_decode_mode | si_sdr_db | waveform_snr_db | stft_l1 | audio_path |")
    lines.append("|---|---|---:|---:|---:|---|")
    for row in summary.get("worst_samples", []):
        lines.append(
            f"| {row['utt_id']} | {row['used_decode_mode']} | "
            f"{_format_optional(row['clean_si_sdr'])} | {_format_optional(row['clean_waveform_snr'])} | "
            f"{_format_optional(row['clean_stft_l1'])} | `{row['audio_path']}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _protocol_label(setting_id: str, metric_align: str, snr_scale_match: bool) -> str:
    return f"{setting_id}: align={metric_align}, scale_match={str(snr_scale_match).lower()}"


def _build_baseline_protocol_table(protocol_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for setting_id, metric_align, snr_scale_match in BASELINE_PROTOCOLS:
        selected = [
            row for row in protocol_rows
            if row["setting_id"] == setting_id
            and row["metric_align"] == metric_align
            and bool(row["snr_scale_match"]) is bool(snr_scale_match)
        ]
        outlier_count = sum(float(row["si_sdr_db"]) <= -10.0 for row in selected)
        waveform_samples = int(selected[0].get("waveform_samples", 0)) if selected else 0
        sample_rate = int(selected[0].get("sample_rate", 16000)) if selected else 16000
        duration_sec = float(waveform_samples / sample_rate) if sample_rate else None
        metric_zero_mean = bool(selected[0].get("metric_zero_mean", True)) if selected else True
        for metric in BASELINE_TABLE_METRICS:
            values = [float(row[metric]) for row in selected]
            dist = _distribution(values)
            table.append(
                {
                    "waveform_samples": waveform_samples,
                    "duration_sec": duration_sec,
                    "protocol_name": setting_id,
                    "setting_id": setting_id,
                    "setting": _protocol_label(setting_id, metric_align, snr_scale_match),
                    "metric_align": metric_align,
                    "snr_scale_match": bool(snr_scale_match),
                    "metric_zero_mean": metric_zero_mean,
                    "metric": metric,
                    "mean": dist["mean"],
                    "median": dist["median"],
                    "std": dist["std"],
                    "p10": dist["p10"],
                    "p90": dist["p90"],
                    "min": dist["min"],
                    "max": dist["max"],
                    "n": len(selected),
                    "outliers": outlier_count,
                    "num_samples": len(selected),
                    "si_sdr_le_minus_10_count": outlier_count,
                }
            )
    return table


def _diagnostic_excluded_metrics(
    clean_rows: list[dict[str, Any]],
    *,
    lag_outlier_threshold: int,
) -> dict[str, dict[str, dict[str, float | None] | int]]:
    cohorts = {
        "all_samples": clean_rows,
        "excluding_si_sdr_le_minus_10db": [
            row for row in clean_rows if float(row["clean_si_sdr"]) > -10.0
        ],
        "excluding_silence_like": [
            row for row in clean_rows if str(row["silence_like"]).lower() != "true"
        ],
        "excluding_abs_best_lag_gt_threshold": [
            row for row in clean_rows if abs(float(row["best_lag_samples"])) <= lag_outlier_threshold
        ],
    }
    metrics = {
        "waveform_snr_db": "clean_waveform_snr",
        "si_sdr_db": "clean_si_sdr",
        "stft_l1": "clean_stft_l1",
    }
    result: dict[str, dict[str, dict[str, float | None] | int]] = {}
    for name, rows in cohorts.items():
        result[name] = {"num_samples": len(rows)}
        for metric, column in metrics.items():
            result[name][metric] = _distribution([float(row[column]) for row in rows])
    return result


def _write_baseline_protocol_reports(output_dir: Path, table_rows: list[dict[str, Any]]) -> None:
    _write_csv(output_dir / "baseline_protocol_comparison.csv", table_rows)
    md_lines = [
        "# Baseline Metric Protocol Comparison",
        "",
        "Recommended protocol for main reporting:",
        "",
        "- `metric_align=peak_xcorr`",
        "- `snr_scale_match=true`",
        "- `metric_zero_mean=true`",
        "- report mean and median together",
        "- always report SI-SDR <= -10 dB outlier count",
        "",
        "`metric_align=none` results are retained for appendix-style comparison.",
        "",
        "| setting | metric | mean | median | std | p10 | p90 | min | max | n | SI-SDR<=-10 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        md_lines.append(
            "| {setting} | {metric} | {mean} | {median} | {std} | {p10} | {p90} | {minv} | {maxv} | {n} | {outliers} |".format(
                setting=row["setting"],
                metric=row["metric"],
                mean=_format_optional(row["mean"]),
                median=_format_optional(row["median"]),
                std=_format_optional(row["std"]),
                p10=_format_optional(row["p10"]),
                p90=_format_optional(row["p90"]),
                minv=_format_optional(row["min"]),
                maxv=_format_optional(row["max"]),
                n=row["num_samples"],
                outliers=row["si_sdr_le_minus_10_count"],
            )
        )
    md_lines.append("")
    (output_dir / "baseline_protocol_comparison.md").write_text("\n".join(md_lines), encoding="utf-8")
    _write_baseline_protocol_pdf(output_dir / "baseline_protocol_comparison.pdf", table_rows)


def _write_final_baseline_report(output_dir: Path, summary: dict[str, Any], table_rows: list[dict[str, Any]]) -> None:
    def row(protocol_name: str, metric: str) -> dict[str, Any] | None:
        for item in table_rows:
            if item.get("protocol_name") == protocol_name and item.get("metric") == metric:
                return item
        return None

    main_snr = row("D", "waveform_snr_db")
    main_sisdr = row("D", "si_sdr_db")
    main_stft = row("D", "stft_l1")
    samples = int(summary.get("waveform_samples", 0))
    sample_rate = int(summary.get("sample_rate", 16000))
    duration = samples / sample_rate if sample_rate else 0.0
    outliers = int(main_sisdr.get("outliers", 0)) if main_sisdr else len(summary.get("si_sdr_outliers_below_minus_10db", []))
    total = int(main_sisdr.get("n", summary.get("num_items_evaluated", 0))) if main_sisdr else int(summary.get("num_items_evaluated", 0))
    lines = [
        "# SpeechTokenizer Codec-only Baseline Protocol",
        "",
        "## Recommended protocol",
        "",
        "- waveform_samples = 32000",
        "- duration = 2 seconds at 16 kHz",
        "- metric_align = peak_xcorr",
        "- snr_scale_match = true",
        "- metric_zero_mean = true",
        "- main result keeps all samples and always reports SI-SDR <= -10 dB outlier count",
        "",
        "## Main clean codec-only result",
        "",
        f"- evaluated waveform_samples = {samples}",
        f"- evaluated duration = {duration:.3g} seconds at {sample_rate} Hz",
        f"- waveform SNR mean = {_format_optional(main_snr.get('mean') if main_snr else summary['clean_codec_metrics_mean']['waveform_snr_db'])} dB",
        f"- SI-SDR mean = {_format_optional(main_sisdr.get('mean') if main_sisdr else summary['clean_codec_metrics_mean']['si_sdr_db'])} dB",
        f"- STFT L1 mean = {_format_optional(main_stft.get('mean') if main_stft else summary['clean_codec_metrics_mean']['stft_l1'])}",
        f"- outlier count = {outliers} / {total}",
        "",
        "## Reference 1-second protocol",
        "",
        "- waveform_samples = 16000",
        "- D protocol result from the fixed comparison run:",
        "  - waveform SNR mean = 5.30472 dB",
        "  - SI-SDR mean = 3.56680 dB",
        "  - STFT L1 mean = 0.0583944",
        "  - outlier count = 1 / 100",
        "",
        "## Interpretation",
        "",
        "- official SpeechTokenizer reconstruction and continuous_sum reconstruction are equivalent in clean codec-only evaluation.",
        "- continuous_sum is not pre-quantization latent; it is post-quantization RVQ codebook embedding summed across layers.",
        "- The advantage of continuous_sum for JSCC is not higher clean SDR, but graceful degradation under channel/latent perturbation.",
        "- 2-second crop is selected as the stable main codec-only baseline because it removes the 1-second outlier and gives the best SI-SDR among tested stable settings.",
        "- `latent_noise_snr_db` is latent-domain perturbation SNR, not wireless SNR.",
        "",
        "## Outlier policy",
        "",
        "- outlier threshold: SI-SDR <= -10 dB",
        "- main result does not remove outliers",
        "- 1-second condition keeps one known outlier: `1188-133604-0012` at `data/mini_librispeech/LibriSpeech/test-clean/1188/133604/1188-133604-0012.flac`",
        "- 2-second condition has 0 outliers in the current test_100 protocol comparison",
        "",
    ]
    (output_dir / "codec_only_baseline_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_baseline_protocol_pdf(path: Path, table_rows: list[dict[str, Any]]) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(path.parent / "plots" / ".matplotlib"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(path) as pdf:
            for setting_id, _, _ in BASELINE_PROTOCOLS:
                selected = [row for row in table_rows if row["setting_id"] == setting_id]
                fig, axis = plt.subplots(figsize=(11, 6))
                axis.axis("off")
                axis.set_title(selected[0]["setting"] if selected else setting_id, loc="left")
                cells = [
                    [
                        row["metric"],
                        _format_optional(row["mean"]),
                        _format_optional(row["median"]),
                        _format_optional(row["std"]),
                        _format_optional(row["p10"]),
                        _format_optional(row["p90"]),
                        _format_optional(row["min"]),
                        _format_optional(row["max"]),
                        str(row["num_samples"]),
                        str(row["si_sdr_le_minus_10_count"]),
                    ]
                    for row in selected
                ]
                table = axis.table(
                    cellText=cells,
                    colLabels=["metric", "mean", "median", "std", "p10", "p90", "min", "max", "n", "outliers"],
                    loc="center",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1.0, 1.4)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
    except Exception as error:
        print(f"warning: baseline protocol PDF generation failed: {error}")


def run_waveform_length_sweep(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for samples in args.waveform_samples:
        child_dir = output_dir / f"waveform_samples_{samples}"
        summary = run_evaluation(
            config_path=args.config,
            manifest_path=args.manifest,
            split=args.split,
            output_dir=child_dir,
            max_items=args.max_items,
            batch_size=args.batch_size,
            enable_latent_noise_sweep=args.enable_latent_noise_sweep,
            latent_noise_snr_db=args.latent_noise_snr_db,
            num_save_examples=args.num_save_examples,
            device_name=args.device,
            seed=args.seed,
            decode_mode=args.decode_mode,
            metric_align=args.metric_align,
            max_lag_samples=args.max_lag_samples,
            snr_scale_match=args.snr_scale_match,
            metric_zero_mean=args.metric_zero_mean,
            worst_k=args.worst_k,
            silence_rms_threshold=args.silence_rms_threshold,
            protocol_comparison=True,
            lag_outlier_threshold=args.lag_outlier_threshold,
            waveform_samples_override=samples,
            compare_official=getattr(args, "compare_official", True),
        )
        summaries[str(samples)] = {
            "summary_path": str(child_dir / "summary.json"),
            "clean_codec_metrics_mean": summary["clean_codec_metrics_mean"],
            "official_metrics_mean": summary["official_metrics_mean"],
            "continuous_sum_metrics_mean": summary["continuous_sum_metrics_mean"],
            "baseline_protocol_comparison": summary["baseline_protocol_comparison"],
        }
        for row in summary["baseline_protocol_comparison"]:
            row = dict(row)
            row["waveform_samples"] = samples
            combined_rows.append(row)
    _write_csv(output_dir / "waveform_length_protocol_comparison.csv", combined_rows)
    _write_length_sweep_markdown(output_dir / "waveform_length_protocol_comparison.md", combined_rows)
    _write_baseline_protocol_pdf(output_dir / "waveform_length_protocol_comparison.pdf", combined_rows)
    result = {
        "mode": "codec_only_waveform_length_sweep",
        "waveform_samples": args.waveform_samples,
        "recommended_baseline_protocol": {
            "metric_align": "peak_xcorr",
            "snr_scale_match": True,
            "metric_zero_mean": True,
            "report": ["mean", "median", "si_sdr_le_minus_10_count"],
            "main_result_uses_all_samples": True,
        },
        "lengths": summaries,
    }
    (output_dir / "waveform_length_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def _write_length_sweep_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Waveform Length Baseline Protocol Comparison",
        "",
        "Recommended main protocol: `metric_align=peak_xcorr`, `snr_scale_match=true`, `metric_zero_mean=true`.",
        "Report mean and median together and always include SI-SDR <= -10 dB outlier count.",
        "",
        "| waveform_samples | setting | metric | mean | median | std | p10 | p90 | min | max | n | SI-SDR<=-10 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {samples} | {setting} | {metric} | {mean} | {median} | {std} | {p10} | {p90} | {minv} | {maxv} | {n} | {outliers} |".format(
                samples=row.get("waveform_samples", ""),
                setting=row["setting"],
                metric=row["metric"],
                mean=_format_optional(row["mean"]),
                median=_format_optional(row["median"]),
                std=_format_optional(row["std"]),
                p10=_format_optional(row["p10"]),
                p90=_format_optional(row["p90"]),
                minv=_format_optional(row["min"]),
                maxv=_format_optional(row["max"]),
                n=row["num_samples"],
                outliers=row["si_sdr_le_minus_10_count"],
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_optional(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def run_evaluation(
    *,
    config_path: str | Path,
    manifest_path: str | Path,
    split: str,
    output_dir: str | Path,
    max_items: int | None,
    batch_size: int,
    enable_latent_noise_sweep: bool,
    latent_noise_snr_db: list[float],
    num_save_examples: int,
    device_name: str | None = None,
    seed: int | None = None,
    decode_mode: str = "continuous_sum",
    metric_align: str = "none",
    max_lag_samples: int = 1000,
    snr_scale_match: bool = False,
    metric_zero_mean: bool = True,
    worst_k: int = 10,
    silence_rms_threshold: float = 1e-4,
    protocol_comparison: bool = False,
    lag_outlier_threshold: int | None = None,
    waveform_samples_override: int | None = None,
    compare_official: bool = True,
) -> dict[str, Any]:
    config_path = Path(config_path)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if waveform_samples_override is not None:
        config = copy.deepcopy(config)
        config.setdefault("codec", {})["waveform_samples"] = int(waveform_samples_override)
    if seed is None:
        seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    device = resolve_device(device_name or config.get("device", "auto"))
    codec = build_codec_only(config, device)
    codec_type = config["codec"].get("type", "mock").lower()
    sample_rate = int(getattr(codec, "sample_rate", config["codec"].get("sample_rate", 16000)))
    waveform_samples = int(config["codec"]["waveform_samples"])
    if lag_outlier_threshold is None:
        lag_outlier_threshold = max_lag_samples
    records = read_manifest(manifest_path, max_items=max_items)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "examples").mkdir(parents=True, exist_ok=True)
    (output_dir / "worst_samples").mkdir(parents=True, exist_ok=True)
    clean_rows: list[dict[str, Any]] = []
    noise_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    protocol_rows: list[dict[str, Any]] = []
    recon_cache: dict[str, dict[str, Tensor]] = {}
    generator = torch.Generator(device=device).manual_seed(seed + 91)
    if decode_mode not in {"official", "continuous_sum", "both"}:
        raise ValueError("decode_mode must be 'official', 'continuous_sum', or 'both'")

    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            waveforms = torch.stack(
                [
                    load_waveform_segment(
                        record["audio_path"],
                        sample_rate=sample_rate,
                        waveform_samples=waveform_samples,
                    )
                    for record in batch_records
                ],
                dim=0,
            ).to(device)
            batch_started = time.perf_counter()
            encode_started = time.perf_counter()
            latents = codec.encode_waveform(waveforms)
            encode_time = time.perf_counter() - encode_started
            decode_started = time.perf_counter()
            continuous_recon = codec.decode_representation(latents)
            continuous_decode_time = time.perf_counter() - decode_started
            official_recon = None
            official_decode_time = 0.0
            if decode_mode in {"official", "both"} or compare_official:
                official_started = time.perf_counter()
                if hasattr(codec, "official_reconstruct_waveform"):
                    official_recon = codec.official_reconstruct_waveform(waveforms)
                else:
                    official_recon = continuous_recon
                official_decode_time = time.perf_counter() - official_started
            clean_recon = official_recon if decode_mode == "official" and official_recon is not None else continuous_recon
            decode_time = official_decode_time if decode_mode == "official" else continuous_decode_time
            batch_total_time = time.perf_counter() - batch_started
            per_item_encode = encode_time / len(batch_records)
            per_item_decode = decode_time / len(batch_records)
            per_item_total = batch_total_time / len(batch_records)

            per_item_metrics = [
                waveform_metrics(
                    waveforms[index : index + 1],
                    clean_recon[index : index + 1],
                    metric_align=metric_align,
                    max_lag_samples=max_lag_samples,
                    snr_scale_match=snr_scale_match,
                    metric_zero_mean=metric_zero_mean,
                )
                for index in range(len(batch_records))
            ]
            per_item_metrics_unaligned = [
                waveform_metrics(
                    waveforms[index : index + 1],
                    clean_recon[index : index + 1],
                    metric_align="none",
                    max_lag_samples=max_lag_samples,
                    snr_scale_match=snr_scale_match,
                    metric_zero_mean=metric_zero_mean,
                )
                for index in range(len(batch_records))
            ]
            continuous_metrics = [
                waveform_metrics(
                    waveforms[index : index + 1],
                    continuous_recon[index : index + 1],
                    metric_align=metric_align,
                    max_lag_samples=max_lag_samples,
                    snr_scale_match=snr_scale_match,
                    metric_zero_mean=metric_zero_mean,
                )
                for index in range(len(batch_records))
            ]
            if official_recon is not None:
                official_metrics = [
                    waveform_metrics(
                        waveforms[index : index + 1],
                        official_recon[index : index + 1],
                        metric_align=metric_align,
                        max_lag_samples=max_lag_samples,
                        snr_scale_match=snr_scale_match,
                        metric_zero_mean=metric_zero_mean,
                    )
                    for index in range(len(batch_records))
                ]
            else:
                official_metrics = continuous_metrics

            latent_energy = latents.square().mean(dim=(1, 2, 3))
            per_layer_energy = latents.square().mean(dim=(2, 3))
            for index, record in enumerate(batch_records):
                metadata = _record_metadata(record, split)
                input_rms = float(waveforms[index].square().mean().sqrt().detach().cpu().item())
                input_peak = float(waveforms[index].abs().max().detach().cpu().item())
                reconstructed_rms = float(clean_recon[index].square().mean().sqrt().detach().cpu().item())
                activity = signal_activity_features(waveforms[index : index + 1], threshold=silence_rms_threshold)
                crop_info = crop_metadata(record, waveform_samples)
                clean_metrics = per_item_metrics[index]
                unaligned_metrics = per_item_metrics_unaligned[index]
                aligned_metrics_no_scale = waveform_metrics(
                    waveforms[index : index + 1],
                    clean_recon[index : index + 1],
                    metric_align="peak_xcorr",
                    max_lag_samples=max_lag_samples,
                    snr_scale_match=False,
                    metric_zero_mean=metric_zero_mean,
                )
                aligned_metrics_scale = waveform_metrics(
                    waveforms[index : index + 1],
                    clean_recon[index : index + 1],
                    metric_align="peak_xcorr",
                    max_lag_samples=max_lag_samples,
                    snr_scale_match=True,
                    metric_zero_mean=metric_zero_mean,
                )
                no_align_no_scale = waveform_metrics(
                    waveforms[index : index + 1],
                    clean_recon[index : index + 1],
                    metric_align="none",
                    max_lag_samples=max_lag_samples,
                    snr_scale_match=False,
                    metric_zero_mean=metric_zero_mean,
                )
                official_item_metrics = official_metrics[index]
                continuous_item_metrics = continuous_metrics[index]
                delta_official = _metric_gaps(
                    "delta_official_minus_continuous_sum",
                    official_item_metrics,
                    continuous_item_metrics,
                )
                delta_alignment = {
                    f"delta_metric_align_vs_none_{key}": clean_metrics[key] - unaligned_metrics[key]
                    for key in CLEAN_METRIC_KEYS
                }
                row = {
                    **metadata,
                    **crop_info,
                    "duration_samples": int(waveforms[index].shape[-1]),
                    "input_rms": input_rms,
                    "input_peak": input_peak,
                    "reconstructed_rms": reconstructed_rms,
                    "output_rms": reconstructed_rms,
                    **activity,
                    "silence_like": bool(input_rms < silence_rms_threshold),
                    "extreme_outlier": bool(clean_metrics["si_sdr_db"] <= -10.0),
                    "used_decode_mode": decode_mode,
                    "latent_shape": _json_list(latents[index].shape),
                    "latent_mean": float(latents[index].mean().detach().cpu().item()),
                    "latent_std": float(latents[index].std(unbiased=False).detach().cpu().item()),
                    "latent_abs_mean": float(latents[index].abs().mean().detach().cpu().item()),
                    "latent_energy": float(latent_energy[index].detach().cpu().item()),
                    "per_layer_energy": _json_list(per_layer_energy[index].detach().cpu().tolist()),
                    **clean_metrics,
                    "clean_waveform_snr": clean_metrics["waveform_snr_db"],
                    "clean_si_sdr": clean_metrics["si_sdr_db"],
                    "clean_stft_l1": clean_metrics["stft_l1"],
                    "si_sdr_no_align": no_align_no_scale["si_sdr_db"],
                    "si_sdr_aligned": aligned_metrics_scale["si_sdr_db"],
                    "si_sdr_gain_from_alignment": aligned_metrics_scale["si_sdr_db"] - no_align_no_scale["si_sdr_db"],
                    "waveform_snr_no_align": no_align_no_scale["waveform_snr_db"],
                    "waveform_snr_aligned": aligned_metrics_scale["waveform_snr_db"],
                    **_prefix_metrics("official", official_item_metrics),
                    **_prefix_metrics("continuous_sum", continuous_item_metrics),
                    **delta_official,
                    **delta_alignment,
                    "encode_time_sec": per_item_encode,
                    "decode_time_sec": per_item_decode,
                    "continuous_sum_decode_time_sec": continuous_decode_time / len(batch_records),
                    "official_decode_time_sec": official_decode_time / len(batch_records),
                    "total_time_sec": per_item_total,
                    "real_time_factor": per_item_total / (waveform_samples / sample_rate),
                }
                clean_rows.append(row)
                if protocol_comparison:
                    protocol_metrics_by_setting = {
                        "A": no_align_no_scale,
                        "B": waveform_metrics(
                            waveforms[index : index + 1],
                            clean_recon[index : index + 1],
                            metric_align="none",
                            max_lag_samples=max_lag_samples,
                            snr_scale_match=True,
                            metric_zero_mean=metric_zero_mean,
                        ),
                        "C": aligned_metrics_no_scale,
                        "D": aligned_metrics_scale,
                    }
                    for setting_id, setting_align, setting_scale in BASELINE_PROTOCOLS:
                        protocol_metric = protocol_metrics_by_setting[setting_id]
                        protocol_rows.append(
                            {
                                "utt_id": metadata["utt_id"],
                                "split": metadata["split"],
                                "waveform_samples": waveform_samples,
                                "sample_rate": sample_rate,
                                "duration_sec": waveform_samples / sample_rate,
                                "setting_id": setting_id,
                                "protocol_name": setting_id,
                                "metric_align": setting_align,
                                "snr_scale_match": bool(setting_scale),
                                "metric_zero_mean": metric_zero_mean,
                                "waveform_snr_db": protocol_metric["waveform_snr_db"],
                                "si_sdr_db": protocol_metric["si_sdr_db"],
                                "stft_l1": protocol_metric["stft_l1"],
                                "best_lag_samples": protocol_metric["best_lag_samples"],
                                "input_rms": input_rms,
                                "output_rms": reconstructed_rms,
                            }
                        )
                comparison_rows.append(
                    {
                        "utt_id": metadata["utt_id"],
                        "split": metadata["split"],
                        "audio_path": metadata["audio_path"],
                        **_prefix_metrics("official", official_item_metrics),
                        **_prefix_metrics("continuous_sum", continuous_item_metrics),
                        **delta_official,
                    }
                )
                global_index = start + index
                utt_id = str(row["utt_id"])
                recon_cache[utt_id] = {
                    "original": waveforms[index].detach().cpu(),
                    "continuous_sum": continuous_recon[index].detach().cpu(),
                }
                if official_recon is not None:
                    recon_cache[utt_id]["official"] = official_recon[index].detach().cpu()
                if global_index < num_save_examples:
                    write_pcm_wave(output_dir / "examples" / f"{utt_id}_original.wav", waveforms[index], sample_rate)
                    write_pcm_wave(
                        output_dir / "examples" / f"{utt_id}_codec_recon_clean.wav",
                        clean_recon[index],
                        sample_rate,
                    )
                    write_pcm_wave(
                        output_dir / "examples" / f"{utt_id}_continuous_sum_recon.wav",
                        continuous_recon[index],
                        sample_rate,
                    )
                    if official_recon is not None:
                        write_pcm_wave(
                            output_dir / "examples" / f"{utt_id}_official_recon.wav",
                            official_recon[index],
                            sample_rate,
                        )

            if enable_latent_noise_sweep:
                signal_power = latents.square().mean(dim=(1, 2, 3), keepdim=True)
                for snr_db in latent_noise_snr_db:
                    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
                    noise_sigma = torch.sqrt(noise_power)
                    noise = torch.randn(
                        latents.shape,
                        generator=generator,
                        device=latents.device,
                        dtype=latents.dtype,
                    ) * noise_sigma
                    noisy_latents = latents + noise
                    noisy_recon = codec.decode_representation(noisy_latents)
                    per_item_noisy_metrics = [
                        waveform_metrics(
                            waveforms[index : index + 1],
                            noisy_recon[index : index + 1],
                            metric_align=metric_align,
                            max_lag_samples=max_lag_samples,
                            snr_scale_match=snr_scale_match,
                            metric_zero_mean=metric_zero_mean,
                        )
                        for index in range(len(batch_records))
                    ]
                    for index, record in enumerate(batch_records):
                        clean = continuous_metrics[index]
                        noisy = per_item_noisy_metrics[index]
                        metadata = _record_metadata(record, split)
                        noise_row = {
                            "utt_id": metadata["utt_id"],
                            "split": metadata["split"],
                            "latent_noise_snr_db": float(snr_db),
                            "latent_noise_power": float(noise_power[index].mean().detach().cpu().item()),
                            "latent_signal_power": float(signal_power[index].mean().detach().cpu().item()),
                            "latent_noise_sigma": float(noise_sigma[index].mean().detach().cpu().item()),
                            **noisy,
                            "delta_waveform_mse_vs_clean": noisy["waveform_mse"] - clean["waveform_mse"],
                            "delta_waveform_snr_db_vs_clean": noisy["waveform_snr_db"] - clean["waveform_snr_db"],
                            "delta_si_sdr_db_vs_clean": noisy["si_sdr_db"] - clean["si_sdr_db"],
                            "delta_stft_l1_vs_clean": noisy["stft_l1"] - clean["stft_l1"],
                        }
                        noise_rows.append(noise_row)
                        global_index = start + index
                        if global_index < num_save_examples:
                            utt_id = str(metadata["utt_id"])
                            label = f"{int(snr_db)}dB" if float(snr_db).is_integer() else f"{snr_db:g}dB"
                            write_pcm_wave(
                                output_dir / "examples" / f"{utt_id}_latent_noise_{label}.wav",
                                noisy_recon[index],
                                sample_rate,
                            )

    _write_csv(output_dir / "per_utterance_metrics.csv", clean_rows)
    _write_csv(output_dir / "decode_comparison_metrics.csv", comparison_rows)
    protocol_table_rows: list[dict[str, Any]] = []
    if protocol_comparison:
        _write_csv(output_dir / "baseline_protocol_detail.csv", protocol_rows)
        protocol_table_rows = _build_baseline_protocol_table(protocol_rows)
        _write_baseline_protocol_reports(output_dir, protocol_table_rows)
    if enable_latent_noise_sweep:
        _write_csv(output_dir / "latent_noise_sweep_metrics.csv", noise_rows)
    else:
        (output_dir / "latent_noise_sweep_metrics.csv").write_text("", encoding="utf-8")

    worst_rows = sorted(clean_rows, key=lambda row: float(row["clean_si_sdr"]))[: max(0, worst_k)]
    for row in worst_rows:
        utt_id = str(row["utt_id"])
        cached = recon_cache.get(utt_id)
        if not cached:
            continue
        write_pcm_wave(output_dir / "worst_samples" / f"{utt_id}_original.wav", cached["original"], sample_rate)
        if "official" in cached:
            write_pcm_wave(output_dir / "worst_samples" / f"{utt_id}_official_recon.wav", cached["official"], sample_rate)
        write_pcm_wave(
            output_dir / "worst_samples" / f"{utt_id}_continuous_sum_recon.wav",
            cached["continuous_sum"],
            sample_rate,
        )
        export_outlier_artifacts(
            output_dir=output_dir,
            utt_id=utt_id,
            original=cached["original"],
            official_recon=cached.get("official"),
            continuous_sum_recon=cached["continuous_sum"],
            sample_rate=sample_rate,
            max_lag_samples=max_lag_samples,
        )

    _save_plots(clean_rows, noise_rows, output_dir)
    summary = _summary(
        clean_rows=clean_rows,
        noise_rows=noise_rows,
        comparison_rows=comparison_rows,
        worst_rows=[
            {
                "utt_id": row["utt_id"],
                "audio_path": row["audio_path"],
                "used_decode_mode": row["used_decode_mode"],
                "clean_si_sdr": float(row["clean_si_sdr"]),
                "clean_waveform_snr": float(row["clean_waveform_snr"]),
                "clean_stft_l1": float(row["clean_stft_l1"]),
            }
            for row in worst_rows
        ],
        manifest_path=manifest_path,
        split=split,
        codec_type=codec_type,
        representation_shape=codec.representation_shape,
        sample_rate=sample_rate,
        waveform_samples=waveform_samples,
        latent_noise_sweep_used=enable_latent_noise_sweep,
        decode_mode=decode_mode,
        metric_align=metric_align,
        max_lag_samples=max_lag_samples,
        snr_scale_match=snr_scale_match,
        metric_zero_mean=metric_zero_mean,
    )
    summary["baseline_protocol_comparison"] = protocol_table_rows
    summary["diagnostic_excluded_metrics"] = _diagnostic_excluded_metrics(
        clean_rows,
        lag_outlier_threshold=lag_outlier_threshold,
    )
    summary["recommended_baseline_protocol"] = {
        "waveform_samples": 32000,
        "duration_sec": 2.0,
        "metric_align": "peak_xcorr",
        "snr_scale_match": True,
        "metric_zero_mean": True,
        "report": ["mean", "median", "si_sdr_le_minus_10_count"],
        "main_result_uses_all_samples": True,
        "outlier_threshold_si_sdr_db": -10.0,
    }
    if protocol_comparison:
        _write_final_baseline_report(output_dir, summary, protocol_table_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_report_markdown(output_dir / "report.md", summary)
    if summary["si_sdr_outliers_below_minus_10db"]:
        print("warning: SI-SDR <= -10 dB outliers:")
        for row in summary["si_sdr_outliers_below_minus_10db"]:
            print(f"  {row['utt_id']} {row['si_sdr_db']:.3f} dB {row['audio_path']}")
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a speech codec without JSCC or wireless channel")
    parser.add_argument("--config", default="configs/train_speechtokenizer.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--max_items", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--enable_latent_noise_sweep", type=str_to_bool, default=False)
    parser.add_argument("--latent_noise_snr_db", type=float, nargs="*", default=[30.0, 20.0, 10.0, 5.0, 0.0])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_save_examples", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--decode_mode", choices=["official", "continuous_sum", "both"], default="continuous_sum")
    parser.add_argument("--compare_official", type=str_to_bool, default=True)
    parser.add_argument("--metric_align", choices=["none", "peak_xcorr"], default="peak_xcorr")
    parser.add_argument("--max_lag_samples", type=int, default=1000)
    parser.add_argument("--snr_scale_match", type=str_to_bool, default=True)
    parser.add_argument("--metric_zero_mean", type=str_to_bool, default=True)
    parser.add_argument("--worst_k", type=int, default=10)
    parser.add_argument("--silence_rms_threshold", type=float, default=1e-4)
    parser.add_argument("--protocol_comparison", type=str_to_bool, default=False)
    parser.add_argument("--lag_outlier_threshold", type=int, default=None)
    parser.add_argument("--waveform_samples", type=int, nargs="*", default=[32000])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.waveform_samples and len(args.waveform_samples) > 1:
        summary = run_waveform_length_sweep(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    waveform_samples_override = args.waveform_samples[0] if args.waveform_samples else None
    summary = run_evaluation(
        config_path=args.config,
        manifest_path=args.manifest,
        split=args.split,
        output_dir=args.output_dir,
        max_items=args.max_items,
        batch_size=args.batch_size,
        enable_latent_noise_sweep=args.enable_latent_noise_sweep,
        latent_noise_snr_db=args.latent_noise_snr_db,
        num_save_examples=args.num_save_examples,
        device_name=args.device,
        seed=args.seed,
        decode_mode=args.decode_mode,
        metric_align=args.metric_align,
        max_lag_samples=args.max_lag_samples,
        snr_scale_match=args.snr_scale_match,
        metric_zero_mean=args.metric_zero_mean,
        worst_k=args.worst_k,
        silence_rms_threshold=args.silence_rms_threshold,
        protocol_comparison=args.protocol_comparison,
        lag_outlier_threshold=args.lag_outlier_threshold,
        waveform_samples_override=waveform_samples_override,
        compare_official=args.compare_official,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
