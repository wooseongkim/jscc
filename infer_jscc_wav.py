from __future__ import annotations

import argparse
import contextlib
import json
import sys
import wave
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from channels.jammer import compute_jsr
from channels.reliability import estimate_unreliable_mask
from eval_jamming import (
    _gates_for_mode,
    _load_checkpoint,
    _load_refiner,
)
from evaluation.paired import (
    estimate_transmitter_feedback,
    generate_paired_evaluation_batch,
    run_mode_on_paired_batch,
)
from speech_jscc.checkpoint import codec_name
from speech_jscc.config import load_config, resolve_device
from speech_jscc.data import load_waveform_segment
from speech_jscc.experiment import build_components


ADAPTATION_MODES = ("uniform", "rule_based", "learned_gate")
ALLOCATION_MODES = ("uniform", "random", "reliability_greedy")
EQUALIZERS = ("estimated", "oracle")
JAMMERS = ("barrage", "narrowband", "burst", "pilot")
REFINER_MODES = ("no_refiner", "refiner_oracle_mask", "refiner_estimated_mask")


def _first_float(config: dict[str, Any], section: str, key: str, fallback: float) -> float:
    value = config.get(section, {}).get(key, fallback)
    if isinstance(value, (list, tuple)):
        if not value:
            return fallback
        return float(value[0])
    return float(value)


def _first_string(config: dict[str, Any], section: str, key: str, fallback: str) -> str:
    value = config.get(section, {}).get(key, fallback)
    if isinstance(value, (list, tuple)):
        if not value:
            return fallback
        return str(value[0])
    return str(value)


def _default_seed(config: dict[str, Any]) -> int:
    eval_config = config.get("eval", {})
    return int(eval_config.get("paired_seed", config.get("seed", 0)))


def _resolve_checkpoint(args: argparse.Namespace, config: dict[str, Any]) -> Path | None:
    value = args.checkpoint
    if value is None:
        value = config.get("eval", {}).get("checkpoint")
    return Path(value) if value else None


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


def _load_optional_states(
    checkpoint: Path | None,
    model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
    codec,
) -> tuple[Any | None, Any | None, dict[str, Any], str]:
    if checkpoint is None or not checkpoint.exists():
        model.checkpoint_metadata = {}
        model.metric_interpretation = "smoke_test_path_check"
        return None, None, {}, "smoke_test_path_check"
    with contextlib.redirect_stdout(sys.stderr):
        learned_gate = _load_checkpoint(checkpoint, model, config, device, codec=codec)
        latent_refiner = _load_refiner(checkpoint, device)
    metadata = getattr(model, "checkpoint_metadata", {})
    interpretation = getattr(model, "metric_interpretation", "smoke_test_path_check")
    return learned_gate, latent_refiner, metadata, interpretation


def _jsonable_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (value.item() if isinstance(value, Tensor) and value.numel() == 1 else value)
        for key, value in metrics.items()
    }


def _scalar(value: Tensor) -> float:
    return float(value.detach().mean().cpu().item())


def run_inference(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Tensor]]:
    config = load_config(args.config)
    requested_device = args.device or config.get("device", "auto")
    device = resolve_device(requested_device)
    seed = int(args.seed) if args.seed is not None else _default_seed(config)
    torch.manual_seed(seed)

    codec, model = build_components(config, device)
    model.eval()
    codec.eval()

    checkpoint = _resolve_checkpoint(args, config)
    learned_gate, latent_refiner, metadata, metric_interpretation = _load_optional_states(
        checkpoint, model, config, device, codec
    )
    if args.adaptation_mode == "learned_gate" and learned_gate is None:
        raise RuntimeError(
            "adaptation-mode=learned_gate requires a checkpoint containing 'learned_gate'"
        )
    if args.refiner_mode != "no_refiner" and latent_refiner is None:
        raise RuntimeError(
            f"refiner-mode={args.refiner_mode} requires a checkpoint containing 'latent_refiner'"
        )

    sample_rate = int(getattr(codec, "sample_rate", config.get("codec", {}).get("sample_rate", 16000)))
    waveform_samples = int(config["codec"]["waveform_samples"])
    waveform = load_waveform_segment(args.input, sample_rate, waveform_samples).to(device).unsqueeze(0)

    with torch.inference_mode():
        representation = codec.encode_waveform(waveform)
        channel_shape = tuple(model.encoder.channel_shape)
        fading = "flat" if len(channel_shape) == 1 else "ofdm"
        snr_db = float(
            args.snr_db
            if args.snr_db is not None
            else _first_float(config, "channel", "snr_db", 8.0)
        )
        jsr_db = float(
            args.jsr_db
            if args.jsr_db is not None
            else _first_float(config, "channel", "jsr_db", 0.0)
        )
        jammer = str(
            args.jammer
            if args.jammer is not None
            else _first_string(config, "channel", "jammer_types", "pilot")
        )
        allocation_mode = str(
            args.allocation_mode
            if args.allocation_mode is not None
            else _first_string(config, "eval", "allocation_modes", "uniform")
        )
        paired_batch = generate_paired_evaluation_batch(
            codec,
            batch_size=1,
            waveform_samples=waveform_samples,
            channel_shape=channel_shape,
            snr_db=snr_db,
            jsr_db=jsr_db,
            jammer_type=jammer,
            jammed_fraction=float(config["channel"]["jammed_fraction"]),
            pilot_spacing=int(config["channel"].get("pilot_spacing", 4)),
            pilot_time_spacing=config["channel"].get("pilot_time_spacing"),
            target_power=float(config["model"]["target_power"]),
            seed=seed,
            device=device,
            fading=fading,
            waveform=waveform,
            representation=representation,
        )
        feedback = estimate_transmitter_feedback(
            paired_batch,
            transmitter_csi=bool(config.get("eval", {}).get("transmitter_csi", True)),
            fading=fading,
        )
        state = feedback["state"]
        reliability = feedback["reliability"]
        gates = _gates_for_mode(
            args.adaptation_mode,
            state,
            int(config["model"]["layers"]),
            list(config.get("eval", {}).get("rule_gate_thresholds_db", [])),
            learned_gate,
        )
        result = run_mode_on_paired_batch(
            codec,
            model,
            paired_batch,
            state,
            gates,
            equalizer=args.equalizer,
            fading=fading,
            allocation_mode=allocation_mode,
            importance_order=config.get("eval", {}).get("layer_importance_order"),
            resource_reliability=reliability,
        )

        if args.refiner_mode == "no_refiner":
            final_reconstruction = result["reconstruction"]
        elif args.refiner_mode == "refiner_oracle_mask":
            final_reconstruction = latent_refiner(
                result["reconstruction"],
                result["decoder_state"],
                paired_batch.jammer_mask,
            )
        elif args.refiner_mode == "refiner_estimated_mask":
            estimated_mask = estimate_unreliable_mask(
                reliability,
                float(config.get("eval", {}).get("unreliable_fraction", 0.25)),
            )
            final_reconstruction = latent_refiner(
                result["reconstruction"],
                result["decoder_state"],
                estimated_mask,
            )
        else:
            raise ValueError(f"unsupported refiner mode: {args.refiner_mode}")

        decoded_waveform = codec.decode_representation(final_reconstruction)

    write_pcm_wave(args.output, decoded_waveform[0], sample_rate)

    latent_mse = (final_reconstruction - paired_batch.representation).square().mean()
    waveform_mse = (decoded_waveform - paired_batch.waveform).square().mean()
    effective_sinr_db = 10.0 * torch.log10(result["post_equalization_sinr"].clamp_min(1e-12))
    measured_jsr_db = compute_jsr(result["transmitted"], result["jammer"], db=True)
    metrics = {
        "input": str(args.input),
        "output": str(args.output),
        "checkpoint": str(checkpoint) if checkpoint is not None and checkpoint.exists() else None,
        "codec_name": metadata.get("codec_name", codec_name(config)),
        "sample_rate": sample_rate,
        "waveform_samples": waveform_samples,
        "snr_db": snr_db,
        "jsr_db": jsr_db,
        "jammer": jammer,
        "equalizer": args.equalizer,
        "adaptation_mode": args.adaptation_mode,
        "allocation_mode": allocation_mode,
        "refiner_mode": args.refiner_mode,
        "latent_mse": _scalar(latent_mse),
        "waveform_mse": _scalar(waveform_mse),
        "effective_sinr_db": _scalar(effective_sinr_db),
        "measured_jsr_db": _scalar(measured_jsr_db),
        "csi_nmse": _scalar(result["csi_nmse"]),
        "pilot_evm": _scalar(result["pilot_evm"]),
        "jammer_mask_ratio": _scalar(result["jammer_mask"].float()),
        "active_layers": _scalar(gates.sum(dim=1)),
        "metric_interpretation": metric_interpretation,
    }
    tensors = {
        "source_waveform": waveform.detach().cpu(),
        "target_representation": representation.detach().cpu(),
        "raw_reconstruction": result["reconstruction"].detach().cpu(),
        "final_reconstruction": final_reconstruction.detach().cpu(),
        "decoded_waveform": decoded_waveform.detach().cpu(),
        "transmitted": result["transmitted"].detach().cpu(),
        "received": result["received"].detach().cpu(),
        "estimated_channel": result["estimated_channel"].detach().cpu(),
        "jammer": result["jammer"].detach().cpu(),
        "jammer_mask": result["jammer_mask"].detach().cpu(),
        "pilot_mask": result["pilot_mask"].detach().cpu(),
        "encoder_state": result["encoder_state"].detach().cpu(),
        "decoder_state": result["decoder_state"].detach().cpu(),
        "layer_gates": gates.detach().cpu(),
        "metrics": metrics,
    }
    return _jsonable_metrics(metrics), tensors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full WAV-to-WAV continuous-latent JSCC inference"
    )
    parser.add_argument("--config", default="configs/eval_speechtokenizer.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--snr-db", type=float, default=None)
    parser.add_argument("--jsr-db", type=float, default=None)
    parser.add_argument("--jammer", choices=JAMMERS, default=None)
    parser.add_argument("--equalizer", choices=EQUALIZERS, default="estimated")
    parser.add_argument("--adaptation-mode", choices=ADAPTATION_MODES, default="uniform")
    parser.add_argument("--allocation-mode", choices=ALLOCATION_MODES, default=None)
    parser.add_argument("--refiner-mode", choices=REFINER_MODES, default="no_refiner")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-pt", default=None)
    parser.add_argument("--metrics-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics, tensors = run_inference(args)
    if args.save_pt:
        save_path = Path(args.save_pt)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensors, save_path)
    if args.metrics_json:
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
