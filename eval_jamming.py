from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from speech_jscc.channels import compute_jsr
from speech_jscc.checkpoint import (
    codec_name,
    normalization_config,
    validate_checkpoint_metadata,
)
from speech_jscc.training.stage1 import validate_stage1_checkpoint_resources
from speech_jscc.config import load_config, resolve_device
from speech_jscc.data import CachedCodecDataset, codec_cache_namespace, resolve_waveform_splits
from speech_jscc.experiment import build_components
from speech_jscc.metrics import summarize_audio_metrics
from evaluation.paired import (
    estimate_transmitter_feedback,
    generate_paired_evaluation_batch,
    run_mode_on_paired_batch,
)
from channels.reliability import estimate_unreliable_mask
from models.channel_state import SINR_INDEX
from models.learned_gate import LearnedLayerGate, load_learned_gate_checkpoint
from models.latent_refiner import LatentRefiner, load_latent_refiner_checkpoint
from speech_jscc.layer_importance import resolve_layer_importance_config


def rule_based_layer_gates(
    channel_state: Tensor,
    layers: int,
    thresholds_db: list[float],
) -> Tensor:
    """Activate a deterministic prefix from post-estimation effective SINR."""
    if channel_state.ndim != 2:
        raise ValueError("channel_state must have shape [B,C]")
    quality_db = channel_state[:, SINR_INDEX] * 20.0
    thresholds = torch.as_tensor(thresholds_db, device=quality_db.device, dtype=quality_db.dtype)
    if thresholds.shape != (layers - 1,):
        raise ValueError("rule_gate_thresholds_db must contain L-1 values")
    if thresholds.numel() > 1 and torch.any(thresholds[1:] < thresholds[:-1]):
        raise ValueError("rule gate thresholds must be nondecreasing")
    enhancement = quality_db[:, None] >= thresholds[None, :]
    return torch.cat(
        (quality_db.new_ones((quality_db.shape[0], 1)), enhancement.to(quality_db.dtype)),
        dim=1,
    )


def _load_checkpoint(
    checkpoint: Path,
    model: nn.Module,
    config: dict[str, Any],
    device: torch.device,
    codec=None,
) -> LearnedLayerGate | None:
    if not checkpoint.exists():
        print(f"warning: {checkpoint} not found; evaluating random model initialization")
        model.checkpoint_metadata = {}
        model.metric_interpretation = "smoke_test_path_check"
        return None
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    metadata = state.get("metadata", {})
    if (metadata.get("training_stage") or {}).get("name") == "stage1_fixed_tx":
        validate_stage1_checkpoint_resources(state, model, config)
    model.load_state_dict(state["model"], strict=True)
    if metadata:
        if codec is None:
            raise ValueError("codec is required to validate checkpoint metadata")
        validate_checkpoint_metadata(metadata, config, codec)
    model.checkpoint_metadata = metadata
    model.metric_interpretation = (
        "trained_checkpoint_performance"
        if metadata.get("speech_tokenizer_metric_valid", False)
        or metadata.get("codec_name") == "mock_continuous"
        else "smoke_test_path_check"
    )
    print(f"loaded={checkpoint}")
    learned_state = state.get("learned_gate")
    if learned_state is None:
        return None
    if "state_dict" in learned_state:
        gate = load_learned_gate_checkpoint(learned_state, device)
    else:
        gate = LearnedLayerGate(
            config["model"]["channel_state_dim"],
            config["model"]["layers"],
            config["eval"].get("learned_gate_hidden_dim", 32),
        ).to(device)
        gate.load_state_dict(learned_state)
    gate.eval()
    return gate


def _load_refiner(checkpoint: Path, device: torch.device) -> LatentRefiner | None:
    if not checkpoint.exists():
        return None
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    payload = state.get("latent_refiner")
    if payload is None:
        return None
    refiner = load_latent_refiner_checkpoint(payload, device)
    refiner.eval()
    return refiner


def available_adaptation_modes(
    requested: list[str],
    learned_gate: LearnedLayerGate | None,
) -> list[str]:
    supported = {"uniform", "rule_based", "learned", "learned_gate"}
    unknown = set(requested) - supported
    if unknown:
        raise ValueError(f"unsupported adaptation modes: {sorted(unknown)}")
    modes = [
        mode
        for mode in requested
        if mode not in {"learned", "learned_gate"} or learned_gate is not None
    ]
    if any(mode in {"learned", "learned_gate"} for mode in requested) and learned_gate is None:
        print("learned gating requested but checkpoint has no 'learned_gate' state; skipping")
    if not modes:
        raise ValueError("no available adaptation modes")
    return modes


def _gates_for_mode(
    mode: str,
    state: Tensor,
    layers: int,
    thresholds_db: list[float],
    learned_gate: LearnedLayerGate | None,
) -> Tensor:
    if mode == "uniform":
        return state.new_ones((state.shape[0], layers))
    if mode == "rule_based":
        return rule_based_layer_gates(state, layers, thresholds_db)
    if mode in {"learned", "learned_gate"} and learned_gate is not None:
        return learned_gate(state)
    raise RuntimeError(f"adaptation mode {mode!r} is unavailable")


def evaluate_paired_condition(
    codec,
    model,
    learned_gate: LearnedLayerGate | None,
    config: dict[str, Any],
    device: torch.device,
    modes: list[str],
    jammer_type: str,
    snr_value: float,
    jsr_value: float,
    equalizer: str,
    seed_base: int,
    latent_refiner: LatentRefiner | None = None,
) -> list[dict[str, Any]]:
    eval_config = config["eval"]
    channel_config = config["channel"]
    sample_rate = int(getattr(codec, "sample_rate", config["codec"].get("sample_rate", 16000)))
    layers = config["model"]["layers"]
    layer_weights = torch.as_tensor(
        eval_config["layer_weights"], device=device, dtype=torch.float32
    )
    if layer_weights.shape != (layers,) or torch.any(layer_weights < 0) or layer_weights.sum() <= 0:
        raise ValueError("eval.layer_weights must contain L nonnegative values with positive sum")
    metadata = getattr(model, "checkpoint_metadata", {})
    is_stage1_fixed_tx = (
        metadata.get("training_stage", {}).get("label")
        == "fixed_tx_channel_aware_rx_jammer_agnostic"
    )
    normalization = metadata.get(
        "normalization", normalization_config(config, section="eval")
    )
    normalization_mode = normalization.get("mode", "none")
    epsilon = float(normalization.get("epsilon", 1e-8))
    if normalization_mode not in {
        "none", "raw", "per_layer_power", "per_layer_nmse", "global_power"
    }:
        raise ValueError(f"unsupported latent normalization mode: {normalization_mode}")

    validation_dataset = None
    train_paths, val_paths = resolve_waveform_splits(config.get("data", {}), config["seed"])
    if val_paths:
        validation_dataset = CachedCodecDataset(
            val_paths,
            codec,
            sample_rate=sample_rate,
            waveform_samples=config["codec"]["waveform_samples"],
            device=device,
            split="val",
            cache_dir=config.get("data", {}).get("latent_cache_dir"),
            cache_namespace=codec_cache_namespace(config, codec),
        )
    metric_interpretation = getattr(
        model, "metric_interpretation", "smoke_test_path_check"
    )
    if codec_name(config) == "speechtokenizer" and validation_dataset is None:
        metric_interpretation = "smoke_test_path_check"

    allocation_modes = ["uniform"] if is_stage1_fixed_tx else eval_config.get("allocation_modes", ["uniform"])
    requested_refiner_modes = ["no_refiner"] if is_stage1_fixed_tx else eval_config.get("refiner_modes", ["no_refiner"])
    refiner_modes = [
        mode
        for mode in requested_refiner_modes
        if mode == "no_refiner" or latent_refiner is not None
    ]
    keys = [
        (mode, allocation_mode, refiner_mode)
        for mode in modes
        for allocation_mode in allocation_modes
        for refiner_mode in refiner_modes
    ]
    accumulators: dict[tuple[str, str, str], dict[str, list[Any]]] = {
        key: {
            "layer_mse": [],
            "latent_loss": [],
            "latent_mse": [],
            "waveform_loss": [],
            "si_sdr": [],
            "stoi": [],
            "stoi_available": [],
            "stoi_error": [],
            "effective_sinr": [],
            "measured_jsr": [],
            "post_channel_jsr": [],
            "active_layers": [],
            "mask_ratio": [],
            "csi_nmse": [],
            "pilot_evm": [],
            "alpha": [],
            "encoder_state": [],
            "decoder_state": [],
        }
        for key in keys
    }
    channel_shape = tuple(model.encoder.channel_shape)
    fading = channel_config.get("fading", "flat" if len(channel_shape) == 1 else "ofdm")
    channel_estimator = channel_config.get("channel_estimator", "auto")
    estimator_num_taps = channel_config.get("estimator_num_taps")
    estimator_ridge_lambda = channel_config.get("estimator_ridge_lambda", 1e-6)
    with torch.no_grad():
        for batch_index in range(eval_config["batches"]):
            batch_size = eval_config["batch_size"]
            waveform = representation = None
            if validation_dataset is not None:
                examples = [
                    validation_dataset[(batch_index * batch_size + offset) % len(validation_dataset)]
                    for offset in range(batch_size)
                ]
                representation = torch.stack([example[0] for example in examples])
                waveform = torch.stack([example[1] for example in examples])
            paired_batch = generate_paired_evaluation_batch(
                codec,
                batch_size=batch_size,
                waveform_samples=config["codec"]["waveform_samples"],
                channel_shape=channel_shape,
                snr_db=snr_value,
                jsr_db=jsr_value,
                jammer_type=jammer_type,
                jammed_fraction=channel_config["jammed_fraction"],
                pilot_spacing=channel_config.get("pilot_spacing", 4),
                pilot_time_spacing=channel_config.get("pilot_time_spacing"),
                target_power=config["model"]["target_power"],
                seed=seed_base + batch_index,
                device=device,
                fading=fading,
                num_taps=channel_config.get("num_taps", 6),
                pdp_decay=channel_config.get("pdp_decay", 0.7),
                channel_estimator=channel_estimator,
                estimator_num_taps=estimator_num_taps,
                estimator_ridge_lambda=estimator_ridge_lambda,
                waveform=waveform,
                representation=representation,
            )
            if is_stage1_fixed_tx:
                state = torch.zeros(
                    batch_size,
                    config["model"]["channel_state_dim"],
                    device=device,
                    dtype=paired_batch.representation.dtype,
                )
                reliability = torch.ones_like(paired_batch.noise.real)
            else:
                feedback = estimate_transmitter_feedback(
                    paired_batch,
                    transmitter_csi=eval_config.get("transmitter_csi", True),
                    fading=fading,
                    channel_estimator=channel_estimator,
                    estimator_num_taps=estimator_num_taps,
                    estimator_ridge_lambda=estimator_ridge_lambda,
                )
                state = feedback["state"]
                reliability = feedback["reliability"]
            for mode in modes:
                gates = _gates_for_mode(
                    mode,
                    state,
                    layers,
                    eval_config["rule_gate_thresholds_db"],
                    learned_gate,
                )
                for allocation_mode in allocation_modes:
                    result = run_mode_on_paired_batch(
                        codec,
                        model,
                        paired_batch,
                        state,
                        gates,
                        equalizer=equalizer,
                        fading=fading,
                        channel_estimator=channel_estimator,
                        estimator_num_taps=estimator_num_taps,
                        estimator_ridge_lambda=estimator_ridge_lambda,
                        allocation_mode=allocation_mode,
                        importance_order=eval_config.get("layer_importance_order"),
                        resource_reliability=reliability,
                        layer_power_allocation=torch.ones(layers, device=device),
                        receiver_state_mode="observable_v1" if is_stage1_fixed_tx else "legacy",
                    )
                    estimated_mask = estimate_unreliable_mask(
                        reliability, eval_config.get("unreliable_fraction", 0.25)
                    )
                    for refiner_mode in refiner_modes:
                        if refiner_mode == "no_refiner":
                            reconstruction = result["reconstruction"]
                        elif refiner_mode == "refiner_oracle_mask":
                            reconstruction = latent_refiner(
                                result["reconstruction"],
                                result["decoder_state"],
                                paired_batch.jammer_mask,
                            )
                        elif refiner_mode == "refiner_estimated_mask":
                            reconstruction = latent_refiner(
                                result["reconstruction"],
                                result["decoder_state"],
                                estimated_mask,
                            )
                        else:
                            raise ValueError(f"unsupported refiner mode: {refiner_mode}")
                        layer_mse = (
                            reconstruction - paired_batch.representation
                        ).square().mean(dim=(0, 2, 3))
                        accumulator = accumulators[(mode, allocation_mode, refiner_mode)]
                        accumulator["layer_mse"].append(layer_mse)
                        accumulator["latent_mse"].append(
                            (layer_mse * layer_weights).sum() / layer_weights.sum()
                        )
                        accumulator["latent_loss"].append(
                            (
                                (
                                    layer_mse
                                    if normalization_mode in {"none", "raw"}
                                    else layer_mse
                                    / (
                                        paired_batch.representation.square()
                                        .mean(dim=(0, 2, 3))
                                        .clamp_min(epsilon)
                                        if normalization_mode in {"per_layer_power", "per_layer_nmse"}
                                        else paired_batch.representation.square().mean().clamp_min(epsilon)
                                    )
                                )
                                * layer_weights
                            ).sum()
                            / layer_weights.sum()
                        )
                        if refiner_mode == "no_refiner":
                            decoded_waveform = result["decoded_waveform"]
                        else:
                            decoded_waveform = codec.decode_representation(reconstruction)
                        if paired_batch.waveform is not None:
                            accumulator["waveform_loss"].append(
                                (decoded_waveform - paired_batch.waveform).square().mean()
                            )
                            audio_metrics = summarize_audio_metrics(
                                paired_batch.waveform,
                                decoded_waveform,
                                sample_rate,
                                enable_stoi=bool(eval_config.get("enable_stoi", False)),
                            )
                            accumulator["si_sdr"].append(
                                decoded_waveform.new_tensor(float(audio_metrics["si_sdr_db"]))
                            )
                            if audio_metrics["stoi"] is not None:
                                accumulator["stoi"].append(
                                    decoded_waveform.new_tensor(float(audio_metrics["stoi"]))
                                )
                            accumulator["stoi_available"].append(
                                decoded_waveform.new_tensor(
                                    1.0 if audio_metrics["stoi_available"] else 0.0
                                )
                            )
                            if audio_metrics["stoi_error"]:
                                accumulator["stoi_error"].append(str(audio_metrics["stoi_error"]))
                        accumulator["effective_sinr"].append(
                            (10.0 * torch.log10(
                                result["post_equalization_sinr"].clamp_min(1e-12)
                            )).mean()
                        )
                        accumulator["measured_jsr"].append(
                            compute_jsr(result["transmitted"], result["jammer"], db=True).mean()
                        )
                        accumulator["post_channel_jsr"].append(
                            result["post_channel_jsr"].clamp_min(1e-12).log10().mul(10.0).mean()
                        )
                        accumulator["active_layers"].append(gates.sum(dim=1).mean())
                        accumulator["mask_ratio"].append(result["jammer_mask"].float().mean())
                        accumulator["csi_nmse"].append(result["csi_nmse"].mean())
                        accumulator["pilot_evm"].append(result["pilot_evm"].mean())
                        accumulator["alpha"].append(gates.mean(dim=0))
                        accumulator["encoder_state"].append(result["encoder_state"].mean(dim=0))
                        accumulator["decoder_state"].append(result["decoder_state"].mean(dim=0))

    rows = []
    for mode, allocation_mode, refiner_mode in keys:
        accumulator = accumulators[(mode, allocation_mode, refiner_mode)]
        mean_layer_mse = torch.stack(accumulator["layer_mse"]).mean(dim=0)
        waveform_mse = (
            torch.stack(accumulator["waveform_loss"]).mean().item()
            if accumulator["waveform_loss"]
            else ""
        )
        si_sdr_db = (
            torch.stack(accumulator["si_sdr"]).mean().item()
            if accumulator["si_sdr"]
            else ""
        )
        stoi = (
            torch.stack(accumulator["stoi"]).mean().item()
            if accumulator["stoi"]
            else ""
        )
        stoi_available = (
            bool(torch.stack(accumulator["stoi_available"]).max().item())
            if accumulator["stoi_available"]
            else False
        )
        stoi_error = "; ".join(sorted(set(accumulator["stoi_error"])))
        row: dict[str, Any] = {
            "adaptation_mode": mode,
            "allocation_mode": allocation_mode,
            "refiner_mode": refiner_mode,
            "equalizer": equalizer,
            "paired_seed": seed_base,
            "jammer": jammer_type,
            "snr_db": float(snr_value),
            "jsr_db": float(jsr_value),
            "latent_mse": torch.stack(accumulator["latent_mse"]).mean().item(),
            "latent_loss": torch.stack(accumulator["latent_loss"]).mean().item(),
            "latent_loss_normalization": normalization_mode,
            "metric_interpretation": metric_interpretation,
            "checkpoint_kind": metadata.get("checkpoint_kind", "untrained_or_legacy"),
            "codec_name": codec_name(config),
            "evaluation_data": "validation_waveforms" if validation_dataset is not None else "synthetic_smoke",
            "waveform_mse": waveform_mse,
            "si_sdr_db": si_sdr_db,
            "stoi": stoi,
            "stoi_available": stoi_available,
            "stoi_error": stoi_error,
            "effective_sinr_db": torch.stack(accumulator["effective_sinr"]).mean().item(),
            "measured_jsr_db": torch.stack(accumulator["measured_jsr"]).mean().item(),
            "post_channel_jsr_db": torch.stack(accumulator["post_channel_jsr"]).mean().item(),
            "csi_nmse": torch.stack(accumulator["csi_nmse"]).mean().item(),
            "pilot_evm": torch.stack(accumulator["pilot_evm"]).mean().item(),
            "fading_model": channel_config.get("fading", "flat" if len(channel_shape) == 1 else "ofdm"),
            "channel_estimator": channel_config.get("channel_estimator", "auto"),
            "estimator_num_taps": channel_config.get("estimator_num_taps", ""),
            "estimator_ridge_lambda": channel_config.get("estimator_ridge_lambda", ""),
            "mean_active_layers": torch.stack(accumulator["active_layers"]).mean().item(),
            "jammer_mask_ratio": torch.stack(accumulator["mask_ratio"]).mean().item(),
            "pesq_placeholder": "",
            "speaker_similarity_placeholder": "",
            "wer_optional": "",
        }
        for layer, value in enumerate(mean_layer_mse.tolist(), start=1):
            row[f"layer_{layer}_mse"] = value
        mean_alpha = torch.stack(accumulator["alpha"]).mean(dim=0)
        for layer, value in enumerate(mean_alpha.tolist(), start=1):
            row[f"alpha_{layer}"] = value
        mean_encoder_state = torch.stack(accumulator["encoder_state"]).mean(dim=0)
        mean_decoder_state = torch.stack(accumulator["decoder_state"]).mean(dim=0)
        for index, value in enumerate(mean_encoder_state.tolist()):
            row[f"encoder_c_{index}"] = value
        for index, value in enumerate(mean_decoder_state.tolist()):
            row[f"decoder_c_{index}"] = value
        rows.append(row)
    return rows


def save_plots(rows: list[dict[str, Any]], output_dir: Path, layers: int) -> list[Path]:
    import os

    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = list(
        dict.fromkeys(
            (
                str(row["adaptation_mode"]),
                str(row.get("allocation_mode", "uniform")),
                str(row.get("refiner_mode", "no_refiner")),
                str(row.get("equalizer", "estimated")),
            )
            for row in rows
        )
    )
    snr_values = sorted({float(row["snr_db"]) for row in rows})
    paths: list[Path] = []

    fig, axis = plt.subplots(figsize=(7, 4))
    for mode, allocation, refiner, equalizer in groups:
        values = [
            sum(
                float(row[f"layer_{layer}_mse"])
                for row in rows
                if row["adaptation_mode"] == mode
                and row.get("allocation_mode", "uniform") == allocation
                and row.get("refiner_mode", "no_refiner") == refiner
                and row.get("equalizer", "estimated") == equalizer
            )
            / sum(
                1
                for row in rows
                if row["adaptation_mode"] == mode
                and row.get("allocation_mode", "uniform") == allocation
                and row.get("refiner_mode", "no_refiner") == refiner
                and row.get("equalizer", "estimated") == equalizer
            )
            for layer in range(1, layers + 1)
        ]
        axis.plot(
            range(1, layers + 1),
            values,
            marker="o",
            label=f"{mode}/{allocation}/{refiner}/{equalizer}",
        )
    axis.set(xlabel="Codec layer", ylabel="Mean latent MSE", title="Layer-wise reconstruction")
    axis.legend()
    axis.grid(alpha=0.25)
    path = output_dir / "layer_mse.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    for metric, ylabel, filename, title in (
        ("waveform_mse", "Waveform MSE", "waveform_metrics.png", "Reconstructed waveform metrics"),
        ("effective_sinr_db", "Effective SINR (dB)", "effective_sinr.png", "Effective SINR"),
    ):
        fig, axis = plt.subplots(figsize=(7, 4))
        for mode, allocation, refiner, equalizer in groups:
            values = []
            for snr in snr_values:
                selected = []
                for row in rows:
                    if (
                        row["adaptation_mode"] == mode
                        and row.get("allocation_mode", "uniform") == allocation
                        and row.get("refiner_mode", "no_refiner") == refiner
                        and row.get("equalizer", "estimated") == equalizer
                        and float(row["snr_db"]) == snr
                        and row.get(metric) not in {"", None}
                    ):
                        selected.append(float(row[metric]))
                values.append(sum(selected) / len(selected) if selected else float("nan"))
            axis.plot(
                snr_values,
                values,
                marker="o",
                label=f"{mode}/{allocation}/{refiner}/{equalizer}",
            )
        axis.set(xlabel="Nominal SNR (dB)", ylabel=ylabel, title=title)
        axis.legend()
        axis.grid(alpha=0.25)
        path = output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep fading, jamming, and JSCC adaptation")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    device = resolve_device(config["device"])
    codec, model = build_components(config, device)
    resolved_importance = resolve_layer_importance_config(
        config,
        section="eval",
        expected_representation_shape=codec.representation_shape,
    )
    if resolved_importance.artifact_path is not None:
        eval_config = config["eval"]
        if resolved_importance.layer_weights is not None:
            eval_config["layer_weights"] = resolved_importance.layer_weights
        if resolved_importance.layer_importance_order is not None:
            eval_config["layer_importance_order"] = resolved_importance.layer_importance_order
        if resolved_importance.base_layers is not None:
            eval_config["base_layers"] = resolved_importance.base_layers
        print(
            "layer_importance_artifact="
            f"{resolved_importance.artifact_path} hash={resolved_importance.artifact_hash} "
            f"weights={eval_config.get('layer_weights')} "
            f"order={eval_config.get('layer_importance_order')} "
            f"base_layers={eval_config.get('base_layers')}"
        )
    checkpoint = Path(args.checkpoint or config["eval"]["checkpoint"])
    learned_gate = _load_checkpoint(checkpoint, model, config, device, codec)
    latent_refiner = _load_refiner(checkpoint, device)
    if codec_name(config) == "speechtokenizer" and getattr(
        model, "metric_interpretation", "smoke_test_path_check"
    ) != "trained_checkpoint_performance":
        print(
            "WARNING: untrained SpeechTokenizer-latent checkpoint MSE is path-check only; "
            "do not report it as model performance."
        )
    elif codec_name(config) == "speechtokenizer" and not config.get("data"):
        print(
            "WARNING: no validation waveform corpus is configured; this evaluation remains "
            "a path smoke test even with a trained checkpoint."
        )
    model.eval()
    modes = available_adaptation_modes(config["eval"]["adaptation_modes"], learned_gate)
    if (
        getattr(model, "checkpoint_metadata", {})
        .get("training_stage", {})
        .get("label")
        == "fixed_tx_channel_aware_rx_jammer_agnostic"
    ):
        modes = ["uniform"]
        print("stage1_fixed_tx_evaluation=true receiver_state_mode=observable_v1")

    rows = []
    condition_index = 0
    for jammer_type in config["channel"]["jammer_types"]:
        for snr_db in config["channel"]["snr_db"]:
            for jsr_db in config["channel"]["jsr_db"]:
                seed_base = config["eval"].get("paired_seed", config["seed"]) + condition_index * 10_000
                for equalizer in config["eval"].get("equalizer_modes", ["estimated"]):
                    condition_rows = evaluate_paired_condition(
                        codec,
                        model,
                        learned_gate,
                        config,
                        device,
                        modes,
                        jammer_type,
                        snr_db,
                        jsr_db,
                        equalizer,
                        seed_base,
                        latent_refiner,
                    )
                    rows.extend(condition_rows)
                    for row in condition_rows:
                        mode = row["adaptation_mode"]
                        print(
                            f"mode={mode} allocation={row['allocation_mode']} "
                            f"refiner={row['refiner_mode']} eq={equalizer} jammer={jammer_type} "
                            f"snr={snr_db:g} jsr={jsr_db:g} latent_mse={row['latent_mse']:.6f} "
                            f"latent_loss={row['latent_loss']:.6f} "
                            f"status={row['metric_interpretation']} "
                            f"sinr={row['effective_sinr_db']:.2f}dB nmse={row['csi_nmse']:.4f}"
                        )
                condition_index += 1

    output = Path(config["eval"]["output_csv"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_paths = save_plots(rows, Path(config["eval"]["plot_dir"]), config["model"]["layers"])
    print(f"rows={len(rows)} csv={output} plots={','.join(map(str, plot_paths))}")
    if config["eval"].get("enable_optional_wer", False):
        print("optional WER requested but no ASR evaluator is configured; CSV field remains empty")


if __name__ == "__main__":
    main()
