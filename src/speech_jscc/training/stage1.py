from __future__ import annotations

import random
from typing import Any

import torch
from torch import Tensor

from evaluation.paired import PairedEvaluationBatch, run_mode_on_paired_batch
from models.observable_channel_state import OBSERVABLE_RECEIVER_STATE_FEATURES
from train_latent_jscc import layer_weighted_latent_mse


STAGE1_LABEL = "fixed_tx_channel_aware_rx_jammer_agnostic"


def build_stage1_optimizer(
    model,
    *,
    learning_rate: float,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Create the Stage-1 optimizer over JSCC encoder and decoder only."""
    parameters = list(model.encoder.parameters()) + list(model.decoder.parameters())
    if not parameters:
        raise ValueError("Stage-1 model has no encoder/decoder parameters")
    return torch.optim.Adam(parameters, lr=float(learning_rate), weight_decay=float(weight_decay))


def _optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}


def _model_parameter_ids(model) -> set[int]:
    return {id(parameter) for parameter in list(model.encoder.parameters()) + list(model.decoder.parameters())}


def assert_stage1_startup_invariants(
    codec,
    model,
    optimizer: torch.optim.Optimizer,
    *,
    allocation_mode: str,
) -> dict[str, Any]:
    if getattr(codec, "training", False):
        raise RuntimeError("Stage-1 requires the codec wrapper to be in eval mode")
    if any(parameter.requires_grad for parameter in codec.parameters()):
        raise RuntimeError("Stage-1 requires every codec parameter to be frozen")
    if hasattr(model, "learned_gate") or hasattr(model, "latent_refiner"):
        raise RuntimeError("Stage-1 model must not instantiate learned_gate or latent_refiner")
    expected_ids = _model_parameter_ids(model)
    actual_ids = _optimizer_parameter_ids(optimizer)
    if actual_ids != expected_ids:
        raise RuntimeError("Stage-1 optimizer must contain exactly encoder and decoder parameters")
    if allocation_mode != "uniform":
        raise RuntimeError("Stage-1 resource allocation mode must be uniform")
    default_power = model.encoder.default_layer_power.detach()
    if not torch.allclose(default_power, torch.ones_like(default_power)):
        raise RuntimeError("Stage-1 default layer power must be uniform ones")
    return {
        "transmitter_policy": {
            "state_mode": "neutral",
            "gate_mode": "all_ones",
            "channel_use_mode": "fixed_equal",
            "power_mode": "uniform",
            "allocation_mode": "uniform",
            "layer_channel_uses": list(model.encoder.layer_channel_uses),
        }
    }


def stage1_fixed_tx_step(
    codec,
    model,
    paired_batch: PairedEvaluationBatch,
    optimizer: torch.optim.Optimizer | None,
    layer_weights: Tensor,
    *,
    latent_normalization: str | dict[str, Any],
    channel_estimator: str = "dft_tap_ls",
    estimator_num_taps: int | None = None,
    estimator_ridge_lambda: float = 1.0e-6,
    gradient_clip_norm: float | None = None,
    fading: str = "multipath_block",
) -> dict[str, Any]:
    """Run one Stage-1 fixed-transmitter update.

    Gradients flow through the trainable JSCC encoder, differentiable channel
    arithmetic, estimated-CSI equalization, and JSCC decoder. The frozen codec
    is used only to provide targets before this step and optional diagnostics
    outside the loss.
    """
    batch_size = paired_batch.representation.shape[0]
    device = paired_batch.representation.device
    dtype = paired_batch.representation.dtype
    transmitter_state = torch.zeros(batch_size, model.encoder.channel_state_dim, device=device, dtype=dtype)
    layer_gates = torch.ones(batch_size, model.encoder.num_layers, device=device, dtype=dtype)
    layer_power = torch.ones(model.encoder.num_layers, device=device, dtype=dtype)

    result = run_mode_on_paired_batch(
        codec,
        model,
        paired_batch,
        transmitter_state,
        layer_gates,
        equalizer="estimated",
        fading=fading,
        channel_estimator=channel_estimator,
        estimator_num_taps=estimator_num_taps or paired_batch.metadata.get("estimator_num_taps"),
        estimator_ridge_lambda=estimator_ridge_lambda,
        allocation_mode="uniform",
        resource_reliability=torch.ones_like(paired_batch.noise.real),
        layer_power_allocation=layer_power,
        receiver_state_mode="observable_v1",
        decode_waveform=False,
    )
    loss, per_layer_mse = layer_weighted_latent_mse(
        result["reconstruction"],
        paired_batch.representation,
        layer_weights,
        latent_normalization,
    )
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                list(model.encoder.parameters()) + list(model.decoder.parameters()),
                float(gradient_clip_norm),
            )
        optimizer.step()
    return {
        "loss": loss.detach(),
        "per_layer_mse": per_layer_mse.detach(),
        "reconstruction": result["reconstruction"].detach(),
        "transmitter_state": transmitter_state.detach(),
        "receiver_state": result["decoder_state"].detach(),
        "layer_gates": result["layer_gates"].detach(),
        "layer_power_fractions": result["layer_power_fractions"].detach(),
        "allocation_mode": result["allocation_mode"],
        "fading_model": result.get("fading_model", paired_batch.metadata.get("fading", fading)),
        "channel_estimator": channel_estimator,
        "csi_nmse": result["csi_nmse"].detach(),
        "pilot_evm": result["pilot_evm"].detach(),
        "post_equalization_sinr": result["post_equalization_sinr"].detach(),
        "transmitted": result["transmitted"].detach(),
        "jammer": result["jammer"].detach(),
        "jammer_mask": result["jammer_mask"].detach(),
        "estimated_channel": result["estimated_channel"].detach(),
        "equalized": result["equalized_estimated"].detach(),
    }


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python_random": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def stage1_metadata(
    config: dict[str, Any],
    *,
    representation_shape: tuple[int, int, int],
    layer_weights: list[float],
    representation_source: str,
    layer_importance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    channel = config.get("channel", {})
    codec = config.get("codec", {})
    grid_shape = tuple(config.get("model", {}).get("grid_shape", ()))
    grid_resources = int(torch.tensor(grid_shape).prod()) if grid_shape else sum(
        int(value) for value in getattr(config.get("model", {}), "layer_channel_uses", [])
    )
    pilot_spacing = int(channel.get("pilot_spacing", 4))
    pilot_time_spacing = int(channel.get("pilot_time_spacing", pilot_spacing))
    pilot_resources = (
        ((grid_shape[0] + pilot_spacing - 1) // pilot_spacing)
        * ((grid_shape[1] + pilot_time_spacing - 1) // pilot_time_spacing)
        if len(grid_shape) == 2
        else 0
    )
    data_channel_uses = int(config.get("model", {}).get("channel_uses", 0))
    layers = int(config.get("model", {}).get("layers", len(layer_weights)))
    base, remainder = divmod(data_channel_uses, layers)
    per_layer_uses = [base + (1 if index < remainder else 0) for index in range(layers)]
    return {
        "training_stage": {
            "name": "stage1_fixed_tx",
            "label": STAGE1_LABEL,
        },
        "transmitter_policy": {
            "state_mode": "neutral",
            "gate_mode": "all_ones",
            "channel_use_mode": "fixed_equal",
            "power_mode": "uniform",
            "allocation_mode": "uniform",
        },
        "receiver_policy": {
            "state_mode": "observable_v1",
            "state_dim": 8,
            "state_feature_schema": list(OBSERVABLE_RECEIVER_STATE_FEATURES),
            "uses_true_channel": False,
            "uses_true_jammer_type": False,
            "uses_true_jammer_mask": False,
            "uses_requested_snr_jsr": False,
        },
        "channel_model": {
            "fading": channel.get("fading", "multipath_block"),
            "true_num_taps": int(channel.get("num_taps", 6)),
            "pdp": channel.get("pdp", "exponential"),
            "pdp_decay": float(channel.get("pdp_decay", 0.7)),
            "block_fading_over_time": bool(channel.get("block_fading_over_time", True)),
            "ideal_cp": bool(channel.get("assume_ideal_cp", True)),
            "signal_jammer_channels": "independent",
        },
        "channel_estimator": {
            "name": channel.get("channel_estimator", "dft_tap_ls"),
            "estimator_num_taps": int(channel.get("estimator_num_taps", channel.get("num_taps", 6))),
            "ridge_lambda": float(channel.get("estimator_ridge_lambda", 1.0e-6)),
            "uses_true_channel": False,
        },
        "codec": {
            "type": codec.get("type", "mock"),
            "frozen": bool(codec.get("freeze", True)),
            "sample_rate": codec.get("sample_rate"),
            "waveform_samples": codec.get("waveform_samples"),
            "n_q": codec.get("n_q"),
        },
        "representation_shape": list(representation_shape),
        "layer_loss_weights": list(layer_weights),
        "layer_importance": layer_importance,
        "representation_source": representation_source,
        "resource_mapping": {
            "version": "pilot_reserved_v1",
            "grid_shape": list(grid_shape),
            "grid_total_resources": grid_resources,
            "pilot_resources": pilot_resources,
            "data_channel_uses": data_channel_uses,
            "per_layer_channel_uses": per_layer_uses,
            "packing_order": "row_major_nonpilot",
            "pilot_overwrite_count": 0,
        },
    }


def build_stage1_checkpoint_payload(
    model,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    best_metric: float | None,
    config: dict[str, Any],
    metadata: dict[str, Any],
    scheduler: Any | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": int(step),
        "best_metric": best_metric,
        "config": config,
        "metadata": metadata,
        "rng_state": _rng_state(),
    }
    return payload


def validate_stage1_checkpoint_resources(payload: dict[str, Any], model, config: dict[str, Any]) -> None:
    """Reject legacy or shape-incompatible Stage-1 checkpoints without partial loading."""
    mapping = (payload.get("metadata") or {}).get("resource_mapping")
    if not isinstance(mapping, dict) or mapping.get("version") != "pilot_reserved_v1":
        raise ValueError("legacy Stage-1 checkpoint is incompatible: pilot_reserved_v1 metadata required")
    expected_uses = int(model.encoder.total_channel_uses)
    expected_layers = list(model.encoder.layer_channel_uses)
    if int(mapping.get("data_channel_uses", -1)) != expected_uses:
        raise ValueError("checkpoint data_channel_uses does not match the configured model")
    if list(mapping.get("per_layer_channel_uses", [])) != expected_layers:
        raise ValueError("checkpoint per-layer channel uses do not match the configured model")
    if int(mapping.get("pilot_overwrite_count", -1)) != 0:
        raise ValueError("checkpoint resource mapping overwrites encoder symbols")


__all__ = [
    "STAGE1_LABEL",
    "assert_stage1_startup_invariants",
    "build_stage1_checkpoint_payload",
    "build_stage1_optimizer",
    "stage1_fixed_tx_step",
    "stage1_metadata",
    "validate_stage1_checkpoint_resources",
]
