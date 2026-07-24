from __future__ import annotations

import hashlib
import random
from dataclasses import replace
from typing import Any

import torch
from torch import Tensor
from channels.pilot import csi_nmse, equalize_with_csi, estimate_channel_ls, extract_data_resources, pilot_evm
from models.observable_channel_state import build_observable_receiver_state_v1
from models.resource_allocator import deallocate_resources
from evaluation.paired import run_mode_on_paired_batch


CONDITIONS = {
    "clean_awgn_reference": {"jammer_scope": "none", "equalizer": "estimated"},
    "full_barrage_estimated_csi": {"jammer_scope": "full", "equalizer": "estimated"},
    "full_barrage_oracle_csi": {"jammer_scope": "full", "equalizer": "oracle"},
    "data_only_barrage_estimated_csi": {"jammer_scope": "data", "equalizer": "estimated"},
    "data_only_barrage_oracle_csi": {"jammer_scope": "data", "equalizer": "oracle"},
    "pilot_only_jammer_estimated_csi": {"jammer_scope": "pilot", "equalizer": "estimated"},
    "full_barrage_oracle_subtraction": {
        "jammer_scope": "full",
        "equalizer": "estimated",
        "diagnostic_only_oracle_jammer_subtraction": True,
    },
}


def stable_tensor_hash(tensor: Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    descriptor = f"{value.dtype}|{tuple(value.shape)}|".encode()
    return hashlib.sha256(descriptor + value.numpy().tobytes()).hexdigest()


def build_condition_mask(condition: str, pilot_mask: Tensor) -> Tensor:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown O5 condition: {condition}")
    pilots = pilot_mask.to(torch.bool)
    scope = CONDITIONS[condition]["jammer_scope"]
    if scope == "none":
        return torch.zeros_like(pilots)
    if scope == "full":
        return torch.ones_like(pilots)
    if scope == "data":
        return ~pilots
    return pilots.clone()


def active_resource_jsr_db(signal: Tensor, jammer: Tensor, mask: Tensor, epsilon: float = 1e-12) -> Tensor:
    mask = torch.broadcast_to(mask.to(signal.device, torch.bool), signal.shape)
    values = []
    for index in range(signal.shape[0]):
        signal_power = signal[index][mask[index]].abs().square().mean().clamp_min(epsilon)
        jammer_power = jammer[index][mask[index]].abs().square().mean().clamp_min(epsilon)
        values.append(10.0 * torch.log10(jammer_power / signal_power))
    return torch.stack(values)


def apply_oracle_subtraction(received: Tensor, jammer: Tensor, jammer_fading: Tensor) -> Tensor:
    return received - jammer_fading * jammer


def _scale_row(reconstruction: Tensor, target: Tensor, epsilon: float) -> dict[str, float]:
    r = reconstruction.flatten().float(); t = target.flatten().float()
    target_power = t.square().mean().clamp_min(epsilon)
    a_star = (r * t).sum() / r.square().sum().clamp_min(epsilon)
    scaled = a_star * r
    def nmse(value: Tensor) -> Tensor: return (value - t).square().mean() / target_power
    def ratio(value: Tensor) -> Tensor: return value.square().mean() / target_power
    def corr(value: Tensor) -> Tensor:
        vc=value-value.mean(); tc=t-t.mean(); return (vc*tc).sum()/((vc.norm()*tc.norm()).clamp_min(epsilon))
    return {"a_star": float(a_star), "original_normalized_mse": float(nmse(r)),
            "rescaled_normalized_mse": float(nmse(scaled)), "original_power_ratio": float(ratio(r)),
            "rescaled_power_ratio": float(ratio(scaled)), "original_correlation": float(corr(r)),
            "rescaled_correlation": float(corr(scaled))}


def optimal_scale_diagnostics(
    reconstruction: Tensor,
    target: Tensor,
    epsilon: float,
    layer_weights: Tensor | None = None,
) -> dict[str, Any]:
    if reconstruction.shape != target.shape or target.ndim != 4:
        raise ValueError("reconstruction and target must match [B,L,T,D]")
    per_layer = [
        {"layer": layer, **_scale_row(reconstruction[:, layer], target[:, layer], epsilon)}
        for layer in range(target.shape[1])
    ]
    weights = (
        torch.ones(target.shape[1], dtype=torch.float64)
        if layer_weights is None
        else layer_weights.detach().cpu().to(torch.float64)
    )
    if weights.numel() != target.shape[1] or float(weights.sum()) <= 0:
        raise ValueError("layer_weights must contain one positive-sum value per layer")
    stage1_loss = sum(
        float(weights[index]) * row["rescaled_normalized_mse"]
        for index, row in enumerate(per_layer)
    ) / float(weights.sum())
    aggregate = _scale_row(reconstruction, target, epsilon)
    return {
        "aggregate": aggregate,
        "per_layer": per_layer,
        "global_power_weighted_rescaled_nmse": aggregate["rescaled_normalized_mse"],
        "stage1_layerwise_rescaled_loss": stage1_loss,
    }


def linear_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    y = torch.tensor(values, dtype=torch.float64); x = torch.arange(len(values), dtype=torch.float64)
    return float(((x-x.mean())*(y-y.mean())).sum() / (x-x.mean()).square().sum())


def condition_batch(base_batch, condition: str, requested_jsr_db: float):
    """Create an offline fixed condition without changing production channel code."""
    mask = build_condition_mask(condition, base_batch.pilot_mask)
    if not mask.any():
        jammer = torch.zeros_like(base_batch.jammer)
    else:
        raw = base_batch.jammer * mask
        dimensions = tuple(range(1, raw.ndim))
        signal_power = torch.ones(raw.shape[0], device=raw.device, dtype=raw.real.dtype)
        target_power = signal_power * 10.0 ** (float(requested_jsr_db) / 10.0)
        raw_power = raw.abs().square().mean(dimensions).clamp_min(1e-12)
        scale = torch.sqrt(target_power / raw_power).reshape(raw.shape[0], *([1] * (raw.ndim - 1)))
        jammer = raw * scale
    return replace(base_batch, jammer=jammer, jammer_mask=mask, jammer_type=condition)


def fixed_realization_hashes(batch: Any, target: Tensor, model: torch.nn.Module) -> dict[str, str]:
    parameters = torch.cat([parameter.detach().flatten().cpu() for parameter in model.parameters()])
    return {
        "latent_target": stable_tensor_hash(target),
        "initial_model_parameters": stable_tensor_hash(parameters),
        "legitimate_channel": stable_tensor_hash(batch.signal_fading),
        "jammer_channel": stable_tensor_hash(batch.jammer_fading),
        "awgn": stable_tensor_hash(batch.noise),
        "raw_jammer_waveform": stable_tensor_hash(batch.jammer),
        "jammer_mask": stable_tensor_hash(batch.jammer_mask),
        "pilot_mask": stable_tensor_hash(batch.pilot_mask),
    }


def assert_paired_hashes(condition_hashes: dict[str, dict[str, str]]) -> None:
    common=("latent_target","initial_model_parameters","legitimate_channel","awgn","pilot_mask")
    values=list(condition_hashes.values())
    for key in common:
        if len({item[key] for item in values}) != 1:
            raise AssertionError(f"paired conditions differ for {key}")
    for group in (("full_barrage_estimated_csi","full_barrage_oracle_csi","full_barrage_oracle_subtraction"),
                  ("data_only_barrage_estimated_csi","data_only_barrage_oracle_csi")):
        present=[condition_hashes[name] for name in group if name in condition_hashes]
        for key in ("jammer_channel","raw_jammer_waveform","jammer_mask"):
            if len(present)>1 and len({item[key] for item in present}) != 1:
                raise AssertionError(f"paired jammer conditions differ for {key}")


def run_offline_condition(codec, model, batch, state, gates, condition: str, config: dict[str, Any]):
    """Run production behavior except the explicitly labeled C6 offline subtraction branch."""
    mode=CONDITIONS[condition]
    result=run_mode_on_paired_batch(codec,model,batch,state,gates,equalizer=mode["equalizer"],fading="multipath_block",channel_estimator="dft_tap_ls",estimator_num_taps=config["channel"]["estimator_num_taps"],allocation_mode="uniform",resource_reliability=torch.ones_like(batch.noise.real),receiver_state_mode="observable_v1",decode_waveform=False)
    if not mode.get("diagnostic_only_oracle_jammer_subtraction"):
        return result
    received_cleaned=apply_oracle_subtraction(result["received"],batch.jammer,batch.jammer_fading)
    estimate=estimate_channel_ls(received_cleaned,result["pilots"],batch.pilot_mask,fading="multipath_block",channel_estimator="dft_tap_ls",estimator_num_taps=config["channel"]["estimator_num_taps"],estimator_ridge_lambda=config["channel"].get("estimator_ridge_lambda",1e-6))
    equalized=equalize_with_csi(received_cleaned,estimate)
    resources=extract_data_resources(equalized,batch.pilot_mask)
    decoder_input=deallocate_resources(resources,result["resource_to_source"])
    receiver_state=build_observable_receiver_state_v1(received_cleaned,result["pilots"],batch.pilot_mask,estimate).detach()
    reconstruction=model.decoder(decoder_input,receiver_state)
    cleaned=dict(result); cleaned.update({"received":received_cleaned,"faded_jammer":torch.zeros_like(result["faded_jammer"]),"estimated_channel":estimate,"equalized_estimated":equalized,"decoder_input":decoder_input,"decoder_state":receiver_state,"reconstruction":reconstruction,"csi_nmse":csi_nmse(batch.signal_fading,estimate),"pilot_evm":pilot_evm(received_cleaned,result["pilots"],batch.pilot_mask,estimate),"diagnostic_only_oracle_jammer_subtraction":True})
    return cleaned


def restore_rng_state(rng_state: dict[str, Any]) -> None:
    """Restore RNG state while keeping PyTorch's default-generator state on CPU."""
    torch_state = torch.as_tensor(rng_state["torch"], device="cpu", dtype=torch.uint8)
    torch.set_rng_state(torch_state)
    random.setstate(rng_state["python"])
