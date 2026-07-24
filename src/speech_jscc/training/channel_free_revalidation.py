from __future__ import annotations

import math
import copy
from collections.abc import Callable, Iterable

import torch
from torch import Tensor, nn

from speech_jscc.training.channel_free_feasibility import (
    multi_resolution_stft_loss,
    negative_si_sdr_loss,
)
from speech_jscc.models.conv_conformer import balanced_ragged_valid_mask


CHANNEL_FREE_EXPERIMENTS = {
    "cf1": {
        "name": "cf1_30frames_1920",
        "symbol_frames": 30,
        "channel_uses": 1920,
        "complex_channels_per_symbol_frame": 8,
        "temporal_symbol_layout": "dense_interpolate",
        "d_model": 256,
        "encoder_conformer_blocks": 4,
        "decoder_conformer_blocks": 4,
        "minimum_steps": 20_000,
    },
    "cf2": {
        "name": "cf2_50frames_1920",
        "symbol_frames": 50,
        "channel_uses": 1920,
        "complex_channels_per_symbol_frame": 5,
        "temporal_symbol_layout": "balanced_ragged",
        "d_model": 256,
        "encoder_conformer_blocks": 4,
        "decoder_conformer_blocks": 4,
        "minimum_steps": 20_000,
    },
    "cf3": {
        "name": "cf3_50frames_3200",
        "symbol_frames": 50,
        "channel_uses": 3200,
        "complex_channels_per_symbol_frame": 8,
        "temporal_symbol_layout": "dense_interpolate",
        "d_model": 256,
        "encoder_conformer_blocks": 4,
        "decoder_conformer_blocks": 4,
        "minimum_steps": 20_000,
    },
    "cf4": {
        "name": "cf4_large_model",
        "symbol_frames": 50,
        "channel_uses": 3200,
        "complex_channels_per_symbol_frame": 8,
        "temporal_symbol_layout": "dense_interpolate",
        "d_model": 384,
        "encoder_conformer_blocks": 6,
        "decoder_conformer_blocks": 6,
        "minimum_steps": 20_000,
    },
    "cf5": {
        "name": "cf5_long_training",
        "source": "best_cf1_to_cf4",
        "minimum_steps": 32_000,
    },
}


def apply_experiment_definition(config: dict, experiment: str) -> dict:
    if experiment not in CHANNEL_FREE_EXPERIMENTS or experiment == "cf5":
        raise ValueError("experiment must be a concrete CF-1 through CF-4 definition")
    output = copy.deepcopy(config)
    definition = CHANNEL_FREE_EXPERIMENTS[experiment]
    model = output["model"]
    for key in (
        "symbol_frames", "channel_uses", "complex_channels_per_symbol_frame",
        "temporal_symbol_layout", "d_model", "encoder_conformer_blocks",
        "decoder_conformer_blocks",
    ):
        model[key] = definition[key]
    if definition["temporal_symbol_layout"] == "balanced_ragged":
        mask = balanced_ragged_valid_mask(
            frames=definition["symbol_frames"],
            max_symbols=definition["complex_channels_per_symbol_frame"],
            valid_symbols=definition["channel_uses"] // 8,
        )
        counts = mask.sum(-1)
        model["temporal_symbol_pattern"] = {
            "frame_valid_symbol_counts": [int(value) for value in counts],
            "four_symbol_frames": torch.where(counts == 4)[0].tolist(),
            "valid_symbols_per_layer": int(mask.sum()),
            "valid_symbols_total": int(mask.sum()) * 8,
            "max_symbols_per_frame": int(mask.shape[1]),
        }
    return output


def feasibility_classification(delta_si_sdr: float, delta_waveform_snr: float,
                               stft_ratio: float) -> str:
    if delta_si_sdr >= -1.0 and delta_waveform_snr >= -1.0 and stft_ratio <= 1.20:
        return "CHANNEL_FREE_FEASIBLE"
    return "CHANNEL_FREE_CONV_CONFORMER_NOT_YET_FEASIBLE"


def checkpoint_filenames() -> dict[str, str]:
    return {
        "per_layer_nmse": "best_per_layer_nmse.pt",
        "summed_latent_nmse": "best_summed_latent_nmse.pt",
        "waveform_si_sdr": "best_waveform_si_sdr.pt",
    }


def curriculum_weights(step: int, *, stage1_steps: int = 4000,
                       stage2_steps: int = 8000,
                       lambda_layer: float = 1.0,
                       lambda_sum: float = 1.0,
                       lambda_stft: float = 0.01,
                       lambda_sisdr: float = 0.001) -> dict[str, float]:
    if step <= 0:
        raise ValueError("step must be positive")
    return {
        "layer": float(lambda_layer),
        "sum": float(lambda_sum),
        "stft": float(lambda_stft if step > stage1_steps else 0.0),
        "sisdr": float(lambda_sisdr if step > stage1_steps + stage2_steps else 0.0),
    }


def summed_decoder_input(layers: Tensor) -> Tensor:
    if layers.ndim != 4:
        raise ValueError("layers must have shape [B,L,T,D]")
    return layers.sum(dim=1)


def _normalized_mse(reconstruction: Tensor, target: Tensor,
                    epsilon: float = 1e-8) -> Tensor:
    return (reconstruction - target).square().mean() / target.square().mean().clamp_min(epsilon)


def per_layer_nmse(reconstruction: Tensor, target: Tensor,
                   epsilon: float = 1e-8) -> Tensor:
    if reconstruction.shape != target.shape or reconstruction.ndim != 4:
        raise ValueError("reconstruction and target must match [B,L,T,D]")
    error = (reconstruction - target).square().mean(dim=(0, 2, 3))
    power = target.square().mean(dim=(0, 2, 3)).clamp_min(epsilon)
    return error / power


def framewise_summed_nmse(reconstruction: Tensor, target: Tensor,
                          epsilon: float = 1e-8) -> Tensor:
    a, b = summed_decoder_input(reconstruction), summed_decoder_input(target)
    return (a - b).square().mean(dim=(0, 2)) / b.square().mean(dim=(0, 2)).clamp_min(epsilon)


def summed_latent_statistics(reconstruction: Tensor, target: Tensor,
                             epsilon: float = 1e-8) -> dict[str, Tensor]:
    a, b = summed_decoder_input(reconstruction), summed_decoder_input(target)
    nmse = _normalized_mse(a, b, epsilon)
    error_power = (a - b).square().mean().clamp_min(epsilon)
    target_power = b.square().mean().clamp_min(epsilon)
    centered_a, centered_b = a - a.mean(), b - b.mean()
    correlation = (centered_a * centered_b).sum() / (
        centered_a.square().sum().sqrt() * centered_b.square().sum().sqrt()
    ).clamp_min(epsilon)
    return {
        "nmse": nmse,
        "snr_db": 10 * torch.log10(target_power / error_power),
        "correlation": correlation,
        "power_ratio": a.square().mean() / target_power,
    }


def waveform_connected_objective(
    reconstruction: Tensor,
    target: Tensor,
    waveform_target: Tensor,
    decode_layers: Callable[[Tensor], Tensor],
    *,
    weights: dict[str, float],
    fft_sizes: tuple[int, ...] = (256, 512, 1024),
) -> tuple[Tensor, dict[str, Tensor]]:
    layer = per_layer_nmse(reconstruction, target).mean()
    summed = summed_latent_statistics(reconstruction, target)["nmse"]
    if float(weights["stft"]) or float(weights["sisdr"]):
        decoded = decode_layers(reconstruction)
        stft = multi_resolution_stft_loss(decoded, waveform_target, fft_sizes=fft_sizes)
        sisdr = negative_si_sdr_loss(decoded, waveform_target)
    else:
        stft = layer.new_zeros(())
        sisdr = layer.new_zeros(())
    components = {"layer": layer, "sum": summed, "stft": stft, "sisdr": sisdr}
    total = sum(float(weights[name]) * value for name, value in components.items())
    return total, components


def component_gradient_norms(
    components: dict[str, Tensor],
    weights: dict[str, float],
    parameters: Iterable[nn.Parameter],
) -> dict[str, float]:
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    result: dict[str, float] = {}
    for name, value in components.items():
        weight = float(weights[name])
        if weight == 0:
            result[name] = 0.0
            continue
        gradients = torch.autograd.grad(
            weight * value, trainable, retain_graph=True, allow_unused=True
        )
        squared = sum(
            gradient.detach().float().square().sum()
            for gradient in gradients if gradient is not None
        )
        result[name] = math.sqrt(float(squared))
    return result
