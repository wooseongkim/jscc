from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from channels.jammer import compute_jsr, make_jammer
from channels.pilot import (
    csi_nmse,
    equalize_with_csi,
    estimate_channel_ls,
    insert_pilots,
    make_pilot_mask,
    pilot_evm,
    remove_pilot_resources,
)
from channels.rayleigh import compute_effective_sinr, rayleigh_channel
from channels.reliability import compute_resource_reliability
from models.channel_state import (
    CHANNEL_STATE_DIM,
    build_channel_state,
    nominal_channel_state,
    rule_based_jammer_posterior,
)
from models.resource_allocator import allocate_resources, deallocate_resources
from speech_jscc.data import synthetic_waveforms


@dataclass(frozen=True)
class PairedEvaluationBatch:
    """All mode-invariant data and stochastic channel components for one batch."""

    seed: int
    jammer_type: str
    target_power: float
    waveform: Tensor | None
    representation: Tensor
    snr_db: Tensor
    jsr_db: Tensor
    pilot_mask: Tensor
    pilots: Tensor
    jammer: Tensor
    noise: Tensor
    signal_fading: Tensor
    jammer_fading: Tensor
    jammer_mask: Tensor


def _complex_normal(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> Tensor:
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    real = torch.randn(shape, device=device, dtype=real_dtype, generator=generator)
    imag = torch.randn(shape, device=device, dtype=real_dtype, generator=generator)
    return torch.complex(real, imag) / math.sqrt(2.0)


def generate_paired_evaluation_batch(
    codec,
    *,
    batch_size: int,
    waveform_samples: int,
    channel_shape: tuple[int, ...],
    snr_db: float,
    jsr_db: float,
    jammer_type: str,
    jammed_fraction: float,
    pilot_spacing: int,
    pilot_time_spacing: int | None,
    target_power: float,
    seed: int,
    device: torch.device,
    fading: str = "auto",
    waveform: Tensor | None = None,
    representation: Tensor | None = None,
) -> PairedEvaluationBatch:
    """Generate a deterministic batch shared by every adaptation mode."""
    if batch_size <= 0 or target_power <= 0:
        raise ValueError("batch_size and target_power must be positive")
    generator = torch.Generator(device=device).manual_seed(seed)
    if representation is None:
        if waveform is None:
            waveform = synthetic_waveforms(batch_size, waveform_samples, device, generator)
        with torch.no_grad():
            representation = codec.encode_waveform(waveform)
    elif representation.shape[0] != batch_size:
        raise ValueError("provided representation batch size does not match batch_size")

    resource_shape = (batch_size, *channel_shape)
    reference = torch.full(
        resource_shape,
        complex(math.sqrt(target_power), 0.0),
        device=device,
        dtype=torch.complex64,
    )
    pilot_mask = make_pilot_mask(
        resource_shape,
        pilot_spacing,
        time_spacing=pilot_time_spacing,
        device=device,
    )
    _, pilots = insert_pilots(reference, pilot_mask)
    snr_values = torch.full((batch_size,), float(snr_db), device=device)
    jsr_values = torch.full((batch_size,), float(jsr_db), device=device)

    jammer, jammer_mask = make_jammer(
        reference,
        jsr_values,
        jammer_type,
        jammed_fraction,
        pilot_mask=pilot_mask if jammer_type == "pilot" else None,
        pilot_spacing=pilot_spacing,
        generator=generator,
    )
    noise_power = target_power / (10.0 ** (float(snr_db) / 10.0))
    noise = _complex_normal(resource_shape, device, reference.dtype, generator) * math.sqrt(
        noise_power
    )
    realization = rayleigh_channel(
        reference,
        torch.zeros_like(reference),
        snr_values,
        fading=fading,
        noise=torch.zeros_like(reference),
        generator=generator,
    )
    return PairedEvaluationBatch(
        seed=seed,
        jammer_type=jammer_type,
        target_power=target_power,
        waveform=waveform,
        representation=representation,
        snr_db=snr_values,
        jsr_db=jsr_values,
        pilot_mask=pilot_mask,
        pilots=pilots,
        jammer=jammer,
        noise=noise,
        signal_fading=realization["signal_fading"],
        jammer_fading=realization["jammer_fading"],
        jammer_mask=jammer_mask,
    )


def _post_equalization_sinr(
    faded_signal: Tensor,
    faded_jammer: Tensor,
    noise: Tensor,
    channel_for_equalization: Tensor,
) -> Tensor:
    desired = equalize_with_csi(faded_signal, channel_for_equalization)
    interference = equalize_with_csi(faded_jammer, channel_for_equalization)
    equalized_noise = equalize_with_csi(noise, channel_for_equalization)
    return compute_effective_sinr(desired, interference, equalized_noise)


def _state_from_channel(
    batch: PairedEvaluationBatch,
    transmitted: Tensor,
    channel: dict[str, Tensor],
    estimated_channel: Tensor,
    equalizer_channel: Tensor,
) -> Tensor:
    batch_size = transmitted.shape[0]
    posterior = rule_based_jammer_posterior(
        batch.jammer_type,
        batch_size,
        device=transmitted.device,
        dtype=transmitted.real.dtype,
    )
    mask_dimensions = tuple(range(1, batch.jammer_mask.ndim))
    mask_ratio = batch.jammer_mask.to(transmitted.real.dtype).mean(mask_dimensions)
    return build_channel_state(
        _post_equalization_sinr(
            channel["faded_signal"],
            channel["faded_jammer"],
            channel["noise"],
            equalizer_channel,
        ),
        compute_jsr(transmitted, batch.jammer),
        csi_nmse(batch.signal_fading, estimated_channel),
        posterior,
        mask_ratio,
    )


def estimate_transmitter_feedback(
    batch: PairedEvaluationBatch,
    *,
    transmitter_csi: bool = True,
    fading: str = "auto",
) -> dict[str, Tensor]:
    """Measure mode-independent state and resource reliability from shared pilots."""
    mask_dimensions = tuple(range(1, batch.jammer_mask.ndim))
    mask_ratio = batch.jammer_mask.float().mean(mask_dimensions)
    if not transmitter_csi:
        state = nominal_channel_state(
            batch.snr_db,
            batch.jsr_db,
            batch.jammer_type,
            mask_ratio,
        )
        return {"state": state, "reliability": torch.ones_like(batch.noise.real)}
    probe = torch.full_like(batch.noise, complex(math.sqrt(batch.target_power), 0.0))
    transmitted, pilots = insert_pilots(probe, batch.pilot_mask)
    channel = rayleigh_channel(
        transmitted,
        batch.jammer,
        batch.snr_db,
        fading=fading,
        signal_fading=batch.signal_fading,
        jammer_fading=batch.jammer_fading,
        noise=batch.noise,
    )
    estimated_channel = estimate_channel_ls(channel["received"], pilots, batch.pilot_mask)
    state = _state_from_channel(
        batch,
        transmitted,
        channel,
        estimated_channel,
        estimated_channel,
    )
    nmse = csi_nmse(batch.signal_fading, estimated_channel)
    confidence = 1.0 / (1.0 + nmse)
    dimensions = tuple(range(1, batch.noise.ndim))
    noise_power = batch.noise.abs().square().mean(dimensions)
    residual_power = (channel["received"] - estimated_channel * transmitted).abs().square()
    broadcast_noise = noise_power.reshape(
        noise_power.shape[0], *([1] * (batch.noise.ndim - 1))
    )
    estimated_jammer_power = (residual_power - broadcast_noise).clamp_min(0.0)
    reliability = compute_resource_reliability(
        estimated_channel.expand_as(batch.noise),
        estimated_jammer_power,
        noise_power,
        confidence,
    )
    return {
        "state": state,
        "reliability": reliability,
        "estimated_channel": estimated_channel,
    }


def estimate_transmitter_channel_state(
    batch: PairedEvaluationBatch,
    *,
    transmitter_csi: bool = True,
    fading: str = "auto",
) -> Tensor:
    return estimate_transmitter_feedback(
        batch, transmitter_csi=transmitter_csi, fading=fading
    )["state"]


def run_mode_on_paired_batch(
    codec,
    model,
    batch: PairedEvaluationBatch,
    channel_state: Tensor,
    layer_gates: Tensor,
    *,
    equalizer: str = "estimated",
    fading: str = "auto",
    allocation_mode: str = "uniform",
    importance_order: tuple[int, ...] | list[int] | None = None,
    resource_reliability: Tensor | None = None,
) -> dict[str, Tensor]:
    """Evaluate one mode without sampling any new waveform/channel randomness."""
    if equalizer not in {"estimated", "oracle"}:
        raise ValueError("equalizer must be 'estimated' or 'oracle'")
    data_symbols, encoder_aux = model.encoder(
        batch.representation,
        channel_state,
        layer_gates=layer_gates,
        return_aux=True,
    )
    if resource_reliability is None:
        resource_reliability = torch.ones_like(data_symbols.real)
    allocation_generator = torch.Generator(device=data_symbols.device).manual_seed(
        batch.seed + 7_919
    )
    allocation = allocate_resources(
        data_symbols,
        resource_reliability,
        model.encoder.layer_channel_uses,
        mode=allocation_mode,
        importance_order=importance_order,
        pilot_mask=batch.pilot_mask,
        generator=allocation_generator,
    )
    transmitted, pilots = insert_pilots(allocation.symbols, batch.pilot_mask)
    channel = rayleigh_channel(
        transmitted,
        batch.jammer,
        batch.snr_db,
        fading=fading,
        signal_fading=batch.signal_fading,
        jammer_fading=batch.jammer_fading,
        noise=batch.noise,
    )
    estimated_channel = estimate_channel_ls(channel["received"], pilots, batch.pilot_mask)
    equalizer_channel = batch.signal_fading if equalizer == "oracle" else estimated_channel
    equalized = equalize_with_csi(channel["received"], equalizer_channel)
    post_equalization_sinr = _post_equalization_sinr(
        channel["faded_signal"],
        channel["faded_jammer"],
        channel["noise"],
        equalizer_channel,
    )
    received_resources = remove_pilot_resources(equalized, batch.pilot_mask)
    decoder_input = deallocate_resources(received_resources, allocation.resource_to_source)
    receiver_state = channel_state
    if model.decoder.channel_state_dim == CHANNEL_STATE_DIM:
        receiver_state = _state_from_channel(
            batch,
            transmitted,
            channel,
            estimated_channel,
            equalizer_channel,
        )
    reconstruction = model.decoder(decoder_input, receiver_state)
    decoded_waveform = codec.decode_representation(reconstruction)
    return {
        **channel,
        "transmitted": transmitted,
        "data_symbols": data_symbols,
        "jammer": batch.jammer,
        "jammer_mask": batch.jammer_mask,
        "pilot_mask": batch.pilot_mask,
        "pilots": pilots,
        "estimated_channel": estimated_channel,
        "equalized_estimated": equalized,
        "post_equalization_sinr": post_equalization_sinr,
        "decoder_input": decoder_input,
        "allocation_mode": allocation_mode,
        "resource_reliability": resource_reliability,
        "resource_to_source": allocation.resource_to_source,
        "layer_assignment": allocation.layer_assignment,
        "reconstruction": reconstruction,
        "encoder_state": channel_state,
        "decoder_state": receiver_state,
        "decoded_waveform": decoded_waveform,
        "csi_nmse": csi_nmse(batch.signal_fading, estimated_channel),
        "pilot_evm": pilot_evm(
            channel["received"], pilots, batch.pilot_mask, batch.signal_fading
        ),
        "layer_gates": encoder_aux["layer_gates"],
        "layer_power_fractions": encoder_aux["layer_power_fractions"],
    }
