from __future__ import annotations

import torch

from evaluation.paired import (
    estimate_transmitter_channel_state,
    generate_paired_evaluation_batch,
    run_mode_on_paired_batch,
)
from speech_jscc.codecs import MockContinuousCodec
from speech_jscc.models import SpeechJSCC


def build_components(config: dict, device: torch.device):
    model_cfg = config["model"]
    shape = (model_cfg["layers"], model_cfg["frames"], model_cfg["latent_dim"])
    codec = MockContinuousCodec(*shape, config["codec"]["waveform_samples"]).to(device)
    model = SpeechJSCC(
        shape,
        model_cfg["channel_uses"],
        model_cfg["channel_state_dim"],
        model_cfg["hidden_dim"],
        model_cfg["target_power"],
    ).to(device)
    return codec, model


def sample_uniform(batch: int, bounds: list[float], device: torch.device) -> torch.Tensor:
    return torch.empty(batch, device=device).uniform_(float(bounds[0]), float(bounds[-1]))


def run_batch(codec, model, config: dict, batch_size: int, device: torch.device,
              snr_db=None, jsr_db=None, jammer_kind=None):
    channel_cfg = config["channel"]
    snr_bounds = channel_cfg.get("snr_db", channel_cfg.get("snr_db_range"))
    jsr_bounds = channel_cfg.get("jsr_db", channel_cfg.get("jsr_db_range"))
    snr_value = float(snr_db) if snr_db is not None else float(sample_uniform(1, snr_bounds, device).item())
    jsr_value = float(jsr_db) if jsr_db is not None else float(sample_uniform(1, jsr_bounds, device).item())
    kinds = channel_cfg.get("jammer_types") or list(channel_cfg["jammer_probabilities"])
    kind = jammer_kind or kinds[0]
    channel_shape = tuple(model.encoder.channel_shape)
    fading = "flat" if len(channel_shape) == 1 else "ofdm"
    paired = generate_paired_evaluation_batch(
        codec,
        batch_size=batch_size,
        waveform_samples=config["codec"]["waveform_samples"],
        channel_shape=channel_shape,
        snr_db=snr_value,
        jsr_db=jsr_value,
        jammer_type=kind,
        jammed_fraction=channel_cfg["jammed_fraction"],
        pilot_spacing=channel_cfg.get("pilot_spacing", 4),
        pilot_time_spacing=channel_cfg.get("pilot_time_spacing"),
        target_power=config["model"]["target_power"],
        seed=config["seed"],
        device=device,
        fading=fading,
    )
    channel_state = estimate_transmitter_channel_state(paired, fading=fading)
    gates = channel_state.new_ones((batch_size, config["model"]["layers"]))
    result = run_mode_on_paired_batch(
        codec, model, paired, channel_state, gates, equalizer="estimated", fading=fading
    )
    return {
        "waveform": paired.waveform,
        "representation": paired.representation,
        "reconstructed": result["reconstruction"],
        **result,
    }
