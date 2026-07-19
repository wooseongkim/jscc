from __future__ import annotations

import torch

from evaluation.paired import (
    estimate_transmitter_channel_state,
    generate_paired_evaluation_batch,
    run_mode_on_paired_batch,
)
from speech_jscc.codecs import MockContinuousCodec, SpeechTokenizerWrapper
from speech_jscc.models import ConvConformerJSCC, SpeechJSCC


def build_components(config: dict, device: torch.device):
    model_cfg = config["model"]
    codec_cfg = config["codec"]
    codec_type = codec_cfg.get("type", "mock").lower()
    if codec_type == "mock":
        configured_shape = (model_cfg["layers"], model_cfg["frames"], model_cfg["latent_dim"])
        codec = MockContinuousCodec(
            *configured_shape, codec_cfg["waveform_samples"], seed=codec_cfg.get("seed", 0)
        ).to(device)
    elif codec_type == "speechtokenizer":
        codec = SpeechTokenizerWrapper(
            config_path=codec_cfg["config_path"],
            checkpoint_path=codec_cfg["checkpoint_path"],
            waveform_samples=codec_cfg["waveform_samples"],
            n_q=codec_cfg.get("n_q"),
            fallback_to_mock=False,
            freeze=codec_cfg.get("freeze", True),
        ).to(device)
        if not codec_cfg.get("freeze", True):
            raise ValueError("SpeechTokenizer fine-tuning is not supported; set codec.freeze: true")
    else:
        raise ValueError(f"unsupported codec type: {codec_type}")
    shape = codec.representation_shape
    model_cfg["layers"], model_cfg["frames"], model_cfg["latent_dim"] = shape
    architecture = model_cfg.get("architecture", "flat_mlp")
    if architecture in {"flat_mlp", "normalized_flat_mlp"}:
        model = SpeechJSCC(shape, model_cfg["channel_uses"], model_cfg["channel_state_dim"],
                           model_cfg["hidden_dim"], model_cfg["target_power"])
    elif architecture == "conv_conformer_v1":
        keys = ("d_model", "encoder_conformer_blocks", "decoder_conformer_blocks", "num_attention_heads",
                "ffn_expansion", "convolution_kernel_size", "dropout", "layer_mixer_blocks",
                "symbol_frames", "complex_channels_per_symbol_frame")
        model = ConvConformerJSCC(shape, model_cfg["channel_uses"], model_cfg["channel_state_dim"],
                                  model_cfg["target_power"], **{key:model_cfg[key] for key in keys if key in model_cfg})
    else:
        raise ValueError(f"unsupported model architecture: {architecture}")
    model = model.to(device)
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
    fading = channel_cfg.get("fading", "flat" if len(channel_shape) == 1 else "ofdm")
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
        num_taps=channel_cfg.get("num_taps", 6),
        pdp_decay=channel_cfg.get("pdp_decay", 0.7),
        channel_estimator=channel_cfg.get("channel_estimator", "auto"),
        estimator_num_taps=channel_cfg.get("estimator_num_taps"),
        estimator_ridge_lambda=channel_cfg.get("estimator_ridge_lambda", 1e-6),
    )
    channel_state = estimate_transmitter_channel_state(
        paired,
        fading=fading,
        channel_estimator=channel_cfg.get("channel_estimator", "auto"),
        estimator_num_taps=channel_cfg.get("estimator_num_taps"),
        estimator_ridge_lambda=channel_cfg.get("estimator_ridge_lambda", 1e-6),
    )
    gates = channel_state.new_ones((batch_size, config["model"]["layers"]))
    result = run_mode_on_paired_batch(
        codec,
        model,
        paired,
        channel_state,
        gates,
        equalizer="estimated",
        fading=fading,
        channel_estimator=channel_cfg.get("channel_estimator", "auto"),
        estimator_num_taps=channel_cfg.get("estimator_num_taps"),
        estimator_ridge_lambda=channel_cfg.get("estimator_ridge_lambda", 1e-6),
    )
    return {
        "waveform": paired.waveform,
        "representation": paired.representation,
        "reconstructed": result["reconstruction"],
        **result,
    }
