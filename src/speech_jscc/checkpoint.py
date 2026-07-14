from __future__ import annotations

from typing import Any


def codec_name(config: dict[str, Any]) -> str:
    value = config.get("codec", {}).get("type", "mock").lower()
    return "speechtokenizer" if value == "speechtokenizer" else "mock_continuous"


def normalization_config(config: dict[str, Any], section: str = "train") -> dict[str, Any]:
    value = config.get(section, {}).get("latent_normalization", {"mode": "none"})
    if isinstance(value, str):
        value = {"mode": value}
    result = dict(value)
    result.setdefault("mode", "none")
    result.setdefault("epsilon", 1e-8)
    return result


def channel_model_metadata(config: dict[str, Any]) -> dict[str, Any]:
    channel = config.get("channel", {})
    model = config.get("model", {})
    channel_uses = model.get("channel_uses")
    inferred = "flat"
    if isinstance(channel_uses, (list, tuple)) and len(channel_uses) == 2:
        inferred = "ofdm"
    fading = channel.get("fading", inferred)
    return {
        "fading": fading,
        "num_taps": int(channel.get("num_taps", 6)),
        "pdp": channel.get("pdp", "exponential"),
        "pdp_decay": float(channel.get("pdp_decay", 0.7)),
        "block_fading_over_time": bool(
            channel.get("block_fading_over_time", fading == "multipath_block")
        ),
        "assume_ideal_cp": bool(channel.get("assume_ideal_cp", fading == "multipath_block")),
        "channel_estimator": channel.get(
            "channel_estimator",
            "block_frequency_ls" if fading == "multipath_block" else "auto",
        ),
        "estimator_num_taps": int(channel.get("estimator_num_taps", channel.get("num_taps", 6))),
        "estimator_ridge_lambda": float(channel.get("estimator_ridge_lambda", 1e-6)),
        "channel_estimator_metadata": {
            "name": channel.get(
                "channel_estimator",
                "block_frequency_ls" if fading == "multipath_block" else "auto",
            ),
            "estimator_num_taps": int(
                channel.get("estimator_num_taps", channel.get("num_taps", 6))
            ),
            "ridge_lambda": float(channel.get("estimator_ridge_lambda", 1e-6)),
            "pilot_time_averaging": True,
            "reconstruction_domain": (
                "tap" if channel.get("channel_estimator") == "dft_tap_ls" else "frequency"
            ),
            "dft_convention": "torch_fft_forward",
            "uses_true_channel": False,
        },
        "snr_reference": "transmit_power",
        "jammer_channel": (
            "independent_multipath" if fading == "multipath_block" else "independent_rayleigh"
        ),
    }


def build_checkpoint_metadata(
    config: dict[str, Any],
    codec,
    *,
    representation_source: str,
) -> dict[str, Any]:
    name = codec_name(config)
    layers, frames, latent_dim = codec.representation_shape
    sample_rate = getattr(codec, "sample_rate", config.get("codec", {}).get("sample_rate"))
    frame_rate = getattr(codec, "frame_rate", None)
    if frame_rate is None and sample_rate:
        duration = config["codec"]["waveform_samples"] / float(sample_rate)
        frame_rate = frames / duration
    trained_from_waveforms = representation_source.startswith("waveform_corpus")
    speech_performance_valid = name == "speechtokenizer" and trained_from_waveforms
    return {
        "format_version": 1,
        "checkpoint_kind": (
            "speechtokenizer_latent_jscc" if name == "speechtokenizer" else "mock_continuous_jscc"
        ),
        "codec_name": name,
        "codec_frozen": bool(config.get("codec", {}).get("freeze", name == "speechtokenizer")),
        "layers": int(layers),
        "frames": int(frames),
        "latent_dim": int(latent_dim),
        "sample_rate": int(sample_rate) if sample_rate is not None else None,
        "frame_rate": float(frame_rate) if frame_rate is not None else None,
        "normalization": normalization_config(config),
        "channel_model": channel_model_metadata(config),
        "representation_source": representation_source,
        "trained_on_codec_latents": True,
        "speech_tokenizer_metric_valid": speech_performance_valid,
    }


def validate_checkpoint_metadata(metadata: dict[str, Any], config: dict[str, Any], codec) -> None:
    expected_name = codec_name(config)
    if metadata.get("codec_name") != expected_name:
        raise ValueError(
            f"checkpoint codec {metadata.get('codec_name')!r} does not match configured codec "
            f"{expected_name!r}"
        )
    expected_shape = tuple(codec.representation_shape)
    actual_shape = tuple(metadata.get(key) for key in ("layers", "frames", "latent_dim"))
    if actual_shape != expected_shape:
        raise ValueError(
            f"checkpoint latent shape {actual_shape} does not match codec shape {expected_shape}"
        )
    requested_channel = channel_model_metadata(config)
    trained_channel = metadata.get("channel_model")
    strict_channel = bool(
        config.get("eval", {}).get("strict_channel_model", False)
        or config.get("train", {}).get("strict_channel_model", False)
    )
    allow_cross = bool(config.get("eval", {}).get("allow_cross_channel_evaluation", False))
    if strict_channel and not allow_cross:
        if not isinstance(trained_channel, dict):
            raise ValueError("checkpoint has no channel_model metadata for strict channel matching")
        keys = (
            "fading",
            "num_taps",
            "pdp",
            "pdp_decay",
            "block_fading_over_time",
            "assume_ideal_cp",
            "channel_estimator",
        )
        mismatches = [
            key
            for key in keys
            if trained_channel.get(key) != requested_channel.get(key)
        ]
        if mismatches:
            raise ValueError(
                "checkpoint channel_model does not match requested channel model: "
                + ", ".join(mismatches)
            )
