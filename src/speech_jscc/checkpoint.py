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

