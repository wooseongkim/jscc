from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LayerImportance:
    layer_weights_mean_one: list[float]
    layer_weights_sum_one: list[float]
    layer_importance_order: list[int]
    base_layers: list[int]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ResolvedLayerImportance:
    layer_weights: list[float] | None
    layer_importance_order: list[int] | None
    base_layers: list[int] | None
    artifact_path: str | None
    artifact_hash: str | None
    artifact: LayerImportance | None


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            import json

            payload = json.load(handle)
        else:
            payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping")
    return payload


def _warn_or_raise(message: str, strict: bool) -> None:
    if strict:
        raise ValueError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _float_list(values: Any, name: str) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"importance.{name} must be a nonempty list")
    converted = [float(value) for value in values]
    if any(value < 0.0 or value != value or value in {float("inf"), float("-inf")} for value in converted):
        raise ValueError(f"importance.{name} must contain finite nonnegative values")
    if sum(converted) <= 0.0:
        raise ValueError(f"importance.{name} must have positive sum")
    return converted


def _int_list(values: Any, name: str) -> list[int]:
    if not isinstance(values, list):
        raise ValueError(f"importance.{name} must be a list")
    return [int(value) for value in values]


def load_layer_importance(
    path: str | Path,
    *,
    expected_codec_type: str | None = None,
    expected_representation_shape: tuple[int, int, int] | list[int] | None = None,
    strict: bool = True,
) -> LayerImportance:
    artifact_path = Path(path)
    payload = _load_payload(artifact_path)
    if int(payload.get("schema_version", -1)) != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported layer importance schema_version={payload.get('schema_version')!r}"
        )
    codec = payload.get("codec")
    importance = payload.get("importance")
    if not isinstance(codec, dict) or not isinstance(importance, dict):
        raise ValueError("layer importance artifact requires codec and importance mappings")
    codec_type = str(codec.get("type", ""))
    representation_shape = tuple(int(value) for value in codec.get("representation_shape", []))
    if len(representation_shape) != 3 or min(representation_shape) <= 0:
        raise ValueError("codec.representation_shape must be [L,T,D]")
    if expected_codec_type is not None and codec_type != expected_codec_type:
        _warn_or_raise(
            f"codec type mismatch: artifact={codec_type!r}, expected={expected_codec_type!r}",
            strict,
        )
    if expected_representation_shape is not None:
        expected = tuple(int(value) for value in expected_representation_shape)
        if representation_shape != expected:
            _warn_or_raise(
                f"representation_shape mismatch: artifact={representation_shape}, expected={expected}",
                strict,
            )

    layers = representation_shape[0]
    mean_one = _float_list(importance.get("layer_weights_mean_one"), "layer_weights_mean_one")
    sum_one = _float_list(importance.get("layer_weights_sum_one"), "layer_weights_sum_one")
    order = _int_list(importance.get("layer_importance_order"), "layer_importance_order")
    base_layers = _int_list(importance.get("base_layers", []), "base_layers")
    if len(mean_one) != layers or len(sum_one) != layers:
        raise ValueError("layer weight length must match codec.representation_shape[0]")
    if sorted(order) != list(range(layers)):
        raise ValueError("importance.layer_importance_order must be a valid permutation")
    if any(layer < 0 or layer >= layers for layer in base_layers):
        raise ValueError("importance.base_layers contains invalid layer indices")
    metadata = dict(payload)
    metadata["path"] = str(artifact_path)
    metadata["artifact_hash"] = file_sha256(artifact_path)
    return LayerImportance(
        layer_weights_mean_one=mean_one,
        layer_weights_sum_one=sum_one,
        layer_importance_order=order,
        base_layers=base_layers,
        metadata=metadata,
    )


def resolve_layer_importance_config(
    config: dict[str, Any],
    *,
    section: str,
    expected_representation_shape: tuple[int, int, int] | None = None,
) -> ResolvedLayerImportance:
    options = config.get("layer_importance") or {}
    if not options.get("path"):
        section_cfg = config.get(section, {})
        return ResolvedLayerImportance(
            layer_weights=section_cfg.get("layer_weights"),
            layer_importance_order=section_cfg.get("layer_importance_order"),
            base_layers=section_cfg.get("base_layers"),
            artifact_path=None,
            artifact_hash=None,
            artifact=None,
        )
    model_cfg = config.get("model", {})
    expected_shape = expected_representation_shape or (
        int(model_cfg["layers"]),
        int(model_cfg["frames"]),
        int(model_cfg["latent_dim"]),
    )
    artifact = load_layer_importance(
        options["path"],
        expected_codec_type=config.get("codec", {}).get("type"),
        expected_representation_shape=expected_shape,
        strict=bool(options.get("strict_metadata", True)),
    )
    section_cfg = config.get(section, {})
    weights = (
        artifact.layer_weights_mean_one
        if bool(options.get("apply_to_loss_weights", False))
        else section_cfg.get("layer_weights")
    )
    order = (
        artifact.layer_importance_order
        if bool(options.get("apply_to_resource_order", False))
        else section_cfg.get("layer_importance_order")
    )
    base_layers = (
        artifact.base_layers
        if bool(options.get("apply_to_base_layers", False))
        else section_cfg.get("base_layers")
    )
    return ResolvedLayerImportance(
        layer_weights=weights,
        layer_importance_order=order,
        base_layers=base_layers,
        artifact_path=str(options["path"]),
        artifact_hash=artifact.metadata["artifact_hash"],
        artifact=artifact,
    )


__all__ = [
    "LayerImportance",
    "ResolvedLayerImportance",
    "file_sha256",
    "load_layer_importance",
    "resolve_layer_importance_config",
]
