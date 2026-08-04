"""Layer replacement and importance statistics for the R4 path."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def replace_layers(representation, layers: Iterable[int], replacement: str = "zero"):
    """Return a copy with only selected layer indices replaced."""
    import torch

    if representation.ndim != 4:
        raise ValueError("representation must have shape [B,L,T,D]")
    indices = sorted(set(int(i) for i in layers))
    if any(i < 0 or i >= representation.shape[1] for i in indices):
        raise ValueError("layer index out of range")
    output = representation.clone()
    if replacement == "zero":
        value = None
    elif replacement == "mean":
        value = representation.mean(dim=(0, 2, 3), keepdim=True)
    else:
        raise ValueError("replacement must be zero or mean")
    for index in indices:
        output[:, index] = 0.0 if value is None else value[:, 0]
    return output


def layer_replacement_metrics(codec, reference_waveform, representation, layers, sample_rate, replacement="zero"):
    from src.evaluation.waveform_metrics import waveform_metrics

    modified = replace_layers(representation, layers, replacement)
    waveform = codec.decode_representation(modified)
    return waveform_metrics(reference_waveform, waveform, sample_rate)


def distribution(values: list[float], *, bootstrap_samples=2000, bootstrap_seed=91027):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("distribution requires finite non-empty values")
    rng = np.random.default_rng(int(bootstrap_seed))
    sample = rng.choice(values, size=(int(bootstrap_samples), values.size), replace=True)
    means = sample.mean(axis=1)
    return {
        "sample_count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1 if values.size > 1 else 0)),
        "standard_error": float(values.std(ddof=1 if values.size > 1 else 0) / math.sqrt(values.size)),
        "p10": float(np.percentile(values, 10)), "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)), "p90": float(np.percentile(values, 90)),
        "min": float(values.min()), "max": float(values.max()),
        "positive_gain_fraction": float(np.mean(values > 0)),
        "bootstrap_ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "bootstrap_samples": int(bootstrap_samples), "bootstrap_seed": int(bootstrap_seed),
    }


def pearson_spearman(x: list[float], y: list[float]) -> dict:
    from scipy.stats import pearsonr, spearmanr

    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return {"pearson": None, "spearman": None, "sample_count": len(x)}
    return {"pearson": float(pearsonr(x, y).statistic), "spearman": float(spearmanr(x, y).statistic), "sample_count": len(x)}


def normalized_weights(values: list[float], mode: str = "sum_one") -> list[float]:
    if not values or any(not math.isfinite(float(v)) for v in values):
        raise ValueError("weights must be finite and non-empty")
    total = sum(float(v) for v in values)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    weights = [float(v) / total for v in values]
    if mode == "sum_one":
        return weights
    if mode == "mean_one":
        return [v * len(values) for v in weights]
    raise ValueError("mode must be sum_one or mean_one")
