from __future__ import annotations

import math
from typing import Any


def compare_protocol_values(
    original: Any,
    new: Any,
    *,
    historical_available: bool = True,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    if not historical_available:
        classification = "unknown"
    elif original == new:
        classification = "same"
    elif isinstance(original, (int, float)) and isinstance(new, (int, float)) and math.isclose(
        float(original), float(new), rel_tol=tolerance, abs_tol=tolerance
    ):
        classification = "numerically_equivalent"
    else:
        classification = "different"
    return {"original_o5": original, "new_c1": new, "classification": classification}


def protocol_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    seed = int(config.get("seed", 23))
    train = config.get("train", {})
    model = config.get("model", {})
    channel = config.get("channel", {})
    normalization = train.get("latent_normalization", {})
    fields = [
        ("source_entry_point", "overfit_stage1_path.py", "diagnose_o5_root_cause.py", True),
        ("model_factory", "SpeechJSCC direct", "build_components", True),
        ("model_initialization_seed", seed, seed, True),
        ("latent_source", "RepresentationSource(train).next_batch(1)", "RepresentationSource(train).next_batch(1)", True),
        ("latent_hash", None, None, False),
        ("representation_shape", [8, 50, 1024], [8, 50, 1024], True),
        ("batch_size", 1, 1, True),
        ("optimizer", "Adam", "Adam", True),
        ("learning_rate", train.get("learning_rate"), train.get("learning_rate"), True),
        ("optimizer_reset_behavior", "fresh per ladder stage", "fresh or exact resume", True),
        ("optimization_steps", 500, "requested CLI budget", True),
        ("evaluation_interval", "final only", "step 0/log interval/final", True),
        ("loss_equation", "uniform per-layer power normalized MSE", "uniform per-layer power normalized MSE", True),
        ("layer_weights", [1] * 8, [1] * 8, True),
        ("normalization_epsilon", normalization.get("epsilon"), normalization.get("epsilon"), True),
        ("grid_shape", [64, 32], [64, 32], True),
        ("pilot_mask", "configured comb mask", "configured comb mask", True),
        ("resource_mapping", "pilot_reserved_v1", "pilot_reserved_v1", True),
        ("per_layer_channel_uses", model.get("channel_uses"), model.get("channel_uses"), True),
        ("snr_db", 10.0, 10.0, True),
        ("requested_jsr_db", 0.0, 0.0, True),
        ("jammer_type", "barrage", "full_barrage_estimated_csi", True),
        ("jammer_normalization", "_make_batch convention", "total-grid normalized condition_batch", True),
        ("fixed_batch_seed", 23003, seed + 23000, True),
        ("fixed_realization_policy", "fixed within O5", "fixed with hash assertion", True),
        ("csi_estimator", "dft_tap_ls", "dft_tap_ls", True),
        ("estimator_num_taps", channel.get("estimator_num_taps", 6), channel.get("estimator_num_taps", 6), True),
        ("equalizer", "estimated legitimate CSI", "estimated legitimate CSI", True),
        ("receiver_state", "observable_v1", "observable_v1", True),
        ("transmitter_state", "zeros", "zeros", True),
        ("gates", "all ones", "all ones", True),
        ("allocation", "uniform equal layer power", "uniform equal layer power", True),
        ("decoder_input_ordering", "pilot-reserved deallocation", "pilot-reserved deallocation", True),
        ("model_mode", "default train mode", "default train mode", True),
        ("gradient_clipping", "none", "none", True),
        ("checkpoint_resume", "none", "exact diagnostic resume supported", True),
        ("metric_aggregation", "post-forward loss; final reconstruction before last update", "logged pre-update states including final", True),
        ("final_metric_step_semantics", "500th forward then optimizer update", "step 500 forward; update occurs when extending", True),
    ]
    rows = []
    for field, old, new, available in fields:
        rows.append({"field": field, **compare_protocol_values(old, new, historical_available=available)})
    return rows


def scientific_comparability(rows: list[dict[str, Any]]) -> str:
    batch = next((row for row in rows if row["field"] == "fixed_batch_seed"), None)
    if batch and batch["classification"] == "different":
        return "different realization, not directly comparable"
    if any(row["classification"] in {"different", "unknown"} for row in rows):
        return "partially comparable"
    return "directly comparable"
