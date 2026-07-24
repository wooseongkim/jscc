from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
from torch import Tensor

from channels.pilot import extract_data_resources, insert_data_and_pilots, make_pilot_mask
from models.resource_allocator import allocate_resources, deallocate_resources
from speech_jscc.diagnostics.random_distribution import SeedDeriver
from speech_jscc.models.conv_conformer import balanced_ragged_valid_mask


WIRELESS_DIAGNOSTIC_VERSION = "waveform_aware_wireless_v1"
CF2_SHORT_FRAMES = [2, 7, 12, 17, 22, 27, 32, 37, 42, 47]


def _grid_shape(config: dict[str, Any]) -> tuple[int, int]:
    return tuple(config.get("model", {}).get("grid_shape", (64, 32)))


def _pilot_mask(batch: int, config: dict[str, Any], device: torch.device) -> Tensor:
    channel = config.get("channel", {})
    return make_pilot_mask(
        (batch, *_grid_shape(config)),
        int(channel.get("pilot_spacing", 4)),
        time_spacing=int(channel.get("pilot_time_spacing", 4)),
        device=device,
    )


def validate_cf2_contract(model, config: dict[str, Any]) -> dict[str, Any]:
    encoder = model.encoder
    expected = balanced_ragged_valid_mask(frames=50, max_symbols=5, valid_symbols=240)
    actual = encoder.symbol_valid_mask.detach().cpu().bool()
    pilot = _pilot_mask(1, config, actual.device)
    data = ~pilot
    counts = actual.sum(-1)
    checks = {
        "representation_shape": list(encoder.representation_shape),
        "valid_symbols_per_layer": int(actual.sum()),
        "valid_symbols_total": int(actual.sum()) * 8,
        "four_symbol_frames": torch.where(counts == 4)[0].tolist(),
        "mask_matches_cf2": bool(torch.equal(actual, expected)),
        "uses_temporal_interpolation": bool(encoder.uses_temporal_interpolation),
        "pilot_resources": int(pilot[0].sum()),
        "data_resources": int(data[0].sum()),
        "pilot_data_disjoint": not bool((pilot & data).any()),
        "pilot_data_exhaustive": bool((pilot | data).all()),
    }
    checks["passed"] = bool(
        checks["representation_shape"] == [8, 50, 1024]
        and tuple(encoder.layer_channel_uses) == (240,) * 8
        and encoder.total_channel_uses == 1920
        and encoder.symbol_frames == 50
        and encoder.complex_channels_per_symbol_frame == 5
        and encoder.temporal_symbol_layout == "balanced_ragged"
        and checks["mask_matches_cf2"]
        and not checks["uses_temporal_interpolation"]
        and checks["pilot_resources"] == 128
        and checks["data_resources"] == 1920
        and checks["pilot_data_disjoint"]
        and checks["pilot_data_exhaustive"]
    )
    if not checks["passed"]:
        raise ValueError(f"CF-2 preflight failed: {checks}")
    return checks


def ragged_tensor_diagnostics(fixed_width: Tensor, valid_mask: Tensor) -> dict[str, Any]:
    if fixed_width.shape[-2:] != valid_mask.shape:
        raise ValueError("fixed-width ragged tensor does not match validity mask")
    invalid = fixed_width[..., ~valid_mask]
    invalid_max = float(invalid.abs().max()) if invalid.numel() else 0.0
    if invalid_max != 0.0:
        raise ValueError("masked ragged slots must remain exactly zero")
    valid = fixed_width[..., valid_mask]
    return {
        "fixed_width_shape": list(fixed_width.shape),
        "valid_symbols_per_layer": int(valid_mask.sum()),
        "masked_slot_max_abs": invalid_max,
        "valid_symbol_power": float(valid.abs().square().mean()),
        "passed": True,
    }


def ideal_ofdm_round_trip(
    symbols: Tensor, model, config: dict[str, Any]
) -> dict[str, Any]:
    if symbols.shape[-1] != 1920 or not torch.is_complex(symbols):
        raise ValueError("ideal OFDM requires [B,1920] complex CF-2 symbols")
    allocation = allocate_resources(
        symbols,
        torch.ones_like(symbols.real),
        model.encoder.layer_channel_uses,
        mode="uniform",
    )
    pilot = _pilot_mask(symbols.shape[0], config, symbols.device)
    grid, pilots = insert_data_and_pilots(allocation.symbols, pilot)
    recovered = extract_data_resources(grid, pilot)
    restored = deallocate_resources(recovered, allocation.resource_to_source)
    data = ~pilot
    return {
        "restored": restored,
        "grid": grid,
        "pilots": pilots,
        "pilot_mask": pilot,
        "resource_to_source": allocation.resource_to_source,
        "max_abs_error": float((restored - symbols).abs().max()),
        "pilot_resources": int(pilot[0].sum()),
        "data_resources": int(data[0].sum()),
        "pilot_data_disjoint": not bool((pilot & data).any()),
        "pilot_data_exhaustive": bool((pilot | data).all()),
        "pilot_leakage_into_data": float(
            (grid[data] - allocation.symbols.flatten()).abs().max()
        ),
    }


def clean_validation_conditions(
    seed: int,
    *,
    utterance_count: int,
    realizations_per_utterance: int,
    snr_bins: tuple[float, ...] = (5.0, 10.0, 15.0),
) -> list[dict[str, Any]]:
    derive = SeedDeriver(seed)
    rows: list[dict[str, Any]] = []
    for utterance in range(utterance_count):
        rows.append(
            {
                "channel_policy": "fixed",
                "utterance_index": utterance,
                "realization": 0,
                "snr_db": 10.0,
                "seed": derive.seed("fixed_clean_10db", 0),
            }
        )
        for snr in snr_bins:
            for realization in range(realizations_per_utterance):
                rows.append(
                    {
                        "channel_policy": "random",
                        "utterance_index": utterance,
                        "realization": realization,
                        "snr_db": float(snr),
                        "seed": derive.seed(
                            "random_clean",
                            utterance,
                            f"{realization}|{float(snr)}",
                        ),
                    }
                )
    return rows


def validation_suite_hash(rows: list[dict[str, Any]]) -> str:
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def wireless_feasibility_gate(
    metrics: dict[str, float],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    limits = {
        "delta_si_sdr_min_db": -1.0,
        "delta_waveform_snr_min_db": -1.0,
        "stft_ratio_max": 1.20,
        **(thresholds or {}),
    }
    checks = {
        "delta_si_sdr_pass": float(metrics["delta_si_sdr_db"])
        >= limits["delta_si_sdr_min_db"],
        "delta_waveform_snr_pass": float(metrics["delta_waveform_snr_db"])
        >= limits["delta_waveform_snr_min_db"],
        "stft_ratio_pass": float(metrics["stft_ratio"]) <= limits["stft_ratio_max"],
    }
    return {"passed": all(checks.values()), "checks": checks, "thresholds": limits}


def ideal_equivalence_gate(
    direct: dict[str, float], ideal: dict[str, float]
) -> dict[str, Any]:
    values = {
        "si_sdr_difference_db": abs(
            float(ideal["si_sdr_db"]) - float(direct["si_sdr_db"])
        ),
        "waveform_snr_difference_db": abs(
            float(ideal["waveform_snr_db"]) - float(direct["waveform_snr_db"])
        ),
        "stft_ratio_to_direct": float(ideal["stft_l1"])
        / max(float(direct["stft_l1"]), 1e-12),
        "summed_latent_nmse_increase": float(ideal["summed_latent_nmse"])
        - float(direct["summed_latent_nmse"]),
    }
    checks = {
        "si_sdr": values["si_sdr_difference_db"] <= 0.05,
        "waveform_snr": values["waveform_snr_difference_db"] <= 0.05,
        "stft": values["stft_ratio_to_direct"] <= 1.01,
        "summed_latent": values["summed_latent_nmse_increase"] <= 1e-6,
    }
    return {"passed": all(checks.values()), "checks": checks, **values}
