from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from evaluate_r4_stratified_allocation import resolve_checkpoint_config_paths, resolve_config_path

from speech_jscc.evaluation.r4_stratified_allocation import (
    COPIES,
    LAYERS,
    SOURCE_SYMBOLS,
    allocation_profile,
    compose_source_power,
    destination_power_from_source_order,
    layer_symbol_indices,
    measured_average_power,
    measured_transmit_symbol_power,
    normalize_layer_power_weights,
    source_layer_indices,
    sqrt_power_amplitude,
    paired_bootstrap_summary,
    reference_check,
    preserve_measured_transmit_power,
    validate_paired_profile_rows,
)


def test_profiles_are_conservative_eight_layer_tiers() -> None:
    assert allocation_profile("uniform").raw_weights == (1.0,) * LAYERS
    assert allocation_profile("core_protection").raw_weights == (1.25,) * 4 + (0.75,) * 4
    assert allocation_profile("layer1_focused").raw_weights == (
        1.10, 1.45, 1.10, 1.10, 0.80, 0.80, 0.80, 0.80
    )
    assert allocation_profile("core_protection").mapping_policy == "balanced_triplet"
    assert allocation_profile("core_protection").repetition_policy == "fixed_three"


def test_source_layer_mapping_is_contiguous_eight_times_240() -> None:
    layers = source_layer_indices()
    assert layers.shape == (SOURCE_SYMBOLS,)
    assert torch.equal(layers, torch.arange(LAYERS).repeat_interleave(240))
    assert torch.equal(layer_symbol_indices(3), torch.arange(720, 960))
    with pytest.raises(ValueError):
        layer_symbol_indices(8)


def test_count_weighted_normalization_preserves_mean_power() -> None:
    weights = normalize_layer_power_weights(
        (1.25, 1.25, 1.25, 1.25, 0.75, 0.75, 0.75, 0.75),
        (100, 100, 100, 100, 200, 200, 200, 200),
    )
    counts = torch.tensor((100, 100, 100, 100, 200, 200, 200, 200), dtype=torch.float64)
    assert torch.dot(counts, weights) / counts.sum() == pytest.approx(1.0)
    assert weights[:4].mean() > weights[4:].mean()


def test_sqrt_amplitude_and_composed_power_preserve_average() -> None:
    normalized = normalize_layer_power_weights(allocation_profile("layer1_focused").raw_weights)
    source_layers = source_layer_indices()
    base = torch.full((COPIES, SOURCE_SYMBOLS), 1.0)
    composed = compose_source_power(base, normalized, source_layers)
    assert torch.allclose(sqrt_power_amplitude(composed).square(), composed)
    assert measured_average_power(composed) == pytest.approx(1.0)
    assert composed[:, 240:480].mean() > composed[:, 960:].mean()


def test_composed_power_preserves_actual_allocator_energy() -> None:
    base = torch.linspace(0.5, 2.0, COPIES * SOURCE_SYMBOLS).reshape(COPIES, SOURCE_SYMBOLS)
    weights = normalize_layer_power_weights(allocation_profile("core_protection").raw_weights)
    composed = compose_source_power(base, weights)
    assert composed.sum() == pytest.approx(base.sum())
    assert torch.isfinite(composed).all()


def test_source_order_power_round_trips_to_destination_order() -> None:
    source_power = torch.arange(COPIES * SOURCE_SYMBOLS, dtype=torch.float32).reshape(COPIES, SOURCE_SYMBOLS)
    resource_to_source = torch.arange(SOURCE_SYMBOLS - 1, -1, -1)
    destination = destination_power_from_source_order(source_power, resource_to_source)
    reconstructed = torch.empty_like(source_power)
    reconstructed[:, resource_to_source] = destination
    assert torch.equal(reconstructed, source_power)


def test_power_calibration_preserves_measured_source_weighted_transmit_power() -> None:
    source = torch.complex(torch.linspace(0.1, 2.0, SOURCE_SYMBOLS)[None], torch.zeros((1, SOURCE_SYMBOLS)))
    base = torch.ones((COPIES, SOURCE_SYMBOLS))
    profile = compose_source_power(base, normalize_layer_power_weights(allocation_profile("layer1_focused").raw_weights))
    calibrated = preserve_measured_transmit_power(source, base, profile)
    assert calibrated.device == source.device
    assert measured_transmit_symbol_power(source, calibrated) == pytest.approx(measured_transmit_symbol_power(source, base))


def test_paired_rows_require_every_primary_profile() -> None:
    rows = [
        {"utterance_id": "u", "realization": 0, "profile": profile}
        for profile in ("uniform", "core_protection", "layer1_focused")
    ]
    validate_paired_profile_rows(rows)
    with pytest.raises(ValueError, match="missing"):
        validate_paired_profile_rows(rows[:-1])


def test_paired_bootstrap_is_deterministic_at_utterance_level() -> None:
    rows = [
        {"utterance_id": utterance, "realization": realization, "profile": profile,
         "si_sdr_db": float(index + realization)}
        for index, utterance in enumerate(("u1", "u2", "u3"))
        for realization in (0, 1)
        for profile in ("uniform", "core_protection", "layer1_focused")
    ]
    first = paired_bootstrap_summary(rows, profile="core_protection", metric="si_sdr_db", samples=100, seed=9)
    second = paired_bootstrap_summary(rows, profile="core_protection", metric="si_sdr_db", samples=100, seed=9)
    assert first == second
    assert first["mean"] == pytest.approx(0.0)
    assert first["positive_gain_fraction"] == 0.0


def test_reference_check_uses_requested_metric_tolerances() -> None:
    passed = reference_check(
        {"si_sdr_db": 0.915, "waveform_snr_db": 3.55, "stft_l1": 0.088},
        {"si_sdr_db": 0.915, "waveform_snr_db": 3.55, "stft_l1": 0.088},
        si_sdr_tolerance=0.02, waveform_snr_tolerance=0.02, stft_tolerance=0.001,
    )
    assert passed["passed"] is True


def test_checked_in_config_declares_only_primary_fixed_power_profiles() -> None:
    config = yaml.safe_load(Path("configs/eval_r4_stratified_allocation.yaml").read_text())
    assert config["allocation_profile"]["mapping_policy"] == "balanced_triplet"
    assert config["allocation_profile"]["repetition_policy"] == "fixed_three"
    assert tuple(config["allocation_profile"]["primary_profiles"]) == (
        "uniform", "core_protection", "layer1_focused"
    )
    assert config["initial_checkpoint"].endswith("best_waveform_si_sdr.pt")


def test_config_relative_paths_resolve_from_config_directory_not_cwd(tmp_path: Path) -> None:
    config = Path("/home/mike/jscc/configs/evaluation.yaml")
    assert resolve_config_path(config, "runs/result.json") == Path("/home/mike/jscc/runs/result.json")
    assert resolve_config_path(config, "/tmp/absolute.json") == Path("/tmp/absolute.json")


def test_checkpoint_codec_paths_resolve_from_repository_root() -> None:
    config = {"codec": {"config_path": "artifacts/config.json", "checkpoint_path": "artifacts/model.pt"}}
    resolve_checkpoint_config_paths(config)
    assert config["codec"]["config_path"] == "/home/mike/jscc/artifacts/config.json"
    assert config["codec"]["checkpoint_path"] == "/home/mike/jscc/artifacts/model.pt"
