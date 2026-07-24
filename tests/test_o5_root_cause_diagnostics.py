from __future__ import annotations

import torch
import pytest
import inspect
from evaluation.paired import run_mode_on_paired_batch

from speech_jscc.diagnostics.o5_root_cause import (
    CONDITIONS,
    active_resource_jsr_db,
    apply_oracle_subtraction,
    build_condition_mask,
    linear_slope,
    optimal_scale_diagnostics,
    stable_tensor_hash,
    assert_paired_hashes,
    restore_rng_state,
)


def test_condition_matrix_contains_c0_through_c6() -> None:
    assert set(CONDITIONS) == {
        "clean_awgn_reference", "full_barrage_estimated_csi",
        "full_barrage_oracle_csi", "data_only_barrage_estimated_csi",
        "data_only_barrage_oracle_csi", "pilot_only_jammer_estimated_csi",
        "full_barrage_oracle_subtraction",
    }


def test_full_data_and_pilot_masks_have_exact_overlap() -> None:
    pilots = torch.zeros(1, 64, 32, dtype=torch.bool); pilots[:, ::4, ::4] = True
    full = build_condition_mask("full_barrage_estimated_csi", pilots)
    data = build_condition_mask("data_only_barrage_estimated_csi", pilots)
    pilot = build_condition_mask("pilot_only_jammer_estimated_csi", pilots)
    assert full.all() and int((full & pilots).sum()) == 128 and int((full & ~pilots).sum()) == 1920
    assert not (data & pilots).any() and int(data.sum()) == 1920
    assert not (pilot & ~pilots).any() and int(pilot.sum()) == 128


def test_active_resource_jsr_accounts_for_sparse_concentration() -> None:
    signal = torch.ones(1, 4, dtype=torch.complex64)
    jammer = torch.tensor([[2j, 0j, 0j, 0j]])
    mask = jammer != 0
    assert active_resource_jsr_db(signal, jammer, mask).item() == pytest.approx(
        10.0 * torch.log10(torch.tensor(4.0)).item()
    )


def test_oracle_subtraction_removes_exact_faded_jammer() -> None:
    received = torch.randn(2, 3, dtype=torch.complex64)
    jammer = torch.randn(2, 3, dtype=torch.complex64)
    fading = torch.randn(2, 3, dtype=torch.complex64)
    torch.testing.assert_close(apply_oracle_subtraction(received, jammer, fading), received - fading * jammer)
    assert "oracle_subtraction" not in inspect.signature(run_mode_on_paired_batch).parameters
    assert CONDITIONS["full_barrage_oracle_subtraction"]["diagnostic_only_oracle_jammer_subtraction"] is True


def test_optimal_scalar_matches_analytic_example_and_is_diagnostic_only() -> None:
    target = torch.tensor([[[[2.0, 4.0]]]])
    reconstruction = 0.25 * target
    original = reconstruction.clone()
    result = optimal_scale_diagnostics(reconstruction, target, epsilon=1e-6)
    assert abs(result["aggregate"]["a_star"] - 4.0) < 1e-6
    assert result["aggregate"]["rescaled_normalized_mse"] < 1e-10
    assert len(result["per_layer"]) == 1
    torch.testing.assert_close(reconstruction, original)


def test_linear_slope_matches_constructed_sequence() -> None:
    assert abs(linear_slope([1.0, 3.0, 5.0, 7.0]) - 2.0) < 1e-8


def test_tensor_hash_is_stable_and_sensitive() -> None:
    tensor = torch.arange(8)
    assert stable_tensor_hash(tensor) == stable_tensor_hash(tensor.clone())
    assert stable_tensor_hash(tensor) != stable_tensor_hash(tensor + 1)


def test_paired_hash_assertion_rejects_mismatched_common_realization() -> None:
    common={key:"same" for key in ("latent_target","initial_model_parameters","legitimate_channel","awgn","pilot_mask","jammer_channel","raw_jammer_waveform","jammer_mask")}
    hashes={"full_barrage_estimated_csi":dict(common),"full_barrage_oracle_csi":dict(common)}
    assert_paired_hashes(hashes)
    hashes["full_barrage_oracle_csi"]["awgn"]="different"
    with pytest.raises(AssertionError,match="awgn"): assert_paired_hashes(hashes)


def test_restore_rng_state_coerces_torch_state_to_cpu_byte_tensor(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(torch, "set_rng_state", lambda state: captured.setdefault("state", state))
    python_state = __import__("random").getstate()

    restore_rng_state({"torch": torch.arange(8, dtype=torch.int64), "python": python_state})

    assert captured["state"].device.type == "cpu"
    assert captured["state"].dtype == torch.uint8
