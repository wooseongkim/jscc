from __future__ import annotations

from speech_jscc.diagnostics.overfit import classify_overfit_result, stages_to_run
from channels.pilot import extract_data_resources, insert_data_and_pilots, make_pilot_mask
import torch


def _result(loss: float, improvement: float, power: float = 0.5, cosine: float = 0.2, corr: float = 0.1):
    return {"final_loss": loss, "relative_improvement_over_zero": improvement,
            "power_ratio": power, "cosine_similarity": cosine, "pearson_correlation": corr}


def test_o0_to_o2_use_loss_or_improvement_plus_alignment_guards() -> None:
    assert classify_overfit_result("O2", _result(0.19, 0.1))[0]
    assert classify_overfit_result("O2", _result(0.8, 0.81))[0]
    assert not classify_overfit_result("O2", _result(0.19, 0.9, power=0.0))[0]
    assert not classify_overfit_result("O2", _result(0.19, 0.9, corr=0.0))[0]


def test_o3_to_o5_use_fifty_percent_or_half_loss() -> None:
    assert classify_overfit_result("O3", _result(0.49, 0.1))[0]
    assert classify_overfit_result("O5", _result(0.8, 0.51))[0]
    assert not classify_overfit_result("O4", _result(0.6, 0.4))[0]


def test_o6_o7_require_five_percent_and_positive_alignment() -> None:
    assert classify_overfit_result("O6", _result(0.94, 0.06))[0]
    assert not classify_overfit_result("O7", _result(0.94, 0.04))[0]
    assert not classify_overfit_result("O6", _result(0.9, 0.1, cosine=0.0))[0]


def test_default_ladder_stops_after_first_failure() -> None:
    prior = [{"stage": "O1", "passed": True}, {"stage": "O2", "passed": False}]
    assert stages_to_run(("O1", "O2", "O2-P", "O3"), prior, False) == ()
    assert stages_to_run(("O1", "O2", "O2-P", "O3"), prior, True) == ("O2-P", "O3")


def test_o2p_identity_is_numerically_identical_to_direct_symbols() -> None:
    torch.manual_seed(23)
    symbols = torch.randn(2, 1920, dtype=torch.complex64)
    mask = make_pilot_mask((2, 64, 32), 4, time_spacing=4)
    grid, _ = insert_data_and_pilots(symbols, mask)
    torch.testing.assert_close(extract_data_resources(grid, mask), symbols)
