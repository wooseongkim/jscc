from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path

import pytest
import torch

from speech_jscc.diagnostics.waveform_aware_wireless import (
    clean_validation_conditions,
    ideal_ofdm_round_trip,
    ragged_tensor_diagnostics,
    validate_cf2_contract,
    wireless_feasibility_gate,
)
from speech_jscc.models.conv_conformer import (
    balanced_ragged_valid_mask,
    masked_complex_power_normalize,
)
from train_waveform_aware_clean_channel import decode_for_wireless_loss


class _Encoder:
    representation_shape = (8, 50, 1024)
    layer_channel_uses = (240,) * 8
    total_channel_uses = 1920
    symbol_frames = 50
    complex_channels_per_symbol_frame = 5
    temporal_symbol_layout = "balanced_ragged"
    uses_temporal_interpolation = False
    symbol_valid_mask = balanced_ragged_valid_mask(
        frames=50, max_symbols=5, valid_symbols=240
    )


class _Model:
    encoder = _Encoder()


def test_cf2_contract_has_exact_ragged_and_pilot_resources():
    summary = validate_cf2_contract(_Model(), {"model": {"grid_shape": [64, 32]}})
    assert summary["representation_shape"] == [8, 50, 1024]
    assert summary["valid_symbols_per_layer"] == 240
    assert summary["valid_symbols_total"] == 1920
    assert summary["four_symbol_frames"] == [2, 7, 12, 17, 22, 27, 32, 37, 42, 47]
    assert summary["uses_temporal_interpolation"] is False
    assert summary["pilot_resources"] == 128
    assert summary["data_resources"] == 1920
    assert summary["passed"] is True


def test_masked_slots_remain_zero_and_do_not_affect_normalized_power():
    mask = _Encoder.symbol_valid_mask
    value = torch.complex(torch.randn(2, 50, 5), torch.randn(2, 50, 5))
    altered = value.clone()
    altered[..., ~mask] = 1e6 + 1e6j
    first = masked_complex_power_normalize(value, mask, 1.0)
    second = masked_complex_power_normalize(altered, mask, 1.0)
    assert torch.equal(first[..., ~mask], torch.zeros_like(first[..., ~mask]))
    assert torch.equal(second[..., ~mask], torch.zeros_like(second[..., ~mask]))
    assert torch.allclose(first[..., mask], second[..., mask])
    assert torch.allclose(first[..., mask].abs().square().mean(-1), torch.ones(2))


def test_ragged_tensor_diagnostics_rejects_nonzero_masked_slots():
    mask = _Encoder.symbol_valid_mask
    fixed = torch.zeros(2, 8, 50, 5, dtype=torch.complex64)
    fixed[..., mask] = 1 + 0j
    assert ragged_tensor_diagnostics(fixed, mask)["passed"] is True
    fixed[..., 2, 4] = 1 + 0j
    with pytest.raises(ValueError, match="masked ragged slots"):
        ragged_tensor_diagnostics(fixed, mask)


def test_ideal_ofdm_round_trip_is_exact_and_pilots_do_not_overlap_data():
    generator = torch.Generator().manual_seed(23)
    symbols = torch.complex(
        torch.randn(2, 1920, generator=generator),
        torch.randn(2, 1920, generator=generator),
    )
    result = ideal_ofdm_round_trip(symbols, _Model(), {"model": {"grid_shape": [64, 32]}})
    assert result["max_abs_error"] <= 1e-7
    assert result["pilot_resources"] == 128
    assert result["data_resources"] == 1920
    assert result["pilot_data_disjoint"] is True
    assert result["pilot_data_exhaustive"] is True


def test_clean_validation_conditions_are_deterministic_and_cover_snr_bins():
    first = clean_validation_conditions(23, utterance_count=64, realizations_per_utterance=2)
    second = clean_validation_conditions(23, utterance_count=64, realizations_per_utterance=2)
    assert first == second
    random_rows = [row for row in first if row["channel_policy"] == "random"]
    assert {row["snr_db"] for row in random_rows} == {5.0, 10.0, 15.0}
    assert len(random_rows) == 64 * 2 * 3
    assert len({row["seed"] for row in random_rows}) == len(random_rows)


@pytest.mark.parametrize(
    ("metrics", "passed"),
    [
        ({"delta_si_sdr_db": -1.0, "delta_waveform_snr_db": -1.0, "stft_ratio": 1.2}, True),
        ({"delta_si_sdr_db": -1.01, "delta_waveform_snr_db": 0.0, "stft_ratio": 1.0}, False),
    ],
)
def test_wireless_feasibility_gate(metrics, passed):
    assert wireless_feasibility_gate(metrics)["passed"] is passed


def test_evaluation_cli_dry_run_does_not_start_codec_or_optimizer():
    result = subprocess.run(
        [
            sys.executable,
            "eval_waveform_aware_wireless.py",
            "--mode",
            "all",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"dry_run": true' in result.stdout.lower()
    assert "best_waveform_si_sdr.pt" in result.stdout


def test_training_refuses_when_zero_shot_gate_already_passed(tmp_path: Path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"random": {"gate": {"passed": True}}}))
    result = subprocess.run(
        [
            sys.executable,
            "train_waveform_aware_clean_channel.py",
            "--zero-shot-summary",
            str(summary),
            "--steps",
            "1",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "fine-tuning is not required" in (result.stdout + result.stderr)


def test_external_script_dry_run_contains_conditional_training_and_checkpoint_names():
    result = subprocess.run(
        ["bash", "scripts/run_waveform_aware_wireless_external.sh", "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "eval_waveform_aware_wireless.py" in result.stdout
    assert "train_waveform_aware_clean_channel.py" in result.stdout
    assert "best_summed_latent_nmse.pt" in result.stdout
    assert "best_waveform_si_sdr.pt" in result.stdout


def test_wireless_waveform_loss_decoder_reenables_frozen_rnn_backward():
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = torch.nn.GRU(3, 1, batch_first=True)

        def forward(self, value):
            return self.rnn(value)[0].transpose(1, 2)

    class Codec(torch.nn.Module):
        representation_shape = (2, 3, 4)
        waveform_samples = 3

        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.decoder = Decoder()
            self.requires_grad_(False)

    codec = Codec()
    codec.eval()
    latent = torch.randn(1, 2, 3, 4, requires_grad=True)
    waveform = decode_for_wireless_loss(codec, latent)
    waveform.square().mean().backward()
    assert latent.grad is not None
    assert float(latent.grad.abs().sum()) > 0
    assert all(parameter.grad is None for parameter in codec.parameters())
