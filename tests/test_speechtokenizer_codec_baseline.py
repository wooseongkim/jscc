from __future__ import annotations

from pathlib import Path

import pytest
import torch

from eval_codec_only import align_for_metrics, waveform_metrics
from speech_jscc.codecs import SpeechTokenizerWrapper


CONFIG = Path("artifacts/codecs/speechtokenizer/speechtokenizer_hubert_avg/config.json")
CHECKPOINT = Path("artifacts/codecs/speechtokenizer/speechtokenizer_hubert_avg/SpeechTokenizer.pt")


def _codec_or_skip() -> SpeechTokenizerWrapper:
    if not CONFIG.exists() or not CHECKPOINT.exists():
        pytest.skip("SpeechTokenizer artifact is not available")
    return SpeechTokenizerWrapper(
        config_path=CONFIG,
        checkpoint_path=CHECKPOINT,
        waveform_samples=16000,
        n_q=8,
        fallback_to_mock=False,
    ).eval()


def test_speechtokenizer_official_and_continuous_reconstruction_are_finite() -> None:
    codec = _codec_or_skip()
    waveform = torch.zeros(2, 16000)

    with torch.inference_mode():
        representation = codec.encode_waveform(waveform)
        continuous = codec.decode_representation(representation)
        official = codec.official_reconstruct_waveform(waveform)

    assert representation.shape == (2, 8, 50, 1024)
    assert continuous.shape == (2, 16000)
    assert official.shape == (2, 16000)
    assert torch.isfinite(representation).all()
    assert torch.isfinite(continuous).all()
    assert torch.isfinite(official).all()
    assert waveform_metrics(waveform, official)["si_sdr_db"] == pytest.approx(
        waveform_metrics(waveform, official)["si_sdr_db"]
    )
    assert waveform_metrics(waveform, continuous)["si_sdr_db"] == pytest.approx(
        waveform_metrics(waveform, continuous)["si_sdr_db"]
    )


def test_peak_xcorr_alignment_preserves_batch_dimension() -> None:
    reference = torch.zeros(2, 32)
    reference[:, 8:16] = 1.0
    estimate = torch.zeros(2, 32)
    estimate[:, 10:18] = 1.0

    aligned_ref, aligned_est, lags = align_for_metrics(
        reference,
        estimate,
        metric_align="peak_xcorr",
        max_lag_samples=5,
    )

    assert aligned_ref.shape[0] == 2
    assert aligned_est.shape[0] == 2
    assert aligned_ref.shape == aligned_est.shape
    assert lags.shape == (2,)
