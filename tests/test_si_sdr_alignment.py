import torch

from speech_jscc.evaluation.si_sdr_alignment import best_cross_correlation_alignment
from speech_jscc.metrics.audio_quality import compute_si_sdr


def test_standard_si_sdr_is_zero_mean_with_fixed_epsilon():
    reference = torch.tensor([[0.0, 1.0, 0.0, -1.0]])
    shifted_offset = reference + 7.0
    assert torch.allclose(compute_si_sdr(reference, shifted_offset), compute_si_sdr(reference, reference))


def test_cross_correlation_recovers_small_delay_without_changing_metric():
    reference = torch.sin(torch.linspace(0, 20, 1600))[None, :]
    estimate = torch.cat((torch.zeros(8), reference[0, :-8]))[None, :]
    result = best_cross_correlation_alignment(reference, estimate, sample_rate=16000, max_lag_ms=2.0)
    assert abs(result.shift_samples) == 8
    assert result.si_sdr_db > 40.0


def test_alignment_rejects_empty_overlap():
    reference = torch.ones(8)
    estimate = torch.ones(8)
    result = best_cross_correlation_alignment(reference, estimate, sample_rate=16000, max_lag_ms=0)
    assert result.overlap_samples == 8
