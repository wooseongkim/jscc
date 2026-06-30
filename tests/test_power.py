import pytest
import torch

from speech_jscc.channels import make_jammer
from speech_jscc.models import JSCCEncoder, normalize_complex_power


def test_complex_power_normalization_is_per_example():
    symbols = torch.complex(torch.randn(4, 100), torch.randn(4, 100))
    normalized = normalize_complex_power(symbols, target_power=1.7)
    power = normalized.abs().square().mean(dim=1)
    torch.testing.assert_close(power, torch.full_like(power, 1.7), rtol=1e-5, atol=1e-5)


def test_encoder_enforces_target_power():
    encoder = JSCCEncoder((2, 3, 4), channel_uses=16, hidden_dim=24, target_power=0.5)
    symbols = encoder(torch.randn(5, 2, 3, 4), torch.randn(5, 2))
    torch.testing.assert_close(symbols.abs().square().mean(1), torch.full((5,), 0.5), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kind", ["barrage", "narrowband", "burst", "pilot"])
def test_jammer_matches_requested_jsr(kind):
    reference = normalize_complex_power(torch.complex(torch.randn(4, 64), torch.randn(4, 64)))
    jsr_db = torch.tensor([-10.0, -3.0, 0.0, 6.0])
    jammer, _ = make_jammer(reference, jsr_db, kind, jammed_fraction=0.25)
    measured = jammer.abs().square().mean(1) / reference.abs().square().mean(1)
    expected = torch.pow(10.0, jsr_db / 10.0)
    torch.testing.assert_close(measured, expected, rtol=1e-5, atol=1e-5)
