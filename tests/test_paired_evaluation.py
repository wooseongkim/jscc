import torch

from evaluation.paired import generate_paired_evaluation_batch, run_mode_on_paired_batch
from speech_jscc.codecs import MockContinuousCodec
from speech_jscc.models import SpeechJSCC


def test_paired_modes_share_waveform_channel_jammer_and_noise():
    codec = MockContinuousCodec(3, 4, 2, 96, seed=5)
    model = SpeechJSCC((3, 4, 2), 18, hidden_dim=24)
    batch = generate_paired_evaluation_batch(
        codec,
        batch_size=4,
        waveform_samples=96,
        channel_shape=(18,),
        snr_db=8.0,
        jsr_db=-2.0,
        jammer_type="burst",
        jammed_fraction=0.25,
        pilot_spacing=4,
        pilot_time_spacing=None,
        target_power=1.0,
        seed=1234,
        device=torch.device("cpu"),
        fading="flat",
    )
    state = torch.stack((batch.snr_db / 20.0, batch.jsr_db / 20.0), dim=1)
    mode_gates = {
        "uniform": torch.ones(4, 3),
        "rule_based": torch.tensor([[1.0, 1.0, 0.0]]).expand(4, 3),
        "learned": torch.tensor([[1.0, 0.7, 0.2]]).expand(4, 3),
    }
    results = {
        mode: run_mode_on_paired_batch(
            codec, model, batch, state, gates, equalizer="estimated", fading="flat"
        )
        for mode, gates in mode_gates.items()
    }

    reference = results["uniform"]
    for mode in ("rule_based", "learned"):
        result = results[mode]
        torch.testing.assert_close(result["noise"], reference["noise"], rtol=0, atol=0)
        torch.testing.assert_close(result["jammer"], reference["jammer"], rtol=0, atol=0)
        torch.testing.assert_close(
            result["signal_fading"], reference["signal_fading"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            result["jammer_fading"], reference["jammer_fading"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            result["faded_jammer"], reference["faded_jammer"], rtol=0, atol=0
        )
        torch.testing.assert_close(result["pilot_mask"], reference["pilot_mask"])


def test_paired_batch_seed_is_reproducible():
    codec = MockContinuousCodec(2, 3, 2, 64, seed=2)
    arguments = dict(
        codec=codec,
        batch_size=2,
        waveform_samples=64,
        channel_shape=(12,),
        snr_db=5.0,
        jsr_db=0.0,
        jammer_type="pilot",
        jammed_fraction=0.25,
        pilot_spacing=3,
        pilot_time_spacing=None,
        target_power=1.0,
        device=torch.device("cpu"),
        fading="flat",
    )
    first = generate_paired_evaluation_batch(**arguments, seed=77)
    repeated = generate_paired_evaluation_batch(**arguments, seed=77)
    different = generate_paired_evaluation_batch(**arguments, seed=78)

    torch.testing.assert_close(first.waveform, repeated.waveform, rtol=0, atol=0)
    torch.testing.assert_close(first.noise, repeated.noise, rtol=0, atol=0)
    torch.testing.assert_close(first.jammer, repeated.jammer, rtol=0, atol=0)
    assert not torch.equal(first.noise, different.noise)

