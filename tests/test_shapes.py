import torch

from speech_jscc.channels import make_jammer, rayleigh_channel
from speech_jscc.codecs import MockContinuousCodec
from speech_jscc.models import SpeechJSCC


def test_end_to_end_tensor_shapes():
    batch, samples = 3, 320
    shape, uses = (2, 5, 4), 12
    codec = MockContinuousCodec(*shape, samples)
    model = SpeechJSCC(shape, uses, hidden_dim=32)
    waveform = torch.randn(batch, samples)
    state = torch.zeros(batch, 2)

    representation = codec.encode_waveform(waveform)
    transmitted = model.encode(representation, state)
    jammer, mask = make_jammer(transmitted, torch.zeros(batch), "burst", 0.25)
    channel = rayleigh_channel(transmitted, jammer, torch.full((batch,), 10.0))
    reconstructed = model.decode(channel["equalized"], state)
    decoded_waveform = codec.decode_representation(reconstructed)

    assert representation.shape == (batch, *shape)
    assert transmitted.shape == (batch, uses)
    assert transmitted.is_complex()
    assert jammer.shape == transmitted.shape
    assert mask.shape == transmitted.shape
    assert channel["received"].shape == transmitted.shape
    assert reconstructed.shape == representation.shape
    assert decoded_waveform.shape == waveform.shape

