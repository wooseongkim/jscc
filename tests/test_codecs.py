import pytest
import torch
from pathlib import Path

from speech_jscc.codecs import (
    BaseCodec,
    EnCodecWrapper,
    MockContinuousCodec,
    SpeechTokenizerWrapper,
)


def test_base_codec_is_abstract():
    with pytest.raises(TypeError):
        BaseCodec()


def test_mock_codec_is_deterministic_and_reconstructs_waveform_shape():
    config = dict(layers=3, frames=8, latent_dim=5, waveform_samples=320, seed=17)
    first_codec = MockContinuousCodec(**config)
    second_codec = MockContinuousCodec(**config)
    waveform = torch.randn(4, 320)

    first = first_codec.encode_waveform(waveform)
    repeated = first_codec.encode_waveform(waveform)
    same_seed = second_codec.encode_waveform(waveform)
    reconstructed = first_codec.decode_representation(first)

    assert first.shape == (4, 3, 8, 5)
    assert reconstructed.shape == waveform.shape
    torch.testing.assert_close(first, repeated)
    torch.testing.assert_close(first, same_seed)


def test_mock_codec_exposes_continuous_codebook_embeddings():
    codec = MockContinuousCodec(2, 6, 4, 240, codebook_size=11, seed=3)
    codebook = codec.get_codebook()

    assert codebook.shape == (2, 11, 4)
    assert codebook.is_floating_point()
    assert not codebook.requires_grad


@pytest.mark.parametrize("wrapper_type", [SpeechTokenizerWrapper, EnCodecWrapper])
def test_optional_codec_wrappers_fall_back_to_mock(wrapper_type):
    wrapper = wrapper_type(
        mock_config={
            "layers": 2,
            "frames": 5,
            "latent_dim": 3,
            "waveform_samples": 160,
            "seed": 9,
        }
    )
    waveform = torch.randn(2, 160)
    representation = wrapper.encode_waveform(waveform)

    assert wrapper.using_mock
    assert wrapper.representation_shape == (2, 5, 3)
    assert representation.shape == (2, 2, 5, 3)
    assert wrapper.decode_representation(representation).shape == waveform.shape
    assert wrapper.get_codebook() is not None


def test_wrapper_can_require_an_external_adapter():
    with pytest.raises(ImportError):
        EnCodecWrapper(fallback_to_mock=False)


def test_requested_src_codec_namespace_is_available():
    from src.codecs.mock_codec import MockContinuousCodec as SourceTreeMockCodec

    codec = SourceTreeMockCodec(2, 4, 3, 80, seed=1)
    assert codec.encode_waveform(torch.zeros(1, 80)).shape == (1, 2, 4, 3)


@pytest.mark.skipif(
    not Path("artifacts/codecs/speechtokenizer/speechtokenizer_hubert_avg/SpeechTokenizer.pt").exists(),
    reason="official SpeechTokenizer checkpoint is not installed",
)
def test_real_speechtokenizer_continuous_embedding_round_trip():
    wrapper = SpeechTokenizerWrapper(
        config_path="artifacts/codecs/speechtokenizer/speechtokenizer_hubert_avg/config.json",
        checkpoint_path="artifacts/codecs/speechtokenizer/speechtokenizer_hubert_avg/SpeechTokenizer.pt",
        waveform_samples=640,
        n_q=2,
        fallback_to_mock=False,
    )
    waveform = torch.zeros(1, 640)
    representation = wrapper.encode_waveform(waveform)
    reconstruction = wrapper.decode_representation(representation)

    assert representation.shape == (1, *wrapper.representation_shape)
    assert wrapper.representation_shape[0] == 2
    assert reconstruction.shape == waveform.shape
    assert wrapper.get_codebook().shape[:2] == (2, 1024)
