from pathlib import Path

import torch

from speech_jscc.checkpoint import build_checkpoint_metadata, validate_checkpoint_metadata
from speech_jscc.codecs import MockContinuousCodec, SpeechTokenizerWrapper
from speech_jscc.data import CachedCodecDataset, load_waveform_segment, resolve_waveform_splits
from train_latent_jscc import layer_weighted_latent_mse


class CountingCodec:
    representation_shape = (2, 3, 4)

    def __init__(self):
        self.calls = 0

    def encode_waveform(self, waveform):
        self.calls += 1
        value = waveform.mean(dim=1).reshape(-1, 1, 1, 1)
        return value.expand(-1, 2, 3, 4).clone()


def test_waveform_split_is_deterministic():
    config = {"waveform_paths": [f"audio_{index}.pt" for index in range(10)], "val_fraction": 0.2}
    first = resolve_waveform_splits(config, seed=17)
    second = resolve_waveform_splits(config, seed=17)

    assert first == second
    assert len(first[0]) == 8
    assert len(first[1]) == 2
    assert set(first[0]).isdisjoint(first[1])


def test_continuous_latent_cache_avoids_reencoding(tmp_path):
    waveform_path = tmp_path / "sample.pt"
    torch.save({"waveform": torch.linspace(-1, 1, 80), "sample_rate": 16000}, waveform_path)
    codec = CountingCodec()
    dataset = CachedCodecDataset(
        [waveform_path],
        codec,
        sample_rate=16000,
        waveform_samples=64,
        device=torch.device("cpu"),
        split="train",
        cache_dir=tmp_path / "cache",
        cache_namespace="test-codec",
    )

    first_latent, first_waveform = dataset[0]
    second_latent, second_waveform = dataset[0]

    assert codec.calls == 1
    assert first_latent.shape == (2, 3, 4)
    assert first_waveform.shape == (64,)
    torch.testing.assert_close(second_latent, first_latent)
    torch.testing.assert_close(second_waveform, first_waveform)


def test_per_layer_power_normalization_removes_scale_bias():
    target = torch.ones(2, 2, 3, 4)
    target[:, 0] *= 10.0
    reconstruction = target * 0.9

    raw_loss, raw_mse = layer_weighted_latent_mse(
        reconstruction, target, torch.ones(2), "none"
    )
    normalized_loss, normalized_raw_mse = layer_weighted_latent_mse(
        reconstruction,
        target,
        torch.ones(2),
        {"mode": "per_layer_power", "epsilon": 1e-8},
    )

    torch.testing.assert_close(raw_mse, torch.tensor([1.0, 0.01]))
    torch.testing.assert_close(normalized_raw_mse, raw_mse)
    torch.testing.assert_close(raw_loss, torch.tensor(0.505))
    torch.testing.assert_close(normalized_loss, torch.tensor(0.01))


def test_checkpoint_metadata_distinguishes_codec_and_records_normalization():
    codec = MockContinuousCodec(2, 3, 4, 120)
    config = {
        "codec": {"type": "mock", "waveform_samples": 120, "sample_rate": 16000},
        "train": {"latent_normalization": {"mode": "per_layer_power", "epsilon": 1e-6}},
    }
    metadata = build_checkpoint_metadata(
        config, codec, representation_source="mock-codec synthetic waveforms"
    )

    assert metadata["checkpoint_kind"] == "mock_continuous_jscc"
    assert metadata["codec_name"] == "mock_continuous"
    assert metadata["latent_dim"] == 4
    assert metadata["sample_rate"] == 16000
    assert metadata["normalization"]["mode"] == "per_layer_power"
    validate_checkpoint_metadata(metadata, config, codec)


def test_real_speechtokenizer_is_frozen_when_installed():
    checkpoint = Path(
        "artifacts/codecs/speechtokenizer/speechtokenizer_hubert_avg/SpeechTokenizer.pt"
    )
    if not checkpoint.exists():
        return
    codec = SpeechTokenizerWrapper(
        config_path=checkpoint.with_name("config.json"),
        checkpoint_path=checkpoint,
        waveform_samples=640,
        n_q=2,
        fallback_to_mock=False,
        freeze=True,
    )

    codec.train()

    assert not codec.model.training
    assert not any(parameter.requires_grad for parameter in codec.parameters())
    assert codec.frame_rate == 50.0


def test_float_wav_loader_works_without_torchaudio():
    path = Path("SpeechTokenizer/samples/example_input.wav")
    if not path.exists():
        return
    waveform = load_waveform_segment(path, sample_rate=16000, waveform_samples=640)

    assert waveform.shape == (640,)
    assert waveform.dtype == torch.float32
    assert torch.isfinite(waveform).all()
