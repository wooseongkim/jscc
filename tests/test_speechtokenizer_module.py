from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest
import torch

from speech_jscc.codecs.wrappers import _speech_tokenizer_class


def test_speechtokenizer_module_imports_when_optional_dependencies_are_installed() -> None:
    speechtokenizer = pytest.importorskip("speechtokenizer")

    assert hasattr(speechtokenizer, "SpeechTokenizer")
    assert hasattr(speechtokenizer.SpeechTokenizer, "load_from_checkpoint")


def test_speechtokenizer_wrapper_resolver_loads_inference_class() -> None:
    if importlib.util.find_spec("speechtokenizer") is None:
        pytest.skip("SpeechTokenizer is not installed")

    model_class = _speech_tokenizer_class()

    assert hasattr(model_class, "load_from_checkpoint")


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("SPEECHTOKENIZER_RUN_SLOW") != "1",
    reason="set SPEECHTOKENIZER_RUN_SLOW=1 to run the full SpeechTokenizer CPU shape test",
)
def test_speechtokenizer_base_config_forward_shapes() -> None:
    if importlib.util.find_spec("speechtokenizer") is None:
        pytest.skip("SpeechTokenizer is not installed")
    from speechtokenizer import SpeechTokenizer

    config_path = Path(".venv/src/speechtokenizer/config/spt_base_cfg.json")
    if not config_path.exists():
        pytest.skip("SpeechTokenizer base config is not available")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = SpeechTokenizer(config).eval()

    with torch.inference_mode():
        waveform = torch.zeros(1, 1, 16000)
        encoded = model.encoder(waveform)
        features = model.forward_feature(waveform, layers=[0, 1])
        decoded = model.decoder(encoded)

    assert encoded.shape == (1, 1024, 50)
    assert [tuple(feature.shape) for feature in features] == [(1, 1024, 50), (1, 1024, 50)]
    assert decoded.shape == (1, 1, 16000)
