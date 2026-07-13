from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from speech_jscc.codecs.base import BaseCodec
from speech_jscc.codecs.mock import MockContinuousCodec


class _Wrapper(BaseCodec):
    external_name = "external codec"

    def __init__(self, codec: BaseCodec | None, fallback_to_mock: bool,
                 mock_config: dict[str, Any] | None):
        super().__init__()
        if codec is None:
            if not fallback_to_mock:
                raise ImportError(f"{self.external_name} adapter was not supplied")
            codec = MockContinuousCodec(**(mock_config or {}))
            self.using_mock = True
        elif not isinstance(codec, BaseCodec):
            raise TypeError("codec must implement BaseCodec")
        else:
            self.using_mock = False
        self.codec = codec

    @property
    def representation_shape(self) -> tuple[int, int, int]:
        return self.codec.representation_shape

    def get_codebook(self) -> Tensor | None:
        return self.codec.get_codebook()

    def encode_waveform(self, waveform: Tensor) -> Tensor:
        return self.codec.encode_waveform(waveform)

    def decode_representation(self, representation: Tensor) -> Tensor:
        return self.codec.decode_representation(representation)

    def official_reconstruct_waveform(self, waveform: Tensor) -> Tensor:
        representation = self.encode_waveform(waveform)
        return self.decode_representation(representation)


def _speech_tokenizer_class():
    """Import the inference model without requiring optional trainer dependencies."""
    try:
        from speechtokenizer import SpeechTokenizer

        return SpeechTokenizer
    except ModuleNotFoundError as import_error:
        package_spec = importlib.util.find_spec("speechtokenizer")
        if package_spec is None or not package_spec.submodule_search_locations:
            raise ImportError("SpeechTokenizer is not installed") from import_error
        package_dir = Path(next(iter(package_spec.submodule_search_locations)))
        alias = "_speech_jscc_speechtokenizer"
        if alias not in sys.modules:
            package = types.ModuleType(alias)
            package.__path__ = [str(package_dir)]
            package.__package__ = alias
            sys.modules[alias] = package
        model_name = f"{alias}.model"
        if model_name not in sys.modules:
            model_spec = importlib.util.spec_from_file_location(model_name, package_dir / "model.py")
            if model_spec is None or model_spec.loader is None:
                raise ImportError("cannot load SpeechTokenizer inference model") from import_error
            module = importlib.util.module_from_spec(model_spec)
            sys.modules[model_name] = module
            model_spec.loader.exec_module(module)
        return sys.modules[model_name].SpeechTokenizer


class SpeechTokenizerWrapper(BaseCodec):
    """Continuous SpeechTokenizer RVQ-embedding adapter for JSCC.

    The wrapper calls `forward_feature` and transports `[B,L,T,D]` embedding
    tensors. RVQ indices are never exposed to the communication system.
    """

    def __init__(
        self,
        codec: BaseCodec | None = None,
        *,
        config_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        waveform_samples: int | None = None,
        n_q: int | None = None,
        fallback_to_mock: bool = True,
        mock_config: dict[str, Any] | None = None,
        freeze: bool = True,
    ):
        super().__init__()
        self.using_mock = False
        self.codec: BaseCodec | None = None
        self.model = None
        if codec is not None:
            if not isinstance(codec, BaseCodec):
                raise TypeError("codec must implement BaseCodec")
            self.codec = codec
            return
        if config_path is None or checkpoint_path is None:
            if not fallback_to_mock:
                raise ValueError("SpeechTokenizer requires config_path and checkpoint_path")
            self.codec = MockContinuousCodec(**(mock_config or {}))
            self.using_mock = True
            return
        if waveform_samples is None or waveform_samples <= 0:
            raise ValueError("waveform_samples must be positive for SpeechTokenizer")

        model_class = _speech_tokenizer_class()
        self.model = model_class.load_from_checkpoint(str(config_path), str(checkpoint_path))
        self.model.eval()
        self.frozen = bool(freeze)
        if freeze:
            self.model.requires_grad_(False)
        self.waveform_samples = int(waveform_samples)
        self.sample_rate = int(self.model.sample_rate)
        self.n_q = int(n_q or self.model.n_q)
        if not 1 <= self.n_q <= self.model.n_q:
            raise ValueError(f"n_q must be between 1 and {self.model.n_q}")
        with torch.inference_mode():
            encoded = self.model.encoder(torch.zeros(1, 1, self.waveform_samples))
        self.frames = int(encoded.shape[-1])
        self.latent_dim = int(encoded.shape[1])
        self.frame_rate = float(self.sample_rate / self.model.downsample_rate)

    def train(self, mode: bool = True):
        """Keep a frozen pretrained codec in inference mode during JSCC training."""
        super().train(mode)
        if self.model is not None and getattr(self, "frozen", False):
            self.model.eval()
        return self

    @property
    def representation_shape(self) -> tuple[int, int, int]:
        if self.codec is not None:
            return self.codec.representation_shape
        return self.n_q, self.frames, self.latent_dim

    def get_codebook(self) -> Tensor | None:
        if self.codec is not None:
            return self.codec.get_codebook()
        layers = self.model.quantizer.vq.layers[: self.n_q]
        return torch.stack([layer.codebook.squeeze(0) for layer in layers], dim=0)

    def encode_waveform(self, waveform: Tensor) -> Tensor:
        if self.codec is not None:
            return self.codec.encode_waveform(waveform)
        if waveform.ndim != 2 or waveform.shape[1] != self.waveform_samples:
            raise ValueError(f"waveform must have shape [B,{self.waveform_samples}]")
        if not torch.isfinite(waveform).all():
            raise ValueError("waveform contains NaN or Inf")
        self.model.eval()
        with torch.no_grad():
            layers = self.model.forward_feature(
                waveform.unsqueeze(1), layers=list(range(self.n_q))
            )
        if len(layers) != self.n_q:
            raise RuntimeError(f"SpeechTokenizer returned {len(layers)} layers, expected {self.n_q}")
        expected_shape = tuple(layers[0].shape)
        for index, layer in enumerate(layers):
            if tuple(layer.shape) != expected_shape:
                raise RuntimeError(
                    f"SpeechTokenizer layer {index} shape {tuple(layer.shape)} "
                    f"does not match layer 0 shape {expected_shape}"
                )
            if not torch.isfinite(layer).all():
                raise RuntimeError(f"SpeechTokenizer layer {index} contains NaN or Inf")
        return torch.stack([layer.permute(0, 2, 1) for layer in layers], dim=1)

    def decode_representation(self, representation: Tensor) -> Tensor:
        if self.codec is not None:
            return self.codec.decode_representation(representation)
        if representation.ndim != 4 or tuple(representation.shape[1:]) != self.representation_shape:
            raise ValueError(f"representation must have shape [B,{self.representation_shape}]")
        if not torch.isfinite(representation).all():
            raise ValueError("representation contains NaN or Inf")
        self.model.eval()
        quantized = representation.permute(0, 1, 3, 2).sum(dim=1)
        expected_quantized_shape = (representation.shape[0], self.latent_dim, self.frames)
        if tuple(quantized.shape) != expected_quantized_shape:
            raise RuntimeError(
                "continuous_sum decoder input shape mismatch: "
                f"got {tuple(quantized.shape)}, expected {expected_quantized_shape}"
            )
        waveform = self.model.decoder(quantized).squeeze(1)
        if waveform.shape[-1] > self.waveform_samples:
            waveform = waveform[..., : self.waveform_samples]
        elif waveform.shape[-1] < self.waveform_samples:
            waveform = F.pad(waveform, (0, self.waveform_samples - waveform.shape[-1]))
        if not torch.isfinite(waveform).all():
            raise RuntimeError("decoded waveform contains NaN or Inf")
        return waveform

    def official_reconstruct_waveform(self, waveform: Tensor) -> Tensor:
        """Reconstruct with the official SpeechTokenizer code path.

        This intentionally goes through `encode()` RVQ codes and `decode(codes)`
        instead of the JSCC-facing continuous `[B,L,T,D]` representation.
        """
        if self.codec is not None:
            representation = self.codec.encode_waveform(waveform)
            return self.codec.decode_representation(representation)
        if waveform.ndim != 2 or waveform.shape[1] != self.waveform_samples:
            raise ValueError(f"waveform must have shape [B,{self.waveform_samples}]")
        if not torch.isfinite(waveform).all():
            raise ValueError("waveform contains NaN or Inf")
        self.model.eval()
        with torch.no_grad():
            codes = self.model.encode(waveform.unsqueeze(1), n_q=self.n_q)
            waveform_out = self.model.decode(codes).squeeze(1)
        if waveform_out.shape[-1] > self.waveform_samples:
            waveform_out = waveform_out[..., : self.waveform_samples]
        elif waveform_out.shape[-1] < self.waveform_samples:
            waveform_out = F.pad(waveform_out, (0, self.waveform_samples - waveform_out.shape[-1]))
        if not torch.isfinite(waveform_out).all():
            raise RuntimeError("official decoded waveform contains NaN or Inf")
        return waveform_out


class EnCodecWrapper(_Wrapper):
    external_name = "EnCodec"

    def __init__(self, codec: BaseCodec | None = None, *, fallback_to_mock: bool = True,
                 mock_config: dict[str, Any] | None = None):
        super().__init__(codec, fallback_to_mock, mock_config)


__all__ = ["EnCodecWrapper", "SpeechTokenizerWrapper"]
