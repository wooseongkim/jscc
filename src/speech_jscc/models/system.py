from __future__ import annotations

from torch import Tensor, nn

from speech_jscc.models.decoder import JSCCDecoder
from speech_jscc.models.encoder import JSCCEncoder


class SpeechJSCC(nn.Module):
    """Analog neural JSCC mapping continuous codec tensors to complex symbols."""

    def __init__(self, representation_shape: tuple[int, int, int], channel_uses: int,
                 channel_state_dim: int = 2, hidden_dim: int = 128, target_power: float = 1.0):
        super().__init__()
        self.encoder = JSCCEncoder(representation_shape, channel_uses, channel_state_dim,
                                   hidden_dim, target_power)
        self.decoder = JSCCDecoder(representation_shape, channel_uses, channel_state_dim,
                                   hidden_dim)
        self.architecture = "flat_mlp"
        self.architecture_version = "flat_mlp_v1"

    def encode(self, representation: Tensor, channel_state: Tensor) -> Tensor:
        return self.encoder(representation, channel_state)

    def decode(self, received: Tensor, channel_state: Tensor) -> Tensor:
        return self.decoder(received, channel_state)
