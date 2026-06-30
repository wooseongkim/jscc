from __future__ import annotations

import math

import torch
from torch import Tensor


def synthetic_waveforms(
    batch_size: int,
    samples: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Generate speech-like harmonic mixtures for a dependency-free smoke run."""
    time = torch.linspace(0.0, 1.0, samples, device=device)[None, :]
    f0 = torch.empty(batch_size, 1, device=device).uniform_(2.0, 8.0, generator=generator)
    phase = torch.empty(batch_size, 1, device=device).uniform_(
        0.0, 2.0 * math.pi, generator=generator
    )
    waveform = torch.sin(2.0 * math.pi * f0 * time + phase)
    waveform += 0.35 * torch.sin(4.0 * math.pi * f0 * time + 0.5 * phase)
    return waveform / waveform.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
