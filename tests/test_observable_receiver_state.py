from __future__ import annotations

import torch

from models.observable_channel_state import build_observable_receiver_state_v1


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    received = torch.ones(2, 8, 4, dtype=torch.complex64)
    pilots = torch.zeros_like(received)
    pilot_mask = torch.zeros(2, 8, 4, dtype=torch.bool)
    pilot_mask[:, ::2, ::2] = True
    pilots[pilot_mask] = 1.0 + 0.0j
    estimated = torch.ones_like(received) * (0.8 + 0.2j)
    received = estimated * pilots + 0.05 * (1.0 + 1.0j)
    return received, pilots, pilot_mask, estimated


def test_observable_receiver_state_shape_and_finite_values() -> None:
    received, pilots, pilot_mask, estimated = _inputs()

    state = build_observable_receiver_state_v1(received, pilots, pilot_mask, estimated)

    assert state.shape == (2, 8)
    assert torch.isfinite(state).all()


def test_observable_receiver_state_depends_on_pilots_not_simulator_labels() -> None:
    received, pilots, pilot_mask, estimated = _inputs()

    baseline = build_observable_receiver_state_v1(received, pilots, pilot_mask, estimated)
    corrupted = build_observable_receiver_state_v1(
        received + pilot_mask.to(received.dtype) * (0.2 + 0.0j),
        pilots,
        pilot_mask,
        estimated,
    )

    assert not torch.allclose(baseline, corrupted)


def test_observable_receiver_state_tracks_deeper_estimated_fades() -> None:
    received, pilots, pilot_mask, estimated = _inputs()
    faded = estimated.clone()
    faded[:, 3:5, :] *= 0.05

    baseline = build_observable_receiver_state_v1(received, pilots, pilot_mask, estimated)
    deeper = build_observable_receiver_state_v1(received, pilots, pilot_mask, faded)

    assert not torch.allclose(baseline[:, 3:5], deeper[:, 3:5])
