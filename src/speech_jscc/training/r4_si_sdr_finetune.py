"""Schedules and resume policy for SI-SDR-aware R4 fine-tuning."""
from __future__ import annotations


def si_sdr_weight(local_step: int, *, target: float, start: int = 250, full: int = 1000) -> float:
    if target < 0 or not 0 <= start < full:
        raise ValueError("invalid SI-SDR ramp")
    if local_step <= start:
        return 0.0
    if local_step >= full:
        return float(target)
    return float(target) * (local_step - start) / (full - start)


def base_loss_weights(source_global_step: int, loss: dict) -> dict[str, float]:
    if source_global_step < 0:
        raise ValueError("source global step must be nonnegative")
    ramp = 0.0 if source_global_step < 4000 else min(1.0, (source_global_step - 4000 + 1) / 8000)
    return {"latent": float(loss["latent_mse_weight"]), "stft": float(loss["multires_stft_weight"]) * ramp, "waveform": float(loss["waveform_l1_weight"]) * ramp, "channel_free": float(loss["channel_free_weight"]), "si_sdr": 0.0}
