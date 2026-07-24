from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from typing import Any, Iterable

import torch

from speech_jscc.diagnostics.content_generalization import parse_speaker_id, summarize_content_metrics
from speech_jscc.diagnostics.o5_root_cause import optimal_scale_diagnostics


EXPOSURE_ENGINE_VERSION = "g0_exposure_normalized_v1"
CHECKPOINT_EPOCHS = (1, 2, 4, 8, 16, 32, 64)


def steps_for_epochs(epochs: int, subset_size: int, batch_size: int) -> int:
    if min(epochs, subset_size, batch_size) <= 0: raise ValueError("epochs, subset size, and batch size must be positive")
    return math.ceil(epochs * subset_size / batch_size)


def verify_resume_replay(current_counts: dict[str, int], first_new_batch: list[str], saved_counts: dict[str, int]) -> None:
    replay = Counter(current_counts); replay.subtract(first_new_batch)
    if dict(replay) != dict(saved_counts): raise ValueError("resume sampler presentation mismatch")


def should_continue_exposure(epoch: int, slopes: dict[str, float], tolerance: float) -> bool:
    if epoch < 16: return True
    if slopes["train_loss"] < -tolerance or slopes["unseen_loss"] < -tolerance: return True
    plateau = all(abs(slopes[key]) <= tolerance for key in ("train_loss", "unseen_loss", "unseen_correlation"))
    return not plateau


class EpochSampler:
    def __init__(self, utterance_ids: list[str], *, batch_size: int, seed: int, subset_key: str):
        if not utterance_ids or batch_size <= 0: raise ValueError("sampler requires utterances and positive batch size")
        self.utterance_ids = list(utterance_ids); self.batch_size = int(batch_size); self.seed = int(seed); self.subset_key = str(subset_key)
        self.presentation_counts = Counter({item: 0 for item in utterance_ids}); self.optimizer_steps = 0

    def permutation(self, epoch: int) -> list[str]:
        digest = hashlib.sha256(f"g0_epoch_v1|{self.seed}|{self.subset_key}|{epoch}".encode()).digest()
        values = list(self.utterance_ids); random.Random(int.from_bytes(digest[:8], "big")).shuffle(values); return values

    def iter_epoch(self, epoch: int):
        values = self.permutation(epoch)
        for start in range(0, len(values), self.batch_size):
            batch = values[start:start + self.batch_size]
            self.presentation_counts.update(batch); self.optimizer_steps += 1; yield batch

    def state_dict(self) -> dict[str, Any]:
        return {"presentation_counts": dict(self.presentation_counts), "optimizer_steps": self.optimizer_steps}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.presentation_counts = Counter(state["presentation_counts"]); self.optimizer_steps = int(state["optimizer_steps"])

    def batch_stream(self, max_epochs: int):
        stream = []
        epoch = 1
        target_steps = steps_for_epochs(max_epochs, len(self.utterance_ids), self.batch_size)
        while len(stream) < target_steps * self.batch_size:
            stream.extend(self.permutation(epoch)); epoch += 1
        for step in range(1, target_steps + 1):
            batch = stream[(step - 1) * self.batch_size:step * self.batch_size]
            self.presentation_counts.update(batch); self.optimizer_steps = step; yield step, batch


def compute_train_baselines(items: Iterable[tuple[str, torch.Tensor]], *, min_speaker_samples: int = 2) -> dict[str, Any]:
    total = None; count = 0; speaker_sums: dict[str, torch.Tensor] = {}; speaker_counts: Counter[str] = Counter()
    for identifier, latent in items:
        value = latent.detach().float().cpu(); total = value.clone() if total is None else total + value; count += 1
        speaker = parse_speaker_id(identifier); speaker_sums[speaker] = value.clone() if speaker not in speaker_sums else speaker_sums[speaker] + value; speaker_counts[speaker] += 1
    if total is None: raise ValueError("baseline computation requires train latents")
    global_mean = total / count
    layer_scalar = global_mean.mean(dim=(1, 2))[:, None, None].expand_as(global_mean)
    speaker_means = {speaker: value / speaker_counts[speaker] for speaker, value in speaker_sums.items() if speaker != "unknown" and speaker_counts[speaker] >= min_speaker_samples}
    return {"zero": torch.zeros_like(global_mean), "global_mean": global_mean, "layerwise_mean": layer_scalar,
            "speaker_means": speaker_means, "speaker_counts": dict(speaker_counts),
            "min_speaker_samples": min_speaker_samples}


def evaluate_baselines(target: torch.Tensor, identifiers: list[str], baselines: dict[str, Any], *, group: str) -> dict[str, Any]:
    output = {}
    for name in ("zero", "global_mean", "layerwise_mean"):
        prediction = baselines[name].to(target.device).unsqueeze(0).expand_as(target)
        output[name] = summarize_content_metrics(prediction, target, group=group)
    available = []; predictions = []
    for identifier in identifiers:
        value = baselines["speaker_means"].get(parse_speaker_id(identifier)); available.append(value is not None)
        if value is not None: predictions.append(value)
    if all(available) and predictions:
        output["speaker_conditional_mean"] = summarize_content_metrics(torch.stack(predictions).to(target.device), target, group=group)
    else:
        output["speaker_conditional_mean"] = {"available": False, "available_count": sum(available), "sample_count": len(identifiers)}
    return output


def exposure_metric_summary(reconstruction: torch.Tensor, target: torch.Tensor, *, group: str) -> dict[str, Any]:
    output = summarize_content_metrics(reconstruction, target, group=group)
    scales = optimal_scale_diagnostics(reconstruction, target, 1e-6, torch.ones(target.shape[1]))
    for layer, row in enumerate(output["per_layer"]):
        current = reconstruction[:, layer].detach().float()
        row["reconstruction_mean"] = float(current.mean()); row["reconstruction_std"] = float(current.std(unbiased=False))
        row["optimal_scalar_rescaled_normalized_loss"] = scales["per_layer"][layer]["rescaled_normalized_mse"]
    keys = [key for key, value in output["per_layer"][0].items() if isinstance(value, (int, float)) and key != "layer"]
    enhancement = {key: sum(float(row[key]) for row in output["per_layer"][1:]) / max(len(output["per_layer"]) - 1, 1) for key in keys}
    enhancement["layers"] = list(range(1, target.shape[1])); output["layers1_to_7_summary"] = enhancement
    output["aggregate"]["optimal_scalar_rescaled_normalized_loss"] = scales["stage1_layerwise_rescaled_loss"]
    output["aggregate"]["reconstruction_mean"] = float(reconstruction.mean()); output["aggregate"]["reconstruction_std"] = float(reconstruction.std(unbiased=False))
    return output


def gradient_norms(model) -> dict[str, Any]:
    def norm(parameters):
        values = [parameter.grad.detach().float().square().sum() for parameter in parameters if parameter.grad is not None]
        return float(torch.sqrt(torch.stack(values).sum())) if values else 0.0
    branches = getattr(model.encoder, "layer_encoders", getattr(model.encoder, "symbol_heads", []))
    output = {"encoder_branches": [norm(branch.parameters()) for branch in branches],
            "decoder": norm(model.decoder.parameters()), "total": norm(model.parameters())}
    output["symbol_heads"] = [norm(head.parameters()) for head in getattr(model.encoder, "symbol_heads", [])]
    output["reconstruction_heads"] = [norm(head.parameters()) for head in getattr(model.decoder, "reconstruction_heads", [])]
    return output
