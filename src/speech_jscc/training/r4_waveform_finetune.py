from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor, nn

from channels.global_triplet_allocator import (
    GlobalTripletAllocation,
    GlobalTripletCSIReport,
    allocate_global_balanced_triplets,
)
from channels.physical_ofdm import (
    NR_LIKE_R4,
    active_grid_masks,
    apply_tti_multipath,
    demodulate_tti,
    estimate_comb_dft_ls,
    insert_physical_pilots,
    modulate_tti,
)
from channels.repetition_mrc import MRCResult, coherent_mrc
from models.observable_channel_state import build_observable_receiver_state_v1
from speech_jscc.training.channel_free_feasibility import (
    decode_frozen_representation_with_gradient,
    multi_resolution_stft_loss,
)
from speech_jscc.training.channel_free_revalidation import per_layer_nmse
from speech_jscc.training.si_sdr_loss import negative_si_sdr_loss


LAYER_IMPORTANCE = [1, 0, 2, 5, 3, 4, 6, 7]


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    start: int
    stop: int
    snr_probabilities: dict[float, float]
    learning_rate: float
    mapping_mode: str
    waveform_scale: float


class R4Curriculum:
    def __init__(
        self,
        *,
        stage_a_learning_rate: float = 1e-4,
        stage_b_learning_rate: float = 5e-5,
        stage_c_learning_rate: float = 2e-5,
        stage_c_probabilities: dict[float, float] | None = None,
    ):
        self.stages = (
            CurriculumStage(
                "A", 0, 4000, {10.0: .6, 15.0: .4},
                stage_a_learning_rate, "bootstrap_fixed", 0.0,
            ),
            CurriculumStage(
                "B", 4000, 12000, {5.0: .5, 10.0: .3, 15.0: .2},
                stage_b_learning_rate, "delayed_csi", 1.0,
            ),
            CurriculumStage(
                "C", 12000, 20000,
                stage_c_probabilities or {5.0: .65, 10.0: .25, 15.0: .1},
                stage_c_learning_rate, "delayed_csi", 1.0,
            ),
        )
        for stage in self.stages:
            if abs(sum(stage.snr_probabilities.values()) - 1) > 1e-9:
                raise ValueError(f"stage {stage.name} SNR probabilities must sum to one")

    def stage(self, step: int) -> CurriculumStage:
        if not 0 <= step < 20000:
            raise ValueError("global step must be in [0,20000)")
        return next(stage for stage in self.stages if stage.start <= step < stage.stop)

    def sample_snr(self, step: int, *, seed: int) -> float:
        stage = self.stage(step)
        generator = random.Random((int(seed) << 20) ^ int(step))
        draw = generator.random()
        cumulative = 0.0
        for snr, probability in stage.snr_probabilities.items():
            cumulative += probability
            if draw <= cumulative:
                return float(snr)
        return float(next(reversed(stage.snr_probabilities)))


@dataclass(frozen=True)
class CleanGate:
    passed: bool
    margins: dict[str, float]
    minimum_margin: float


def clean_gate(metrics: dict[str, float], thresholds: dict[str, float]) -> CleanGate:
    margins = {
        "pure_neural_si_sdr": metrics["pure_neural_si_sdr_delta_db"]
        - thresholds["min_pure_neural_si_sdr_delta_db"],
        "noiseless_r4_si_sdr": metrics["noiseless_r4_si_sdr_delta_db"]
        - thresholds["min_noiseless_r4_si_sdr_delta_db"],
        "clean_stft": thresholds["max_clean_stft_increase"]
        - metrics["clean_stft_increase"],
        "clean_latent": thresholds["max_clean_latent_mse_ratio"]
        - metrics["clean_latent_mse_ratio"],
    }
    minimum = min(margins.values())
    return CleanGate(minimum >= 0, margins, minimum)


class CheckpointSelector:
    FILENAMES = (
        "best_5db_si_sdr.pt",
        "best_clean_gate.pt",
        "best_validation_average.pt",
        "last.pt",
    )

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.best_5db = -math.inf
        self.best_average = -math.inf
        self.best_clean = -math.inf

    def consider(
        self, *, step: int, metrics: dict[str, float], gate: CleanGate
    ) -> list[str]:
        decisions: list[str] = []
        five = float(metrics["5db_delta_si_sdr_vs_initial_r4"])
        average = float(metrics["validation_average_delta_si_sdr_vs_initial_r4"])
        if five > self.best_5db:
            self.best_5db = five
            decisions.append("best_5db_si_sdr.pt")
        if average > self.best_average:
            self.best_average = average
            decisions.append("best_validation_average.pt")
        if gate.passed:
            score = five + 1e-6 * average
            if score > self.best_clean:
                self.best_clean = score
                decisions.append("best_clean_gate.pt")
        return decisions

    def state_dict(self) -> dict[str, float]:
        return {
            "best_5db": self.best_5db,
            "best_average": self.best_average,
            "best_clean": self.best_clean,
        }

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.best_5db = float(state["best_5db"])
        self.best_average = float(state["best_average"])
        self.best_clean = float(state["best_clean"])


def validate_initial_checkpoint_metadata(
    config: dict, expected: dict | None = None
) -> None:
    codec = config.get("codec", {})
    model = config.get("model", {})
    if codec.get("type") != "speechtokenizer" or not codec.get("freeze", False):
        raise ValueError("checkpoint must use a frozen SpeechTokenizer codec")
    shape = (model.get("layers"), model.get("frames"), model.get("latent_dim"))
    if shape != (8, 50, 1024):
        raise ValueError(f"incompatible latent shape {shape}; expected (8,50,1024)")
    architecture_expected = {
        "architecture": "conv_conformer_v1",
        "channel_uses": 1920,
        "channel_state_dim": 8,
        "symbol_frames": 50,
        "temporal_symbol_layout": "balanced_ragged",
    }
    for key, value in architecture_expected.items():
        if model.get(key) != value:
            raise ValueError(f"incompatible {key}: {model.get(key)!r}; expected {value!r}")
    if expected is not None:
        for key in ("config_path", "checkpoint_path"):
            wanted = expected.get(f"codec_{key}")
            if wanted is not None and codec.get(key) != wanted:
                raise ValueError(
                    f"incompatible SpeechTokenizer {key}: {codec.get(key)!r}; "
                    f"expected {wanted!r}"
                )


def freeze_codec_for_input_gradient(codec: nn.Module) -> nn.Module:
    codec.eval()
    codec.requires_grad_(False)
    if any(parameter.requires_grad for parameter in codec.parameters()):
        raise RuntimeError("SpeechTokenizer parameters must remain frozen")
    return codec


def _normalized_layer_loss(reconstruction: Tensor, target: Tensor) -> Tensor:
    return per_layer_nmse(reconstruction, target).mean()


def r4_training_objective(
    reconstruction: Tensor,
    target: Tensor,
    waveform_target: Tensor,
    decode_layers: Callable[[Tensor], Tensor],
    *,
    weights: dict[str, float],
    channel_free_reconstruction: Tensor,
    fft_sizes: tuple[int, ...] = (256, 512, 1024),
    si_sdr_clip_db: float | None = 30.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    latent = _normalized_layer_loss(reconstruction, target)
    decoded = decode_layers(reconstruction)
    stft = multi_resolution_stft_loss(decoded, waveform_target, fft_sizes=fft_sizes)
    waveform = (decoded - waveform_target).abs().mean()
    channel_free = _normalized_layer_loss(channel_free_reconstruction, target)
    components = {
        "latent": latent,
        "stft": stft,
        "waveform": waveform,
        "channel_free": channel_free,
    }
    if "si_sdr" in weights:
        components["si_sdr"] = negative_si_sdr_loss(decoded, waveform_target, clip_db=si_sdr_clip_db)[0]
    total = sum(float(weights.get(name, 0.0)) * value for name, value in components.items())
    return total, components


def component_gradient_norm_for_module(
    component: Tensor, weight: float, module: nn.Module
) -> float:
    if float(weight) == 0:
        return 0.0
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(
        float(weight) * component,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    squared = sum(
        gradient.detach().float().square().sum()
        for gradient in gradients
        if gradient is not None
    )
    return math.sqrt(float(squared))


@dataclass(frozen=True)
class R4ForwardCondition:
    snr_db: float
    tti: int
    tap_coefficients: Tensor
    tap_delay_samples: tuple[int, ...] = (0, 2, 4, 6, 8, 10)
    noise_seed: int = 0
    noiseless: bool = False
    perfect_csi: bool = False
    fixed_mapping: bool = False
    noise_variance_override: float | None = None


@dataclass
class R4WaveformOutput:
    reconstruction: Tensor
    decoded_waveform: Tensor | None
    tx_symbols: Tensor
    mapped_symbols: Tensor
    transmitted_resources: Tensor
    transmitted_time_domain: Tensor
    received_time_domain: Tensor
    received_resources: Tensor
    estimated_channel: Tensor
    true_channel: Tensor
    combined_symbols: Tensor
    decoder_input: Tensor
    decoder_state: Tensor
    csi_nmse: Tensor
    pilot_evm: Tensor
    effective_sinr: Tensor
    mapping_indices: Tensor
    resource_to_source: Tensor
    layer_assignment: Tensor
    realized_snr_db: Tensor
    transmit_power: Tensor
    mrc_output_power: Tensor
    next_delayed_csi: GlobalTripletCSIReport
    allocation: GlobalTripletAllocation
    jammer_grid: Tensor | None = None
    jammer_mask: Tensor | None = None
    signal_received_time: Tensor | None = None
    jammer_received_time: Tensor | None = None
    jammer_statistics: dict | None = None


@dataclass
class R4PhysicalLayerOutput:
    data_grid: Tensor
    tx_grid: Tensor
    pilots: Tensor
    tx_time: Tensor
    faded_time: Tensor
    noise: Tensor
    received_time: Tensor
    received_grid: Tensor
    estimated_channel: Tensor
    true_channel: Tensor
    raw_observations: Tensor
    estimated_channel_source_order: Tensor
    true_channel_source_order: Tensor
    combined: MRCResult
    oracle_combined: MRCResult
    decoder_state: Tensor
    noise_variance: Tensor
    source_power: Tensor
    jammer_grid: Tensor
    jammer_mask: Tensor
    jammer_time: Tensor
    faded_jammer_time: Tensor
    jammer_statistics: dict


def r4_physical_layer_forward(
    source: Tensor,
    allocation: GlobalTripletAllocation,
    taps: Tensor,
    *,
    snr_db: float,
    noise_generator: torch.Generator,
    tap_delay_samples: tuple[int, ...],
    estimator_num_taps: int,
    estimator_ridge_lambda: float,
    epsilon: float,
    noiseless: bool = False,
    perfect_csi: bool = False,
    noise_variance_override: float | Tensor | None = None,
    jammer_type: str = "no_jammer",
    jammer_jsr_db: float | None = None,
    jammer_seed: int = 0,
    jammer_subband_fraction: float = 0.25,
    jammer_burst_fraction: float = 0.25,
    jammer_tone_count: int = 4,
) -> R4PhysicalLayerOutput:
    """Single source of truth for differentiable R4 OFDM, LS-CSI, and MRC."""
    profile = NR_LIKE_R4
    data_grid = allocation.place(source)
    # Kept here (rather than in an evaluator) so jammer and clean runs share
    # exactly the canonical OFDM, LS-estimation, and MRC forward.
    from speech_jscc.evaluation.r4_jammer_baseline import build_r4_jammer
    jammer = build_r4_jammer(
        data_grid, active_grid_masks(profile, device=source.device).candidate_data,
        jammer_type=jammer_type, jsr_db=jammer_jsr_db, seed=jammer_seed,
        subband_fraction=jammer_subband_fraction,
        burst_fraction=jammer_burst_fraction, tone_count=jammer_tone_count,
        epsilon=epsilon,
    )
    tx_grid, pilots = insert_physical_pilots(data_grid, profile)
    tx_time = modulate_tti(tx_grid, profile)
    faded = apply_tti_multipath(tx_time, taps, profile)
    if jammer_type == "no_jammer":
        # Preserve the historical no-jammer arithmetic exactly; the clean
        # anchor is a protocol check, not merely a close numerical comparison.
        jammer_time = torch.zeros_like(tx_time)
        faded_jammer = torch.zeros_like(faded)
    else:
        jammer_time = modulate_tti(jammer.grid, profile)
        faded_jammer = apply_tti_multipath(jammer_time, taps, profile)
    source_power = source.abs().square().mean().detach()
    variance = source_power / (10 ** (float(snr_db) / 10))
    if noise_variance_override is not None:
        variance = source_power.new_tensor(noise_variance_override)
    if noiseless:
        noise = torch.zeros_like(faded)
        variance = source_power.new_tensor(epsilon)
    else:
        noise = torch.complex(
            torch.randn(faded.shape, generator=noise_generator, device=source.device),
            torch.randn(faded.shape, generator=noise_generator, device=source.device),
        ) * torch.sqrt(variance / 2)
    received_time = faded + noise if jammer_type == "no_jammer" else faded + faded_jammer + noise
    received_grid = demodulate_tti(received_time, profile)
    true_fft = torch.fft.fft(taps, n=profile.n_fft)
    true_grid = true_fft[:, list(profile.active_fft_bins), None].expand_as(received_grid)
    estimated = true_grid if perfect_csi else estimate_comb_dft_ls(
        received_grid,
        pilots,
        profile,
        num_taps=estimator_num_taps,
        tap_delay_samples=tap_delay_samples,
        ridge_lambda=estimator_ridge_lambda,
    )
    raw = allocation.extract_source_order(received_grid)
    h_hat = allocation.extract_source_order(estimated)
    h_true = allocation.extract_source_order(true_grid)
    combined = coherent_mrc(
        raw, h_hat, allocation.power_source_order, variance,
        source_power=float(source_power), epsilon=epsilon,
    )
    oracle = coherent_mrc(
        raw, h_true, allocation.power_source_order, variance,
        source_power=float(source_power), epsilon=epsilon,
    )
    masks = active_grid_masks(profile, device=source.device)
    decoder_state = build_observable_receiver_state_v1(
        received_grid, pilots, masks.pilot, estimated
    )
    return R4PhysicalLayerOutput(
        data_grid=data_grid, tx_grid=tx_grid, pilots=pilots, tx_time=tx_time,
        faded_time=faded, noise=noise, received_time=received_time,
        received_grid=received_grid, estimated_channel=estimated,
        true_channel=true_grid, raw_observations=raw,
        estimated_channel_source_order=h_hat,
        true_channel_source_order=h_true, combined=combined,
        oracle_combined=oracle, decoder_state=decoder_state,
        noise_variance=variance, source_power=source_power,
        jammer_grid=jammer.grid, jammer_mask=jammer.mask,
        jammer_time=jammer_time, faded_jammer_time=faded_jammer,
        jammer_statistics=jammer.statistics,
    )


class R4WaveformForward:
    """Shared differentiable R4 physical forward used by training and evaluation."""

    def __init__(
        self,
        codec: nn.Module,
        model: nn.Module,
        *,
        estimator_ridge_lambda: float = 1e-6,
        epsilon: float = 1e-12,
        layer_importance_order: list[int] | None = None,
        minimum_copy_time_separation_symbols: int = 0,
    ):
        self.codec = codec
        self.model = model
        self.profile = NR_LIKE_R4
        self.estimator_ridge_lambda = float(estimator_ridge_lambda)
        self.epsilon = float(epsilon)
        self.importance = layer_importance_order or LAYER_IMPORTANCE
        self.minimum_copy_time_separation_symbols = int(
            minimum_copy_time_separation_symbols
        )
        if self.minimum_copy_time_separation_symbols < 0:
            raise ValueError("minimum copy time separation must be nonnegative")

    def forward(
        self,
        representation: Tensor,
        waveform: Tensor | None = None,
        channel_condition: R4ForwardCondition | None = None,
        delayed_csi: GlobalTripletCSIReport | None = None,
        random_generator: torch.Generator | None = None,
        training: bool = True,
        jammer_type: str = "no_jammer",
        jammer_jsr_db: float | None = None,
        jammer_seed: int = 0,
        jammer_subband_fraction: float = 0.25,
        jammer_burst_fraction: float = 0.25,
        jammer_tone_count: int = 4,
    ) -> R4WaveformOutput:
        if channel_condition is None:
            raise ValueError("an explicit R4 channel condition is required")
        batch = representation.shape[0]
        state = representation.new_zeros((batch, self.model.encoder.channel_state_dim))
        source = self.model.encoder(representation, state)
        report = delayed_csi
        if channel_condition.tti == 0 or channel_condition.fixed_mapping:
            report = None
            allocation_tti = 0
        else:
            allocation_tti = channel_condition.tti
        allocation = allocate_global_balanced_triplets(
            profile=self.profile,
            tx_tti=allocation_tti,
            report=report,
            layer_importance_order=self.importance,
            minimum_time_separation_symbols=self.minimum_copy_time_separation_symbols,
        )
        coefficients = channel_condition.tap_coefficients.to(source.device)
        if coefficients.shape[0] == 1 and batch > 1:
            coefficients = coefficients.expand(batch, -1)
        taps = torch.zeros(
            batch,
            max(channel_condition.tap_delay_samples) + 1,
            dtype=coefficients.dtype,
            device=coefficients.device,
        ).scatter(
            1,
            torch.tensor(
                channel_condition.tap_delay_samples, device=coefficients.device
            )[None].expand(batch, -1),
            coefficients,
        )
        generator = random_generator or torch.Generator(
            device=source.device
        ).manual_seed(channel_condition.noise_seed)
        physical = r4_physical_layer_forward(
            source, allocation, taps,
            snr_db=channel_condition.snr_db,
            noise_generator=generator,
            tap_delay_samples=channel_condition.tap_delay_samples,
            estimator_num_taps=coefficients.shape[-1],
            estimator_ridge_lambda=self.estimator_ridge_lambda,
            epsilon=self.epsilon,
            noiseless=channel_condition.noiseless,
            perfect_csi=channel_condition.perfect_csi,
            noise_variance_override=channel_condition.noise_variance_override,
            jammer_type=jammer_type,
            jammer_jsr_db=jammer_jsr_db,
            jammer_seed=jammer_seed,
            jammer_subband_fraction=jammer_subband_fraction,
            jammer_burst_fraction=jammer_burst_fraction,
            jammer_tone_count=jammer_tone_count,
        )
        reconstruction = self.model.decoder(
            physical.combined.estimate, physical.decoder_state
        )
        decoded = None
        if waveform is not None:
            decoded = decode_frozen_representation_with_gradient(self.codec, reconstruction)
        error = (physical.combined.estimate - source).abs().square().sum().clamp_min(self.epsilon)
        signal = source.abs().square().sum().clamp_min(self.epsilon)
        masks = active_grid_masks(self.profile, device=source.device)
        pilot_residual = (
            physical.received_grid - physical.estimated_channel * physical.pilots
        )[masks.pilot[None].expand(batch, -1, -1)]
        pilot_reference = (
            physical.estimated_channel * physical.pilots
        )[masks.pilot[None].expand(batch, -1, -1)]
        current_reliability = physical.estimated_channel.abs().square().mean(0)[
            masks.candidate_data
        ].detach().cpu()
        next_report = GlobalTripletCSIReport.from_reliability(
            channel_condition.tti, current_reliability
        )
        return R4WaveformOutput(
            reconstruction=reconstruction,
            decoded_waveform=decoded,
            tx_symbols=source,
            mapped_symbols=allocation.extract_source_order(physical.data_grid),
            transmitted_resources=physical.tx_grid,
            transmitted_time_domain=physical.tx_time,
            received_time_domain=physical.received_time,
            received_resources=physical.received_grid,
            estimated_channel=physical.estimated_channel,
            true_channel=physical.true_channel,
            combined_symbols=physical.combined.estimate,
            decoder_input=physical.combined.estimate,
            decoder_state=physical.decoder_state,
            csi_nmse=(physical.estimated_channel - physical.true_channel).abs().square().mean()
            / physical.true_channel.abs().square().mean().clamp_min(self.epsilon),
            pilot_evm=torch.sqrt(
                pilot_residual.abs().square().sum()
                / pilot_reference.abs().square().sum().clamp_min(self.epsilon)
            ),
            effective_sinr=signal / error,
            mapping_indices=allocation.selected_candidate_indices,
            resource_to_source=allocation.resource_to_source,
            layer_assignment=allocation.resource_to_source // 240,
            realized_snr_db=10 * torch.log10(
                physical.source_power
                / physical.noise.abs().square().mean().clamp_min(self.epsilon)
            ),
            transmit_power=allocation.power_source_order.sum(),
            mrc_output_power=physical.combined.estimate.abs().square().mean(),
            next_delayed_csi=next_report,
            allocation=allocation,
            jammer_grid=physical.jammer_grid,
            jammer_mask=physical.jammer_mask,
            signal_received_time=physical.faded_time,
            jammer_received_time=physical.faded_jammer_time,
            jammer_statistics=physical.jammer_statistics,
        )


__all__ = [
    "CheckpointSelector",
    "CleanGate",
    "CurriculumStage",
    "R4Curriculum",
    "R4ForwardCondition",
    "R4PhysicalLayerOutput",
    "R4WaveformForward",
    "R4WaveformOutput",
    "clean_gate",
    "component_gradient_norm_for_module",
    "freeze_codec_for_input_gradient",
    "r4_training_objective",
    "r4_physical_layer_forward",
    "validate_initial_checkpoint_metadata",
]
