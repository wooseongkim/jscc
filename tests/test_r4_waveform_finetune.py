from __future__ import annotations

import torch

from channels.global_triplet_allocator import (
    GlobalTripletCSIReport,
    allocate_global_balanced_triplets,
)
from channels.physical_ofdm import NR_LIKE_R4
from speech_jscc.training.r4_waveform_finetune import (
    CheckpointSelector,
    R4Curriculum,
    R4ForwardCondition,
    R4WaveformForward,
    clean_gate,
    freeze_codec_for_input_gradient,
    r4_training_objective,
    validate_initial_checkpoint_metadata,
)
from train_r4_waveform_finetune import effective_training_steps, restore_training_state


IMPORTANCE = [1, 0, 2, 5, 3, 4, 6, 7]


def test_curriculum_has_exact_boundaries_probabilities_and_learning_rates():
    curriculum = R4Curriculum(
        stage_a_learning_rate=1e-4,
        stage_b_learning_rate=5e-5,
        stage_c_learning_rate=2e-5,
    )
    assert curriculum.stage(0).name == "A"
    assert curriculum.stage(3999).snr_probabilities == {10.0: 0.6, 15.0: 0.4}
    assert curriculum.stage(4000).name == "B"
    assert curriculum.stage(11999).snr_probabilities == {5.0: 0.5, 10.0: 0.3, 15.0: 0.2}
    assert curriculum.stage(12000).name == "C"
    assert curriculum.stage(19999).snr_probabilities == {5.0: 0.65, 10.0: 0.25, 15.0: 0.1}
    assert [curriculum.stage(step).learning_rate for step in (0, 4000, 12000)] == [
        1e-4,
        5e-5,
        2e-5,
    ]


def test_snr_sampling_is_deterministic_and_stage_a_never_samples_5db():
    curriculum = R4Curriculum()
    first = [curriculum.sample_snr(step, seed=91) for step in range(100)]
    second = [curriculum.sample_snr(step, seed=91) for step in range(100)]
    assert first == second
    assert set(first) <= {10.0, 15.0}


def test_smoke_budget_is_applied_before_long_run_guard():
    assert effective_training_steps(20000, 1, allow_long_run=False) == 1
    try:
        effective_training_steps(20000, None, allow_long_run=False)
    except ValueError as error:
        assert "long" in str(error)
    else:
        raise AssertionError("unapproved long run was accepted")


def test_mapping_round_trip_preserves_source_gradients():
    report = GlobalTripletCSIReport.from_reliability(
        0, torch.linspace(0.1, 2.0, NR_LIKE_R4.candidate_data_re)
    )
    allocation = allocate_global_balanced_triplets(
        profile=NR_LIKE_R4,
        tx_tti=1,
        report=report,
        layer_importance_order=IMPORTANCE,
    )
    source = torch.randn(2, 1920, dtype=torch.complex64, requires_grad=True)
    recovered = allocation.extract_source_order(allocation.place(source))
    recovered.abs().square().mean().backward()
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()
    assert float(source.grad.abs().sum()) > 0


class _Codec(torch.nn.Module):
    representation_shape = (2, 5, 3)
    waveform_samples = 5

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, 1, bias=False)

    def decode_representation(self, representation):
        return self.projection(representation.sum(1)).squeeze(-1)


class _JSCC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 3)
        self.decoder = torch.nn.Linear(3, 3)

    def forward(self, target):
        return self.decoder(self.encoder(target))


def test_waveform_objective_reaches_both_jscc_modules_but_not_codec():
    codec, jscc = _Codec(), _JSCC()
    freeze_codec_for_input_gradient(codec)
    target = torch.randn(2, 2, 5, 3)
    reconstruction = jscc(target)
    reconstruction.retain_grad()
    waveform = codec.decode_representation(target).detach()
    total, components = r4_training_objective(
        reconstruction,
        target,
        waveform,
        codec.decode_representation,
        weights={"latent": 1.0, "stft": 0.1, "waveform": 0.1, "channel_free": 0.2},
        channel_free_reconstruction=jscc(target),
        fft_sizes=(4,),
    )
    total.backward()
    assert reconstruction.grad is not None and float(reconstruction.grad.abs().sum()) > 0
    assert float(jscc.encoder.weight.grad.abs().sum()) > 0
    assert float(jscc.decoder.weight.grad.abs().sum()) > 0
    assert all(parameter.grad is None for parameter in codec.parameters())
    assert set(components) == {"latent", "stft", "waveform", "channel_free"}


def test_clean_checkpoint_is_never_eligible_when_a_constraint_fails(tmp_path):
    thresholds = {
        "min_pure_neural_si_sdr_delta_db": -0.1,
        "min_noiseless_r4_si_sdr_delta_db": -0.1,
        "max_clean_stft_increase": 0.002,
        "max_clean_latent_mse_ratio": 1.05,
    }
    failed = {
        "pure_neural_si_sdr_delta_db": -0.2,
        "noiseless_r4_si_sdr_delta_db": 0.0,
        "clean_stft_increase": 0.0,
        "clean_latent_mse_ratio": 1.0,
    }
    gate = clean_gate(failed, thresholds)
    selector = CheckpointSelector(tmp_path)
    decisions = selector.consider(
        step=250,
        metrics={
            "5db_delta_si_sdr_vs_initial_r4": 0.3,
            "validation_average_delta_si_sdr_vs_initial_r4": 0.2,
        },
        gate=gate,
    )
    assert not gate.passed
    assert "best_clean_gate.pt" not in decisions


def test_checkpoint_metadata_rejects_incompatible_latent_shape():
    config = {
        "codec": {"type": "speechtokenizer", "freeze": True},
        "model": {
            "architecture": "conv_conformer_v1",
            "layers": 8,
            "frames": 49,
            "latent_dim": 1024,
            "channel_uses": 1920,
            "channel_state_dim": 8,
        },
    }
    try:
        validate_initial_checkpoint_metadata(config)
    except ValueError as error:
        assert "latent shape" in str(error)
    else:
        raise AssertionError("incompatible latent metadata was accepted")


def test_checkpoint_metadata_rejects_wrong_codec_identity():
    config = {
        "codec": {
            "type": "speechtokenizer", "freeze": True,
            "config_path": "actual.json", "checkpoint_path": "actual.pt",
        },
        "model": {
            "architecture": "conv_conformer_v1", "layers": 8, "frames": 50,
            "latent_dim": 1024, "channel_uses": 1920, "channel_state_dim": 8,
            "symbol_frames": 50, "temporal_symbol_layout": "balanced_ragged",
        },
    }
    try:
        validate_initial_checkpoint_metadata(
            config, {"codec_checkpoint_path": "different.pt"}
        )
    except ValueError as error:
        assert "SpeechTokenizer checkpoint_path" in str(error)
    else:
        raise AssertionError("wrong codec checkpoint identity was accepted")


def test_resume_restores_step_optimizer_selector_and_validation_manifest(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    selector = CheckpointSelector(tmp_path)
    selector.best_5db = 0.4
    path = tmp_path / "resume.pt"
    torch.save({
        "diagnostic_type": "r4_waveform_finetune",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": 123,
        "curriculum_stage": "A",
        "selector": selector.state_dict(),
        "validation_manifest": {"paths": ["a"], "light_seeds": [1], "full_seeds": [2]},
        "delayed_csi": None,
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": __import__("random").getstate(),
    }, path)
    restored = restore_training_state(
        path, model=model, optimizer=optimizer,
        selector=CheckpointSelector(tmp_path),
        validation_manifest={"paths": ["a"], "light_seeds": [1], "full_seeds": [2]},
        device=torch.device("cpu"),
    )
    assert restored["global_step"] == 123
    assert restored["curriculum_stage"] == "A"
    assert restored["selector"].best_5db == 0.4


class _PhysicalEncoder(torch.nn.Module):
    channel_state_dim = 8

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, representation, state):
        flat = representation.flatten(1)
        return torch.complex(flat[:, :1920], flat[:, 1920:3840]) * self.scale


class _PhysicalDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, symbols, state):
        values = torch.cat((symbols.real, symbols.imag), dim=1)
        repeats = (8 * 50 * 1024 + values.shape[1] - 1) // values.shape[1]
        return values.repeat(1, repeats)[:, : 8 * 50 * 1024].reshape(
            symbols.shape[0], 8, 50, 1024
        ) * self.scale


class _PhysicalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _PhysicalEncoder()
        self.decoder = _PhysicalDecoder()


class _PhysicalCodec(torch.nn.Module):
    representation_shape = (8, 50, 1024)
    waveform_samples = 50

    def decode_representation(self, representation):
        return representation.sum(1).mean(-1)


def test_shared_ls_r4_forward_is_deterministic_finite_and_differentiable():
    codec, model = _PhysicalCodec(), _PhysicalModel()
    freeze_codec_for_input_gradient(codec)
    engine = R4WaveformForward(codec, model)
    representation = torch.randn(1, 8, 50, 1024)
    waveform = codec.decode_representation(representation)
    condition = R4ForwardCondition(
        snr_db=10,
        tti=0,
        tap_coefficients=torch.tensor(
            [[1 + 0j, .3 + .1j, .2 - .1j, .1 + 0j, .05j, .02 + 0j]]
        ),
        noise_seed=19,
    )
    first = engine.forward(representation, waveform, condition, training=True)
    second = engine.forward(representation, waveform, condition, training=True)
    assert torch.equal(first.combined_symbols, second.combined_symbols)
    assert first.decoder_input.shape == (1, 1920)
    assert first.reconstruction.shape == (1, 8, 50, 1024)
    assert first.mapping_indices.numel() == 5760
    assert torch.isfinite(first.csi_nmse)
    assert torch.isfinite(first.pilot_evm)
    assert torch.isfinite(first.effective_sinr)
    loss = (first.decoded_waveform - waveform).abs().mean()
    loss.backward()
    assert float(model.encoder.scale.grad.abs()) > 0
    assert float(model.decoder.scale.grad.abs()) > 0
    assert all(parameter.grad is None for parameter in codec.parameters())
