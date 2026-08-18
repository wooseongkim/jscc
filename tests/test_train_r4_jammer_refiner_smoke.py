import torch
from torch import nn

from models.adaptive_latent_refiner import MoEAdaptiveLatentRefiner
from models.jammer_estimator import JAMMER_TYPE_CLASSES, JammerEstimator
from speech_jscc.training.train_r4_jammer_refiner import (
    PhaseSpec,
    build_phase_schedule,
    build_refiner_optimizer,
    canonical_jammer_type,
    sample_jammer_condition,
    _write_validation_csv,
    TRAINER_JAMMER_TYPE_CLASSES,
)


def test_narrowband_alias_is_normalized_to_physical_subband():
    assert canonical_jammer_type("narrowband") == "subband"
    assert canonical_jammer_type("broadband_awgn") == "broadband_awgn"


def test_trainer_uses_only_approved_five_class_vocabulary():
    assert TRAINER_JAMMER_TYPE_CLASSES == (
        "no_jammer", "broadband_awgn", "subband", "burst", "tone"
    )


def test_phase_schedule_is_contiguous_and_has_requested_phases():
    phases = build_phase_schedule(
        {
            "estimator_pretrain_steps": 2,
            "oracle_mask_refiner_steps": 3,
            "learned_mask_moe_steps": 4,
            "estimator_lr": 1e-4,
            "refiner_lr": 2e-4,
        }
    )
    assert [phase.name for phase in phases] == [
        "estimator_pretrain", "oracle_mask_refiner", "learned_mask_moe"
    ]
    assert [(phase.start, phase.stop) for phase in phases] == [(0, 2), (2, 5), (5, 9)]


def test_default_optimizer_excludes_jscc_encoder_and_decoder():
    model = nn.Module()
    model.encoder = nn.Linear(3, 3)
    model.decoder = nn.Linear(3, 3)
    estimator = JammerEstimator(hidden_dim=8)
    refiner = MoEAdaptiveLatentRefiner(
        representation_shape=(2, 3, 4), channel_state_dim=8,
        num_experts=len(JAMMER_TYPE_CLASSES), hidden_dim=8,
    )
    phase = PhaseSpec("learned_mask_moe", 0, 1, 1e-4, 1e-4)
    optimizer = build_refiner_optimizer(
        estimator, refiner, model, {"weight_decay": 1e-5, "unfreeze_jscc_decoder": False}, phase
    )
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    jscc = {id(parameter) for parameter in model.parameters()}
    assert optimized.isdisjoint(jscc)


def test_jammer_sampling_is_seed_deterministic_and_uses_supported_taxonomy():
    config = {
        "jammer_types": ["no_jammer", "broadband_awgn", "narrowband", "burst", "tone"],
        "jammer_probabilities": {"no_jammer": .2, "broadband_awgn": .25, "narrowband": .25, "burst": .2, "tone": .1},
        "snr_db_choices": [5, 10, 15],
        "jsr_db_choices": [0, 5, 10],
    }
    first = sample_jammer_condition(config, step=9, seed=1234)
    second = sample_jammer_condition(config, step=9, seed=1234)
    assert first == second
    assert first["jammer_type"] in JAMMER_TYPE_CLASSES


def test_validation_csv_preserves_phase_and_global_step_columns(tmp_path):
    path = tmp_path / "validation.csv"
    _write_validation_csv(path, [{
        "global_step": 2, "phase": "estimator_pretrain", "jammer_type": "no_jammer",
        "type_accuracy": 1.0, "mask_iou": 1.0, "mask_f1": 1.0,
        "mask_bce": 0.0, "latent_nmse": 0.0, "no_jammer_degradation": 0.0,
        "si_sdr_db": 1.0, "raw_si_sdr_db": 0.5, "si_sdr_delta_vs_raw_db": 0.5,
    }])
    columns = path.read_text().splitlines()[0]
    assert "global_step" in columns
    assert "si_sdr_db" in columns
    assert "raw_si_sdr_db" in columns
    assert "si_sdr_delta_vs_raw_db" in columns
