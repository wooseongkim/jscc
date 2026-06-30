import torch

from eval_jamming import _load_checkpoint, evaluate_paired_condition
from evaluation.paired import generate_paired_evaluation_batch
from models.channel_state import (
    CHANNEL_STATE_DIM,
    JAMMER_POSTERIOR_SLICE,
    MASK_RATIO_INDEX,
    build_channel_state,
    rule_based_jammer_posterior,
)
from models.learned_gate import (
    LearnedLayerGate,
    load_learned_gate_checkpoint,
    save_learned_gate_checkpoint,
)
from models.latent_refiner import LatentRefiner
from speech_jscc.codecs import MockContinuousCodec
from speech_jscc.experiment import build_components
from speech_jscc.models import SpeechJSCC
from train_latent_jscc import joint_learned_gate_step


def small_paired_batch(codec):
    return generate_paired_evaluation_batch(
        codec,
        batch_size=4,
        waveform_samples=96,
        channel_shape=(18,),
        snr_db=8.0,
        jsr_db=-3.0,
        jammer_type="narrowband",
        jammed_fraction=0.25,
        pilot_spacing=3,
        pilot_time_spacing=None,
        target_power=1.0,
        seed=314,
        device=torch.device("cpu"),
        fading="flat",
    )


def test_channel_state_schema_contains_posterior_and_mask_ratio():
    posterior = rule_based_jammer_posterior(
        "burst", 3, device=torch.device("cpu")
    )
    state = build_channel_state(
        torch.tensor([1.0, 2.0, 4.0]),
        torch.tensor([0.1, 0.2, 0.3]),
        torch.tensor([0.01, 0.02, 0.03]),
        posterior,
        torch.tensor([0.25, 0.5, 0.75]),
    )

    assert state.shape == (3, CHANNEL_STATE_DIM)
    torch.testing.assert_close(state[:, JAMMER_POSTERIOR_SLICE], posterior)
    torch.testing.assert_close(state[:, MASK_RATIO_INDEX], torch.tensor([0.25, 0.5, 0.75]))


def test_joint_training_updates_learned_gate_parameters():
    codec = MockContinuousCodec(3, 4, 2, 96, seed=5)
    model = SpeechJSCC((3, 4, 2), 18, channel_state_dim=CHANNEL_STATE_DIM, hidden_dim=24)
    gate = LearnedLayerGate(CHANNEL_STATE_DIM, 3, hidden_dim=12)
    refiner = LatentRefiner((3, 4, 2), CHANNEL_STATE_DIM, hidden_dim=12)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(gate.parameters()) + list(refiner.parameters()), lr=1e-2
    )
    before = [parameter.detach().clone() for parameter in gate.parameters()]

    result = joint_learned_gate_step(
        codec,
        model,
        gate,
        refiner,
        small_paired_batch(codec),
        optimizer,
        torch.ones(3),
        lambda_budget=0.01,
        lambda_smooth=0.01,
        lambda_refine=1.0,
        power_penalty_weight=0.01,
    )

    assert result["alpha"].shape == (4, 3)
    assert torch.all((result["alpha"] >= 0) & (result["alpha"] <= 1))
    assert any(not torch.equal(old, new) for old, new in zip(before, gate.parameters()))


def test_learned_gate_checkpoint_round_trip(tmp_path):
    gate = LearnedLayerGate(CHANNEL_STATE_DIM, 4, hidden_dim=10)
    state = torch.randn(5, CHANNEL_STATE_DIM)
    expected = gate(state)
    path = tmp_path / "gate.pt"
    torch.save(save_learned_gate_checkpoint(gate), path)

    payload = torch.load(path, weights_only=True)
    restored = load_learned_gate_checkpoint(payload, torch.device("cpu"))

    torch.testing.assert_close(restored(state), expected)


def test_paired_evaluation_runs_all_three_modes_from_checkpoint(tmp_path):
    config = {
        "seed": 2,
        "device": "cpu",
        "model": {
            "layers": 3,
            "frames": 4,
            "latent_dim": 2,
            "channel_uses": 18,
            "channel_state_dim": CHANNEL_STATE_DIM,
            "hidden_dim": 24,
            "target_power": 1.0,
        },
        "codec": {"waveform_samples": 96},
        "channel": {
            "jammer_types": ["burst"],
            "snr_db": [5.0],
            "jsr_db": [0.0],
            "jammed_fraction": 0.25,
            "pilot_spacing": 3,
            "pilot_time_spacing": 3,
        },
        "eval": {
            "batches": 1,
            "batch_size": 3,
            "layer_weights": [1.0, 1.0, 1.0],
            "rule_gate_thresholds_db": [0.0, 6.0],
            "learned_gate_hidden_dim": 10,
            "transmitter_csi": True,
        },
    }
    codec, model = build_components(config, torch.device("cpu"))
    gate = LearnedLayerGate(CHANNEL_STATE_DIM, 3, hidden_dim=10)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {"model": model.state_dict(), "learned_gate": save_learned_gate_checkpoint(gate)},
        checkpoint,
    )
    loaded_gate = _load_checkpoint(checkpoint, model, config, torch.device("cpu"))

    rows = evaluate_paired_condition(
        codec,
        model,
        loaded_gate,
        config,
        torch.device("cpu"),
        ["uniform", "rule_based", "learned_gate"],
        "burst",
        5.0,
        0.0,
        "estimated",
        123,
    )

    assert {row["adaptation_mode"] for row in rows} == {
        "uniform",
        "rule_based",
        "learned_gate",
    }
    assert {row["paired_seed"] for row in rows} == {123}
    assert all("alpha_1" in row and "decoder_c_7" in row for row in rows)
