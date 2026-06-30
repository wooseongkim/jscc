import torch

from eval_jamming import (
    LearnedLayerGate,
    available_adaptation_modes,
    rule_based_layer_gates,
)


def test_rule_based_gating_activates_layer_prefixes():
    quality = torch.tensor([-3.0, 4.0, 9.0, 20.0])
    state = torch.zeros(4, 8)
    state[:, 0] = quality / 20.0
    gates = rule_based_layer_gates(state, 4, [0.0, 6.0, 12.0])
    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    torch.testing.assert_close(gates, expected)


def test_learned_gate_shape_and_range():
    gate = LearnedLayerGate(channel_state_dim=3, num_codec_layers=5, hidden_dim=8)
    values = gate(torch.randn(6, 3))

    assert values.shape == (6, 5)
    assert torch.all((values >= 0) & (values <= 1))


def test_learned_mode_is_only_used_when_available():
    requested = ["uniform", "rule_based", "learned"]
    assert available_adaptation_modes(requested, None) == ["uniform", "rule_based"]

    gate = LearnedLayerGate(channel_state_dim=2, num_codec_layers=4)
    assert available_adaptation_modes(requested, gate) == requested
