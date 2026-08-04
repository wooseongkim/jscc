import torch

from speech_jscc.evaluation.r4_layer_ablation import (
    distribution,
    normalized_weights,
    replace_layers,
)


def test_replace_layers_changes_only_requested_layer():
    value = torch.arange(2 * 8 * 3 * 4, dtype=torch.float32).reshape(2, 8, 3, 4)
    replaced = replace_layers(value, [2], "zero")
    for layer in range(8):
        if layer == 2:
            assert torch.count_nonzero(replaced[:, layer]) == 0
        else:
            torch.testing.assert_close(replaced[:, layer], value[:, layer])


def test_mean_replacement_is_distinct_and_shape_preserving():
    value = torch.randn(1, 8, 5, 7)
    replaced = replace_layers(value, [1], "mean")
    assert replaced.shape == value.shape
    assert torch.equal(replaced[:, 0], value[:, 0])
    assert not torch.equal(replaced[:, 1], value[:, 1])


def test_weights_and_bootstrap_are_deterministic():
    values = [1.0, 2.0, 3.0]
    assert sum(normalized_weights(values)) == 1.0
    assert normalized_weights(values) == normalized_weights(values)
    first = distribution(values, bootstrap_samples=100, bootstrap_seed=7)
    second = distribution(values, bootstrap_samples=100, bootstrap_seed=7)
    assert first == second
