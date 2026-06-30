import torch

from channels.reliability import compute_resource_reliability
from evaluation.paired import (
    estimate_transmitter_feedback,
    generate_paired_evaluation_batch,
    run_mode_on_paired_batch,
)
from models.channel_state import CHANNEL_STATE_DIM
from models.latent_refiner import LatentRefiner
from models.resource_allocator import allocate_resources
from speech_jscc.codecs import MockContinuousCodec
from speech_jscc.models import SpeechJSCC


def test_reliability_decreases_when_jammer_power_increases():
    channel = torch.ones(2, 4, 5, dtype=torch.complex64)
    noise_power = torch.full((2,), 0.1)
    low_jammer = compute_resource_reliability(channel, 0.1, noise_power, 0.9)
    high_jammer = compute_resource_reliability(channel, 2.0, noise_power, 0.9)

    assert torch.all(high_jammer < low_jammer)


def test_greedy_allocation_gives_important_layers_more_reliable_resources():
    symbols = torch.complex(torch.arange(8, dtype=torch.float32), torch.zeros(8)).reshape(1, 2, 4)
    reliability = torch.tensor([[[8.0, 1.0, 7.0, 2.0], [6.0, 3.0, 5.0, 4.0]]])
    allocation = allocate_resources(
        symbols,
        reliability,
        layer_channel_uses=(4, 4),
        mode="reliability_greedy",
        importance_order=[0, 1],
    )

    important = reliability[allocation.layer_assignment == 0].mean()
    detail = reliability[allocation.layer_assignment == 1].mean()
    assert important > detail


def test_paired_allocation_modes_share_channel_jammer_and_noise():
    codec = MockContinuousCodec(3, 4, 2, 96, seed=8)
    model = SpeechJSCC(
        (3, 4, 2), (3, 6), channel_state_dim=CHANNEL_STATE_DIM, hidden_dim=24
    )
    batch = generate_paired_evaluation_batch(
        codec,
        batch_size=3,
        waveform_samples=96,
        channel_shape=(3, 6),
        snr_db=7.0,
        jsr_db=0.0,
        jammer_type="narrowband",
        jammed_fraction=0.33,
        pilot_spacing=2,
        pilot_time_spacing=3,
        target_power=1.0,
        seed=901,
        device=torch.device("cpu"),
        fading="ofdm",
    )
    feedback = estimate_transmitter_feedback(batch, fading="ofdm")
    gates = torch.ones(3, 3)
    results = {
        mode: run_mode_on_paired_batch(
            codec,
            model,
            batch,
            feedback["state"],
            gates,
            equalizer="estimated",
            fading="ofdm",
            allocation_mode=mode,
            importance_order=[0, 1, 2],
            resource_reliability=feedback["reliability"],
        )
        for mode in ("uniform", "random", "reliability_greedy")
    }

    reference = results["uniform"]
    for mode in ("random", "reliability_greedy"):
        torch.testing.assert_close(results[mode]["noise"], reference["noise"], rtol=0, atol=0)
        torch.testing.assert_close(results[mode]["jammer"], reference["jammer"], rtol=0, atol=0)
        torch.testing.assert_close(
            results[mode]["signal_fading"], reference["signal_fading"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            results[mode]["jammer_fading"], reference["jammer_fading"], rtol=0, atol=0
        )


def test_refiner_improves_controlled_corruption_without_catastrophic_degradation():
    torch.manual_seed(12)
    shape = (2, 8, 3)
    clean = torch.randn(6, *shape)
    resource_mask = torch.zeros(6, 16, dtype=torch.bool)
    resource_mask[:, 4:12] = True
    corruption = torch.zeros_like(clean)
    corruption[:, :, 2:6] = 0.5
    noisy = clean + corruption
    state = torch.zeros(6, CHANNEL_STATE_DIM)
    state[:, -1] = resource_mask.float().mean(dim=1)
    refiner = LatentRefiner(shape, CHANNEL_STATE_DIM, hidden_dim=24)
    optimizer = torch.optim.Adam(refiner.parameters(), lr=2e-2)
    initial_mse = (noisy - clean).square().mean()

    for _ in range(25):
        refined = refiner(noisy, state, resource_mask)
        loss = (refined - clean).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_mse = (refiner(noisy, state, resource_mask) - clean).square().mean()
    assert final_mse <= initial_mse

