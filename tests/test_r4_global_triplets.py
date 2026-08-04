import torch

from channels.global_triplet_allocator import (
    GlobalTripletCSIReport,
    allocate_global_balanced_triplets,
)
from channels.physical_ofdm import (
    NR_LIKE_R3,
    NR_LIKE_R4,
    active_grid_masks,
    apply_tti_multipath,
    demodulate_tti,
    modulate_tti,
)
from channels.temporal_multipath import (
    delay_samples_for_rate,
    expand_taps_to_sample_delays,
)


IMPORTANCE = [1, 0, 2, 5, 3, 4, 6, 7]


def test_r4_profile_and_masks() -> None:
    p = NR_LIKE_R4
    masks = active_grid_masks(p)
    assert (p.n_fft, p.sample_rate_hz, p.active_subcarriers) == (512, 15.36e6, 360)
    assert p.occupied_bandwidth_hz == 10.8e6
    assert p.cp_samples == 36
    assert abs(p.tti_duration_s - 0.0009989583333333334) < 1e-12
    assert (p.total_active_re, int(masks.pilot.sum()), int(masks.candidate_data.sum())) == (
        10080,
        360,
        9720,
    )
    assert 0 not in p.active_fft_bins
    assert set(p.active_fft_bins).isdisjoint(p.guard_fft_bins)


def test_physical_delays_preserved_across_profiles() -> None:
    delays_s = tuple(index / NR_LIKE_R3.sample_rate_hz for index in range(6))
    assert delay_samples_for_rate(delays_s, NR_LIKE_R3.sample_rate_hz) == (0, 1, 2, 3, 4, 5)
    r4 = delay_samples_for_rate(delays_s, NR_LIKE_R4.sample_rate_hz)
    assert r4 == (0, 2, 4, 6, 8, 10)
    taps = torch.arange(1, 7, dtype=torch.float32).to(torch.complex64)[None]
    expanded = expand_taps_to_sample_delays(taps, r4)
    assert expanded.shape == (1, 11)
    assert torch.equal(expanded[:, list(r4)], taps)
    assert torch.count_nonzero(expanded).item() == 6
    assert max(r4) < NR_LIKE_R4.cp_samples


def test_r4_ofdm_identity_and_sparse_channel_reference() -> None:
    generator = torch.Generator().manual_seed(4)
    grid = torch.complex(
        torch.randn(1, 360, 28, generator=generator),
        torch.randn(1, 360, 28, generator=generator),
    )
    waveform = modulate_tti(grid, NR_LIKE_R4)
    assert (demodulate_tti(waveform, NR_LIKE_R4) - grid).abs().max() < 1e-6
    delays = (0, 2, 4, 6, 8, 10)
    coeff = torch.complex(
        torch.randn(1, 6, generator=generator),
        torch.randn(1, 6, generator=generator),
    ) / 4
    taps = expand_taps_to_sample_delays(coeff, delays)
    received = demodulate_tti(apply_tti_multipath(waveform, taps, NR_LIKE_R4), NR_LIKE_R4)
    bins = torch.tensor(NR_LIKE_R4.active_fft_bins)
    delay_tensor = torch.tensor(delays)
    design = torch.exp(
        -2j * torch.pi * bins[:, None] * delay_tensor[None] / NR_LIKE_R4.n_fft
    )
    response = coeff @ design.T
    assert (received - grid * response[..., None]).abs().max() < 3e-5


def _allocation(tti: int = 1):
    reliability = torch.linspace(0.01, 4.0, NR_LIKE_R4.candidate_data_re)
    report = None if tti == 0 else GlobalTripletCSIReport.from_reliability(0, reliability)
    return allocate_global_balanced_triplets(
        profile=NR_LIKE_R4,
        tx_tti=tti,
        report=report,
        layer_importance_order=IMPORTANCE,
        min_selected_re_per_subcarrier=8,
        max_selected_re_per_subcarrier=24,
        minimum_frequency_separation_subcarriers=60,
        q_min=0.5,
        q_max=2.0,
        branch_min_fraction=0.15,
    )


def _time_interleaved_allocation(tti: int = 1):
    reliability = torch.linspace(0.01, 4.0, NR_LIKE_R4.candidate_data_re)
    report = None if tti == 0 else GlobalTripletCSIReport.from_reliability(0, reliability)
    return allocate_global_balanced_triplets(
        profile=NR_LIKE_R4,
        tx_tti=tti,
        report=report,
        layer_importance_order=IMPORTANCE,
        min_selected_re_per_subcarrier=8,
        max_selected_re_per_subcarrier=24,
        minimum_frequency_separation_subcarriers=60,
        minimum_time_separation_symbols=7,
        q_min=0.5,
        q_max=2.0,
        branch_min_fraction=0.15,
    )


def test_global_triplet_mapping_counts_separation_and_inverse() -> None:
    allocation = _allocation()
    selected = allocation.selected_candidate_indices
    assert selected.shape == (3, 1920)
    assert torch.unique(selected).numel() == 5760
    assert allocation.unused_candidate_re == 3960
    coords = active_grid_masks(NR_LIKE_R4).candidate_data.nonzero()[selected]
    frequencies = coords[..., 0]
    assert torch.all(frequencies[0] != frequencies[1])
    assert torch.all(frequencies[0] != frequencies[2])
    assert torch.all(frequencies[1] != frequencies[2])
    pairwise = torch.stack(
        [
            (frequencies[0] - frequencies[1]).abs(),
            (frequencies[0] - frequencies[2]).abs(),
            (frequencies[1] - frequencies[2]).abs(),
        ]
    )
    assert int(pairwise.min()) >= 60
    counts = torch.bincount(frequencies.flatten(), minlength=360)
    assert int(counts.min()) >= 8 and int(counts.max()) <= 24
    assert torch.unique(counts).numel() > 1
    assert torch.unique(allocation.resource_to_source).numel() == 1920
    source = torch.complex(torch.arange(1920.0)[None], torch.zeros(1, 1920))
    recovered = allocation.extract_source_order(allocation.place(source))
    expected = source[:, None, :] * allocation.power_source_order.sqrt()[None]
    assert torch.equal(recovered, expected)
    layer_counts = torch.bincount(allocation.resource_to_source // 240, minlength=8)
    assert torch.equal(layer_counts, torch.full((8,), 240))


def test_bootstrap_is_deterministic_and_report_is_causal() -> None:
    first = _allocation(0)
    second = _allocation(0)
    assert torch.equal(first.selected_candidate_indices, second.selected_candidate_indices)
    reliability = torch.ones(NR_LIKE_R4.candidate_data_re)
    current = GlobalTripletCSIReport.from_reliability(1, reliability)
    try:
        allocate_global_balanced_triplets(
            profile=NR_LIKE_R4,
            tx_tti=1,
            report=current,
            layer_importance_order=IMPORTANCE,
        )
    except ValueError as error:
        assert "causal" in str(error)
    else:
        raise AssertionError("same-TTI CSI leakage was accepted")


def test_time_interleaved_triplets_enforce_minimum_copy_time_separation() -> None:
    allocation = _time_interleaved_allocation()
    coordinates = active_grid_masks(NR_LIKE_R4).candidate_data.nonzero()[
        allocation.selected_candidate_indices
    ]
    times = coordinates[..., 1]
    pairwise = torch.stack(
        [
            (times[0] - times[1]).abs(),
            (times[0] - times[2]).abs(),
            (times[1] - times[2]).abs(),
        ]
    )
    # The OFDM frame is not cyclic: copies must be separated in actual symbol time.
    assert int(pairwise.min()) >= 7
    assert allocation.minimum_time_separation_symbols == 7
    assert int(allocation.time_separation_levels.min()) >= 7
    frequencies = coordinates[..., 0]
    frequency_pairwise = torch.stack(
        [
            (frequencies[0] - frequencies[1]).abs(),
            (frequencies[0] - frequencies[2]).abs(),
            (frequencies[1] - frequencies[2]).abs(),
        ]
    )
    assert int(frequency_pairwise.min()) >= 60
    assert torch.isclose(allocation.power_source_order.sum(), torch.tensor(5760.0), atol=1e-3)


def test_time_interleaved_triplets_are_deterministic_and_default_is_unchanged() -> None:
    baseline = _allocation()
    explicit_default = allocate_global_balanced_triplets(
        profile=NR_LIKE_R4,
        tx_tti=1,
        report=GlobalTripletCSIReport.from_reliability(
            0, torch.linspace(0.01, 4.0, NR_LIKE_R4.candidate_data_re)
        ),
        layer_importance_order=IMPORTANCE,
        min_selected_re_per_subcarrier=8,
        max_selected_re_per_subcarrier=24,
        minimum_frequency_separation_subcarriers=60,
        minimum_time_separation_symbols=0,
        q_min=0.5,
        q_max=2.0,
        branch_min_fraction=0.15,
    )
    assert torch.equal(baseline.selected_candidate_indices, explicit_default.selected_candidate_indices)
    first = _time_interleaved_allocation()
    second = _time_interleaved_allocation()
    assert torch.equal(first.selected_candidate_indices, second.selected_candidate_indices)
    assert torch.equal(first.resource_to_source, second.resource_to_source)


def test_triplet_and_branch_power_constraints() -> None:
    allocation = _allocation()
    q = allocation.triplet_power_multiplier
    fractions = allocation.branch_power_fractions
    powers = allocation.power_source_order
    assert float(q.min()) >= 0.5 - 1e-6
    assert float(q.max()) <= 2.0 + 1e-6
    assert torch.isclose(q.mean(), torch.tensor(1.0), atol=1e-6)
    assert float(fractions.min()) >= 0.15 - 1e-6
    assert torch.allclose(fractions.sum(0), torch.ones(1920), atol=1e-6)
    assert torch.allclose(powers.sum(0), 3 * q, atol=1e-5)
    assert torch.isclose(powers.sum(), torch.tensor(5760.0), atol=1e-3)
    order = torch.argsort(allocation.predicted_triplet_gain)
    assert float(q[order[:100]].mean()) >= float(q[order[-100:]].mean())
