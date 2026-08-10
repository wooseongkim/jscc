import pytest
import torch

from evaluate_r4_broadband_uep_profiles import _profiles_from_json
from channels.global_triplet_allocator import GlobalTripletCSIReport, allocate_global_balanced_triplets
from channels.physical_ofdm import NR_LIKE_R4
from channels.r4_uep_allocator import UEP_PROFILES, UEPProfile, allocate_r4_uep
from channels.repetition_mrc import coherent_mrc


ORDER = [1, 0, 2, 5, 3, 4, 6, 7]


def _allocation(name: str):
    report = GlobalTripletCSIReport.from_reliability(
        0, torch.linspace(.01, 4., NR_LIKE_R4.candidate_data_re)
    )
    return allocate_r4_uep(profile=NR_LIKE_R4, tx_tti=1, report=report,
                           layer_importance_order=ORDER, uep_profile=UEP_PROFILES[name])


def test_profiles_preserve_re_budget_and_copy_counts():
    for profile in UEP_PROFILES.values():
        assert sum(profile.repetition) == 24
        assert torch.isclose((profile.normalized_power() * torch.tensor(profile.repetition)).sum() / 24, torch.tensor(1.0))
    allocation = _allocation("RP2")
    expected = torch.repeat_interleave(torch.tensor(UEP_PROFILES["RP2"].repetition), 240)
    assert allocation.selected_re_count == 5760
    assert torch.equal(allocation.copy_count_per_source, expected)
    assert torch.isclose(allocation.power_source_order.sum(), torch.tensor(5760.0), atol=2e-3)
    assert torch.all(allocation.power_source_order[allocation.selected_candidate_indices.ge(0)] > 0)
    source = torch.randn(1, 1920, dtype=torch.complex64)
    recovered = allocation.extract_source_order(allocation.place(source))
    active = allocation.selected_candidate_indices.ge(0)
    expected_values = source[:, None] * allocation.power_source_order.sqrt()[None]
    assert torch.allclose(recovered[active[None]], expected_values[active[None]])


def test_u0_delegates_to_legacy_triplet_mapping():
    u0 = _allocation("U0")
    report = GlobalTripletCSIReport.from_reliability(0, torch.linspace(.01, 4., NR_LIKE_R4.candidate_data_re))
    legacy = allocate_global_balanced_triplets(profile=NR_LIKE_R4, tx_tti=1, report=report, layer_importance_order=ORDER)
    assert u0.selected_candidate_indices.shape == (3, 1920)
    assert torch.equal(u0.selected_candidate_indices, legacy.selected_candidate_indices)
    assert torch.equal(u0.power_source_order, legacy.power_source_order)
    assert torch.equal(u0.power_source_order.gt(0).sum(0), torch.full((1920,), 3))


def test_variable_copy_mrc_handles_missing_copies_and_legacy_three_copy():
    source = torch.randn(1, 1920, dtype=torch.complex64)
    allocation = _allocation("R2")
    h = torch.ones(1, 4, 1920, dtype=torch.complex64)
    observation = source[:, None] * h * allocation.power_source_order.sqrt()[None]
    result = coherent_mrc(observation, h, allocation.power_source_order, 1e-9)
    assert torch.allclose(result.estimate, source, atol=1e-3)


def test_invalid_profiles_are_rejected():
    with pytest.raises(ValueError, match="sum to 24"):
        UEPProfile("bad", (4,) + (3,) * 7, (1,) * 8)
    with pytest.raises(ValueError, match="positive"):
        UEPProfile("bad", (3,) * 8, (1, 1, 1, 1, 1, 1, 1, 0))


def test_profile_json_normalizes_float32_simplex_roundoff(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text('''{"profiles": {"candidate": {
        "repetition": [3, 4, 3, 1, 5, 1, 4, 3],
        "power_share": [0.08763791620731354, 0.2577466368675232,
                        0.2155657857656479, 0.10236416757106781,
                        0.10394956916570663, 0.11150587350130081,
                        0.06598904728889465, 0.055240992456674576]}}}''')
    profile = _profiles_from_json(path)["candidate"]
    assert sum(profile.power_share) == pytest.approx(1.0, abs=1e-12)
