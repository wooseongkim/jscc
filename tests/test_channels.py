import pytest
import torch

from channels.jammer import (
    compute_jsr,
    jammer_mask_statistics,
    make_jammer,
    make_jammer_mask,
)
from channels.rayleigh import compute_effective_sinr, rayleigh_channel


def unit_power(shape):
    value = torch.complex(torch.randn(shape), torch.randn(shape))
    dimensions = tuple(range(1, value.ndim))
    return value / value.abs().square().mean(dimensions, keepdim=True).sqrt()


def test_flat_rayleigh_shapes_and_signal_power():
    signal = unit_power((4, 128))
    jammer = torch.zeros_like(signal)
    fading = torch.ones((4, 1), dtype=signal.dtype)
    result = rayleigh_channel(
        signal,
        jammer,
        torch.full((4,), 12.0),
        fading="flat",
        signal_fading=fading,
        jammer_fading=fading,
    )

    torch.testing.assert_close(signal.abs().square().mean(1), torch.ones(4))
    torch.testing.assert_close(result["faded_signal"], signal)
    assert result["received"].shape == signal.shape
    assert result["equalized"].shape == signal.shape
    assert result["noise"].shape == signal.shape
    assert result["signal_fading"].shape == (4, 1)
    assert result["effective_sinr"].shape == (4,)
    assert result["received"].is_complex()


def test_ofdm_grid_shapes_and_pilot_jamming():
    signal = unit_power((3, 16, 10))
    pilot_mask = torch.zeros((16, 10), dtype=torch.bool)
    pilot_mask[::4, ::2] = True
    jammer, mask = make_jammer(signal, 0.0, "pilot", pilot_mask=pilot_mask)
    result = rayleigh_channel(signal, jammer, 10.0, fading="ofdm")

    assert mask.shape == signal.shape
    assert torch.equal(mask[0], pilot_mask)
    assert torch.all(jammer[~mask] == 0)
    assert result["received"].shape == signal.shape
    assert result["signal_fading"].shape == signal.shape
    assert result["jammer_fading"].shape == signal.shape
    assert result["effective_sinr"].shape == (3,)


@pytest.mark.parametrize("kind", ["barrage", "narrowband", "burst", "pilot"])
@pytest.mark.parametrize("shape", [(4, 64), (4, 12, 8)])
def test_jammer_power_and_mask_statistics(kind, shape):
    signal = unit_power(shape)
    requested_jsr_db = torch.tensor([-9.0, -3.0, 0.0, 6.0])
    jammer, mask = make_jammer(signal, requested_jsr_db, kind, jammed_fraction=0.25)
    measured_jsr_db = compute_jsr(signal, jammer, db=True)
    statistics = jammer_mask_statistics(mask)

    torch.testing.assert_close(measured_jsr_db, requested_jsr_db, rtol=1e-5, atol=1e-5)
    assert jammer.shape == signal.shape
    assert mask.shape == signal.shape
    assert mask.dtype == torch.bool
    assert torch.all(statistics["active_count"] > 0)
    assert torch.all((statistics["mask_ratio"] > 0) & (statistics["mask_ratio"] <= 1))


def test_narrowband_and_burst_mask_axes_on_ofdm_grid():
    narrowband = make_jammer_mask((2, 20, 12), "narrowband", 0.25)
    burst = make_jammer_mask((2, 20, 12), "burst", 0.25)

    assert torch.all(narrowband.any(dim=2) == narrowband.all(dim=2))
    assert torch.all(burst.any(dim=1) == burst.all(dim=1))
    assert torch.all(narrowband.sum(dim=(1, 2)) == 5 * 12)
    assert torch.all(burst.sum(dim=(1, 2)) == 20 * 3)


def test_effective_sinr_from_controlled_components():
    signal = torch.ones((2, 32), dtype=torch.complex64)
    jammer = torch.full_like(signal, 0.5)
    noise = torch.full_like(signal, 0.5j)
    expected = torch.full((2,), 2.0)

    torch.testing.assert_close(compute_effective_sinr(signal, jammer, noise), expected)
    torch.testing.assert_close(
        compute_effective_sinr(signal, jammer, noise, db=True),
        10.0 * torch.log10(expected),
    )

