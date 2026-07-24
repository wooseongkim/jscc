import math
import torch
import pytest
from channels.jammer import make_jammer
from channels.pilot import make_pilot_mask
from speech_jscc.diagnostics.j4_burst import active_window_jsr_db, burst_diagnostics, j4_policy


def test_burst_is_one_nonwrapping_full_band_time_interval():
    reference=torch.ones(3,64,32,dtype=torch.complex64)
    jammer,mask=make_jammer(reference,0.,"burst",.25,generator=torch.Generator().manual_seed(4))
    for item in mask:
        active=torch.where(item.any(dim=0))[0]
        assert active.numel()==8
        assert torch.equal(active,torch.arange(active[0],active[0]+8))
        assert item[:,active].all() and not item[:,~item.any(dim=0)].any()
    assert jammer[~mask].abs().max()==0


def test_burst_diagnostics_separate_global_and_active_jsr_and_layer_overlap():
    reference=torch.ones(1,64,32,dtype=torch.complex64);pilots=make_pilot_mask(reference.shape,4,time_spacing=4)
    jammer,mask=make_jammer(reference,-3.,"burst",.25,generator=torch.Generator().manual_seed(2))
    value=burst_diagnostics(reference,jammer,mask,pilots,requested_fraction=.25,requested_global_jsr_db=-3.)
    assert value["actual_burst_fraction"]==.25 and value["burst_symbol_count"]==8
    assert value["active_window_jsr_db"]==pytest.approx(-3-10*math.log10(.25))
    assert value["full_band_inside_burst_verified"] and value["contiguous_burst_verified"]
    assert value["leakage_power_outside_burst"]==0
    assert len(value["per_layer_burst_data_fraction"])==8


def test_j4_policy_is_reproducible_and_covers_durations():
    a=[j4_policy(23,i,[5,15],[-10,0],[.125,.25,.5]) for i in range(1,30)]
    assert a==[j4_policy(23,i,[5,15],[-10,0],[.125,.25,.5]) for i in range(1,30)]
    assert {x["burst_fraction"] for x in a}=={.125,.25,.5}
    assert len({x["seed"] for x in a})==len(a)
