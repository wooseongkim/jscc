import math
import torch

from channels.jammer import make_jammer
from channels.pilot import make_pilot_mask
from speech_jscc.diagnostics.j3_narrowband import narrowband_diagnostics, local_inband_jsr_db, j3_policy


def test_narrowband_is_contiguous_frequency_band_across_all_time():
    reference=torch.ones(2,64,32,dtype=torch.complex64)
    jammer,mask=make_jammer(reference,0.,"narrowband",.25,generator=torch.Generator().manual_seed(3))
    for row in mask:
        active=torch.where(row.any(dim=1))[0]
        assert active.numel()==16
        assert torch.equal(active,torch.arange(active[0],active[0]+16))
        assert row[active].all() and not row[~row.any(dim=1)].any()
    assert jammer[~mask].abs().max()==0


def test_global_and_local_jsr_and_overlap_are_explicit():
    reference=torch.ones(1,64,32,dtype=torch.complex64)
    jammer,mask=make_jammer(reference,-3.,"narrowband",.25,generator=torch.Generator().manual_seed(4))
    pilots=make_pilot_mask(reference.shape,4,time_spacing=4)
    value=narrowband_diagnostics(reference,jammer,mask,pilots,requested_fraction=.25,requested_global_jsr_db=-3.)
    assert value["requested_global_jsr_db"]== -3.
    assert value["actual_jammed_subcarrier_count"]==16
    assert value["jammed_subcarrier_fraction"]==.25
    assert value["local_inband_jsr_db"]==pytest.approx(-3-10*math.log10(.25),abs=1e-5)
    assert value["leakage_power_outside_band"]==0
    assert value["contiguous_band_verified"]
    assert 0<=value["pilot_resource_overlap_ratio"]<=1
    assert 0<=value["data_resource_overlap_ratio"]<=1
    assert len(value["per_layer_jammed_data_fraction"]) == 8
    assert sum(value["per_layer_jammed_data_count"]) == int((mask & ~pilots).sum())


def test_j3_policy_varies_location_inputs_and_is_reproducible():
    a=[j3_policy(23,i,[5,15],[-10,0],[.125,.25,.5]) for i in range(1,20)]
    b=[j3_policy(23,i,[5,15],[-10,0],[.125,.25,.5]) for i in range(1,20)]
    assert a==b
    assert {row["jammed_fraction"] for row in a}=={.125,.25,.5}
    assert len({row["seed"] for row in a})==len(a)


import pytest
