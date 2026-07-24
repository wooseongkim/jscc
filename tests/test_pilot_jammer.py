import math
import torch
import pytest
from channels.pilot import make_pilot_mask
from speech_jscc.diagnostics.j5_pilot import make_pilot_local_jammer,pilot_jammer_diagnostics

def test_partial_pilot_attack_is_exact_subset_and_has_no_data_leakage():
    ref=torch.ones(2,64,32,dtype=torch.complex64);pilots=make_pilot_mask(ref.shape,4,time_spacing=4)
    jammer,mask=make_pilot_local_jammer(ref,pilots,0.,.25,generator=torch.Generator().manual_seed(3))
    assert (mask & ~pilots).sum()==0
    assert torch.all(mask.sum((1,2))==32)
    assert jammer[~pilots].abs().max()==0
    repeated=make_pilot_local_jammer(ref,pilots,0.,.25,generator=torch.Generator().manual_seed(3))[1]
    assert torch.equal(mask,repeated)

def test_pilot_local_jsr_and_global_equivalent_are_distinct():
    ref=torch.ones(1,64,32,dtype=torch.complex64);pilots=make_pilot_mask(ref.shape,4,time_spacing=4)
    jammer,mask=make_pilot_local_jammer(ref,pilots,-3.,.5,generator=torch.Generator().manual_seed(2))
    d=pilot_jammer_diagnostics(ref,jammer,mask,pilots,requested_pilot_jsr_db=-3.)
    assert d['requested_pilot_jsr_db']==-3
    assert d['attacked_pilot_count']==64 and d['total_pilot_count']==128
    assert d['requested_global_equivalent_jsr_db']==pytest.approx(-3+10*math.log10(64/2048))
    assert d['jammer_leakage_power_on_data_resources']==0
