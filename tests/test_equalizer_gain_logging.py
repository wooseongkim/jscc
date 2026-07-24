import torch
from evaluation.paired import equalizer_gain_statistics

def test_raw_and_applied_equalizer_gain_are_reported():
    h=torch.tensor([[.05+0j,.2+0j,1+0j]],dtype=torch.complex64)
    raw=equalizer_gain_statistics(h,None);clipped=equalizer_gain_statistics(h,10.)
    assert raw['applied_equalizer_gain_max']==raw['raw_equalizer_gain_max']
    assert clipped['raw_equalizer_gain_max']>10
    assert clipped['applied_equalizer_gain_max']<=10
    assert clipped['equalizer_gain_clip_threshold']==10
    assert clipped['clipped_resource_fraction']>0
