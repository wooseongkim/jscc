from __future__ import annotations
import torch
from evaluation.paired import generate_paired_evaluation_batch
from speech_jscc.codecs import MockContinuousCodec
from speech_jscc.models import SpeechJSCC
from speech_jscc.diagnostics.gradients import gradient_update_audit

def test_every_encoder_branch_and_decoder_receive_gradients_and_update():
    codec=MockContinuousCodec(2,3,4,96,seed=1); model=SpeechJSCC((2,3,4),24,8,16)
    wave=torch.randn(1,96); target=codec.encode_waveform(wave)
    batch=generate_paired_evaluation_batch(codec,batch_size=1,waveform_samples=96,channel_shape=(8,4),snr_db=15,jsr_db=0,jammer_type="none",jammed_fraction=.25,pilot_spacing=2,pilot_time_spacing=2,target_power=1,seed=4,device=torch.device("cpu"),fading="multipath_block",num_taps=2,channel_estimator="dft_tap_ls",estimator_num_taps=2,waveform=wave,representation=target)
    config={"train":{"learning_rate":1e-3,"latent_normalization":{"mode":"per_layer_power","epsilon":1e-6}},"channel":{"estimator_num_taps":2}}
    audit=gradient_update_audit(codec,model,batch,config)
    assert not audit["codec_has_gradient"]
    assert all(g["finite"] and g["gradient_norm"]>0 and g["update_norm"]>0 for g in audit["groups"].values())
    assert all(v["has_gradient"] and v["gradient_norm"]>0 for v in audit["intermediates"].values())
