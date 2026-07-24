import torch
from speech_jscc.training.channel_free_feasibility import (multi_resolution_stft_loss,negative_si_sdr_loss,
    summed_latent_loss,feasibility_classification,configure_bottleneck,select_unseen_speaker_paths)
from speech_jscc.training.channel_free_feasibility import enable_frozen_rnn_backward,decode_frozen_representation_with_gradient
from eval_channel_free_feasibility import prepare_output_directory


def test_evaluation_overwrite_replaces_existing_output_directory(tmp_path):
    output = tmp_path / "final_comparison"
    output.mkdir()
    (output / "stale.json").write_text("stale")

    audio = prepare_output_directory(output, overwrite=True)

    assert output.is_dir()
    assert audio == output / "waveform_examples"
    assert audio.is_dir()
    assert not (output / "stale.json").exists()

def test_waveform_losses_are_differentiable_and_zero_for_identity():
    target=torch.randn(2,2048);reconstruction=target.clone().requires_grad_(True)
    loss=multi_resolution_stft_loss(reconstruction,target,fft_sizes=(128,256))+negative_si_sdr_loss(reconstruction,target)
    loss.backward();assert reconstruction.grad is not None and torch.isfinite(reconstruction.grad).all()
    assert multi_resolution_stft_loss(target,target,fft_sizes=(128,256))<1e-7

def test_summed_latent_loss_uses_sum_over_codec_layers():
    target=torch.randn(2,8,5,7);reconstruction=target.clone();assert summed_latent_loss(reconstruction,target)==0

def test_waveform_metrics_control_feasibility_classification():
    assert feasibility_classification(delta_si_sdr=-.9,delta_waveform_snr=-.8,stft_ratio=1.1)=='CHANNEL_FREE_FEASIBLE'
    assert feasibility_classification(delta_si_sdr=-2.,delta_waveform_snr=-.8,stft_ratio=1.1)=='MARGINAL'
    assert feasibility_classification(delta_si_sdr=-3.1,delta_waveform_snr=0.,stft_ratio=1.)=='CHANNEL_FREE_INFEASIBLE'

def test_large_bottleneck_changes_only_symbol_budget_fields():
    cfg={'model':{'architecture':'conv_conformer_v1','channel_uses':1920,'symbol_frames':30,'complex_channels_per_symbol_frame':8}}
    large=configure_bottleneck(cfg,7680)
    assert large['model']['channel_uses']==7680 and large['model']['complex_channels_per_symbol_frame']==32
    assert cfg['model']['channel_uses']==1920

def test_unseen_selection_excludes_train_speakers_and_round_robins():
    train=['/x/1/1/1-1-1.flac'];validation=['/x/1/2/1-2-1.flac','/x/2/1/2-1-1.flac','/x/3/1/3-1-1.flac']
    selected=select_unseen_speaker_paths(train,validation,limit=2,seed=3)
    assert {str(x).split('/')[-3] for x in selected}=={'2','3'}

def test_frozen_eval_rnn_is_backward_enabled_without_unfreezing_weights():
    module=torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.LSTM(3,3,batch_first=True));module.eval()
    for parameter in module.parameters():parameter.requires_grad_(False)
    assert enable_frozen_rnn_backward(module)==1
    assert module[0].training is False and module[1].training is True
    assert not any(parameter.requires_grad for parameter in module.parameters())

def test_gradient_decode_bypasses_wrapper_eval_reset():
    class Decoder(torch.nn.Module):
        def __init__(self):super().__init__();self.lstm=torch.nn.LSTM(3,3,batch_first=True)
        def forward(self,x):return self.lstm(x.transpose(1,2))[0].mean(-1).unsqueeze(1)
    class Codec(torch.nn.Module):
        def __init__(self):super().__init__();self.model=torch.nn.Module();self.model.decoder=Decoder();self.waveform_samples=5;self.representation_shape=(2,5,3)
    codec=Codec();codec.eval();[p.requires_grad_(False) for p in codec.parameters()];representation=torch.randn(1,2,5,3,requires_grad=True)
    waveform=decode_frozen_representation_with_gradient(codec,representation);waveform.sum().backward()
    assert representation.grad is not None and codec.model.decoder.lstm.training
