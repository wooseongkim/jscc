import pytest, torch
from pathlib import Path
from diagnose_g0_architecture_screening import _root
from speech_jscc.diagnostics.architecture_screening import parameter_report, direct_bypass
from speech_jscc.models.conv_conformer import ConvConformerJSCC
from speech_jscc.models.system import SpeechJSCC

def test_parameter_report_and_direct_bypass():
    model=ConvConformerJSCC((8,50,1024),1920,8,d_model=16,encoder_conformer_blocks=0,decoder_conformer_blocks=0,num_attention_heads=4,ffn_expansion=2,convolution_kernel_size=7)
    target=torch.randn(1,8,50,1024); state=torch.zeros(1,8)
    reconstruction=direct_bypass(model,target,state)
    assert reconstruction.shape==target.shape
    report=parameter_report(model)
    assert report["total_trainable_parameters"]<30_000_000 and report["speech_tokenizer_trainable_parameters"]==0

def test_direct_bypass_has_no_channel_argument():
    assert "channel" not in direct_bypass.__code__.co_varnames


def test_parameter_report_allows_historical_flat_mlp_output_layer():
    model=SpeechJSCC((8,50,1024),1920,8,32,1.0)
    report=parameter_report(model)
    assert report["total_trainable_parameters"]>0


def test_aggregate_root_is_architecture_screening_directory():
    path=Path("runs/stage1_content_generalization/g0_architecture_screening_v1/normalized_flat_mlp/subset_16")
    assert _root(path)==Path("runs/stage1_content_generalization/g0_architecture_screening_v1")
