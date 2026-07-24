import torch
from speech_jscc.models.conv_conformer import ConvConformerJSCC
from speech_jscc.diagnostics.conv_conformer_integration import mapping_equivalence

def model(): return ConvConformerJSCC((8,50,1024),1920,8,d_model=16,encoder_conformer_blocks=0,decoder_conformer_blocks=0,num_attention_heads=4,ffn_expansion=2,convolution_kernel_size=7,dropout=0)

def test_g1_mapping_is_numerically_identical():
    result=mapping_equivalence(model().eval(),torch.randn(1,8,50,1024),pilot_spacing=4,time_spacing=4)
    assert result["grid_total_resources"]==2048 and result["pilot_resources"]==128 and result["data_resources"]==1920
    assert result["overwrite_count"]==0 and result["decoder_input_max_abs_error"]<=1e-6
    assert result["reconstruction_max_abs_error"]<=1e-5 and result["masks_disjoint"] and result["masks_exhaustive"]
