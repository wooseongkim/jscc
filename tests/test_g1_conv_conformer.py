import torch
from speech_jscc.models.conv_conformer import ConvConformerJSCC
from speech_jscc.diagnostics.conv_conformer_integration import forward_integration_path

def test_g1_gradients_reach_all_heads():
    model=ConvConformerJSCC((8,50,1024),1920,8,d_model=16,encoder_conformer_blocks=0,decoder_conformer_blocks=0,num_attention_heads=4,ffn_expansion=2,convolution_kernel_size=7)
    out=forward_integration_path("g1_mapping_train",None,model,torch.randn(1,8,50,1024),{"model":{"grid_shape":[64,32]},"channel":{"pilot_spacing":4,"pilot_time_spacing":4}})
    out["reconstruction"].square().mean().backward()
    assert all(any(p.grad is not None and p.grad.abs().sum()>0 for p in h.parameters()) for h in model.encoder.symbol_heads)
    assert all(any(p.grad is not None and p.grad.abs().sum()>0 for p in h.parameters()) for h in model.decoder.reconstruction_heads)
