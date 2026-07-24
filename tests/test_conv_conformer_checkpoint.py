import pytest, torch
from speech_jscc.models.conv_conformer import ConvConformerJSCC
from speech_jscc.models.architecture_checkpoint import architecture_metadata, load_architecture_checkpoint

def test_checkpoint_rejects_cross_architecture_and_missing_metadata(tmp_path):
    model=ConvConformerJSCC((8,50,1024),1920,8,d_model=16,encoder_conformer_blocks=0,decoder_conformer_blocks=0,num_attention_heads=4,ffn_expansion=2,convolution_kernel_size=7)
    for payload in ({"model":model.state_dict()}, {"model":model.state_dict(),"architecture_metadata":{"model_architecture":"flat_mlp"}}):
        path=tmp_path/str(len(list(tmp_path.iterdir())))
        torch.save(payload,path)
        with pytest.raises(ValueError): load_architecture_checkpoint(path,model,expected_metadata=architecture_metadata(model,{},"m","c"))

def test_checkpoint_strict_round_trip_and_normalization_mismatch(tmp_path):
    model=ConvConformerJSCC((8,50,1024),1920,8,d_model=16,encoder_conformer_blocks=0,decoder_conformer_blocks=0,num_attention_heads=4,ffn_expansion=2,convolution_kernel_size=7)
    meta=architecture_metadata(model,{"normalization_mode":"none","normalization_stats_hash":"none"},"m","c")
    path=tmp_path/"ok.pt"; torch.save({"model":model.state_dict(),"architecture_metadata":meta},path)
    load_architecture_checkpoint(path,model,expected_metadata=meta)
    bad=dict(meta,normalization_stats_hash="different")
    with pytest.raises(ValueError,match="normalization"):
        load_architecture_checkpoint(path,model,expected_metadata=bad)
