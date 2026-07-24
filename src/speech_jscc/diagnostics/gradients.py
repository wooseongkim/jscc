from __future__ import annotations
import torch
from evaluation.paired import run_mode_on_paired_batch
from train_latent_jscc import layer_weighted_latent_mse

def gradient_update_audit(codec, model, batch, config):
    state=torch.zeros(batch.representation.shape[0],model.encoder.channel_state_dim,device=batch.representation.device)
    gates=torch.ones(batch.representation.shape[0],model.encoder.num_layers,device=batch.representation.device)
    opt=torch.optim.Adam(list(model.encoder.parameters())+list(model.decoder.parameters()),lr=float(config["train"]["learning_rate"]))
    before={n:p.detach().clone() for n,p in model.named_parameters()}; opt.zero_grad(set_to_none=True)
    result=run_mode_on_paired_batch(codec,model,batch,state,gates,equalizer="estimated",fading="multipath_block",channel_estimator="dft_tap_ls",estimator_num_taps=config["channel"]["estimator_num_taps"],allocation_mode="uniform",resource_reliability=torch.ones_like(batch.noise.real),receiver_state_mode="observable_v1",decode_waveform=False)
    intermediates={k:result[k] for k in ("data_symbols","transmitted","received","equalized_estimated","decoder_input","reconstruction")}
    for value in intermediates.values(): value.retain_grad()
    loss,_=layer_weighted_latent_mse(result["reconstruction"],batch.representation,torch.ones(model.encoder.num_layers,device=state.device),config["train"]["latent_normalization"]); loss.backward()
    def stats(params):
        ps=list(params); grads=[p.grad for p in ps if p.grad is not None]; total=torch.sqrt(sum((g.square().sum() for g in grads),torch.tensor(0.,device=state.device)))
        return {"gradient_norm":float(total),"maximum_absolute_gradient":max(float(g.abs().max()) for g in grads),"finite":all(bool(torch.isfinite(g).all()) for g in grads)}
    groups={f"encoder_branch_{i}":stats(branch.parameters()) for i,branch in enumerate(model.encoder.layer_encoders)}; groups["decoder"]=stats(model.decoder.parameters())
    opt.step()
    for name,values in groups.items():
        prefix="encoder.layer_encoders."+name.rsplit("_",1)[-1] if name.startswith("encoder") else "decoder."
        updates=[(p.detach()-before[n]).square().sum() for n,p in model.named_parameters() if n.startswith(prefix)]
        values["update_norm"]=float(torch.sqrt(sum(updates,torch.tensor(0.,device=state.device))))
    return {"loss":float(loss.detach()),"groups":groups,"intermediates":{k:{"has_gradient":v.grad is not None,"gradient_norm":float(v.grad.norm()) if v.grad is not None else 0.0} for k,v in intermediates.items()},"codec_has_gradient":any(p.grad is not None for p in codec.parameters())}
