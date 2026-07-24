from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch

from channels.pilot import extract_data_resources, insert_data_and_pilots
from models.resource_allocator import allocate_resources, deallocate_resources
from evaluation.paired import run_mode_on_paired_batch
from speech_jscc.config import load_config, resolve_device
from speech_jscc.diagnostics.metrics import latent_metric_rows, zero_predictor_loss
from speech_jscc.diagnostics.overfit import classify_overfit_result
from speech_jscc.experiment import build_components
from speech_jscc.models import SpeechJSCC
from train_latent_jscc import RepresentationSource, layer_weighted_latent_mse
from train_stage1_fixed_tx import _make_batch


def _summary(stage, initial, best, final, best_step, target, reconstruction, zero):
    rows = latent_metric_rows(reconstruction, target, epsilon=1e-6, predictor=stage, scenario=stage)
    def mean(key): return sum(float(r[key]) for r in rows) / len(rows)
    result = {"stage": stage, "initial_loss": initial, "best_loss": best, "final_loss": final,
              "zero_predictor_loss": zero, "relative_improvement_over_zero": (zero-final)/zero,
              "steps_to_best": best_step, "power_ratio": mean("power_ratio"),
              "cosine_similarity": mean("cosine_similarity"),
              "pearson_correlation": mean("pearson_correlation")}
    result["passed"], result["failure_reasons"] = classify_overfit_result(stage, result)
    return result


def run_stage(stage, config, codec, target, device, steps, *, legacy=False):
    torch.manual_seed(int(config["seed"]))
    random.seed(int(config["seed"]))
    uses = [64, 32] if legacy else config["model"]["channel_uses"]
    model = SpeechJSCC(tuple(target.shape[1:]), uses, 8, config["model"]["hidden_dim"], 1.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"]["learning_rate"]))
    weights = torch.ones(target.shape[1], device=device)
    state = torch.zeros(target.shape[0], 8, device=device)
    zero = float(zero_predictor_loss(target, weights, 1e-6)[0])
    batch = None
    if stage in {"O2-P", "O3", "O4", "O5"}:
        snr = 100.0 if stage == "O3" else 10.0
        jammer = "barrage" if stage == "O5" else "none"
        batch = _make_batch(codec, model, config, target=target, waveform=None, snr_db=snr,
                            jsr_db=0.0, jammer_type=jammer, seed=23003, device=device)
    losses=[]; best=float("inf"); best_step=0; final_reconstruction=None
    for step in range(1, steps+1):
        optimizer.zero_grad(set_to_none=True)
        symbols = model.encoder(target, state, layer_gates=torch.ones(target.shape[0],target.shape[1],device=device))
        if stage == "O1": decoder_input=symbols
        elif stage == "O2":
            a=allocate_resources(symbols,torch.ones_like(symbols.real),model.encoder.layer_channel_uses,mode="uniform")
            decoder_input=deallocate_resources(a.symbols,a.resource_to_source)
        elif stage == "O2-P":
            grid,_=insert_data_and_pilots(symbols,batch.pilot_mask)
            decoder_input=extract_data_resources(grid,batch.pilot_mask)
        else:
            result=run_mode_on_paired_batch(codec,model,batch,state,torch.ones(target.shape[0],target.shape[1],device=device),equalizer="estimated",fading="multipath_block",channel_estimator="dft_tap_ls",estimator_num_taps=6,allocation_mode="uniform",resource_reliability=torch.ones_like(batch.noise.real),receiver_state_mode="observable_v1",decode_waveform=False)
            final_reconstruction=result["reconstruction"]
            decoder_input=None
        if decoder_input is not None: final_reconstruction=model.decoder(decoder_input,state)
        loss,_=layer_weighted_latent_mse(final_reconstruction,target,weights,config["train"]["latent_normalization"])
        loss.backward(); optimizer.step(); value=float(loss.detach()); losses.append(value)
        if value<best: best=value; best_step=step
    return _summary(stage,losses[0],best,losses[-1],best_step,target,final_reconstruction.detach(),zero)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--output_dir",required=True)
    p.add_argument("--steps",type=int,default=500); p.add_argument("--continue_after_failure",action="store_true")
    args=p.parse_args(); config=load_config(args.config); device=resolve_device(config.get("device","auto"))
    torch.manual_seed(int(config["seed"])); random.seed(int(config["seed"])); codec,_=build_components(config,device); codec.eval()
    target,_=RepresentationSource(config,codec,device,"train").next_batch(1); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    pre=run_stage("O3",config,codec,target,device,args.steps,legacy=True); (out/"pre_fix_o3.json").write_text(json.dumps(pre,indent=2))
    results=[]
    for stage in ("O1","O2","O2-P","O3","O4","O5"):
        result=run_stage(stage,config,codec,target,device,args.steps); results.append(result)
        with (out/"overfit_results.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=[k for k in result if k!="failure_reasons"]); w.writeheader(); w.writerows([{k:v for k,v in r.items() if k!="failure_reasons"} for r in results])
        (out/"overfit_report.md").write_text("# Stage-1 Overfit Ladder\n\n"+"\n".join(f"- {r['stage']}: {'PASS' if r['passed'] else 'FAIL'} loss={r['final_loss']:.6f}, corr={r['pearson_correlation']:.6f}" for r in results))
        if not result["passed"] and not args.continue_after_failure: break

if __name__ == "__main__": main()
