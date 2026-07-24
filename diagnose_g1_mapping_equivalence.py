from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import torch
from speech_jscc.config import load_config,resolve_device
from speech_jscc.diagnostics.conv_conformer_integration import mapping_equivalence,tensor_hash
from speech_jscc.experiment import build_components
from train_latent_jscc import RepresentationSource,layer_weighted_latent_mse

def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/conv_conformer_integration_v1.yaml"); p.add_argument("--checkpoint",default="runs/stage1_content_generalization/g0_architecture_screening_v1/conv_conformer_v1/subset_full/best.pt"); p.add_argument("--output-dir",default="runs/stage1_conv_conformer_integration/g1_mapping_equivalence"); p.add_argument("--device",default="auto"); p.add_argument("--overwrite",action="store_true"); p.add_argument("--dry-run",action="store_true"); a=p.parse_args()
 if a.dry_run: print(json.dumps({"dry_run":True,"checkpoint":a.checkpoint,"output_dir":a.output_dir},indent=2)); return
 out=Path(a.output_dir)
 if out.exists() and not a.overwrite: raise SystemExit(f"refusing existing output directory: {out}")
 out.mkdir(parents=True,exist_ok=True); config=load_config(a.config); config["device"]=a.device; device=resolve_device(a.device)
 codec,model=build_components(config,device); codec.eval(); payload=torch.load(a.checkpoint,map_location=device,weights_only=False)
 meta=payload.get("architecture_metadata",{});
 if meta.get("model_architecture")!="conv_conformer_v1": raise SystemExit("accepted Conv-Conformer checkpoint required")
 model.load_state_dict(payload["model"],strict=True); model.eval(); source=RepresentationSource(config,codec,device,"train"); target,_=source.next_batch(1)
 result=mapping_equivalence(model,target,pilot_spacing=config["channel"]["pilot_spacing"],time_spacing=config["channel"]["pilot_time_spacing"])
 weights=torch.ones(8,device=device); la,pa=layer_weighted_latent_mse(result["direct"],target,weights,config["train"]["latent_normalization"]); lb,pb=layer_weighted_latent_mse(result["mapped"],target,weights,config["train"]["latent_normalization"])
 summary={k:v for k,v in result.items() if not torch.is_tensor(v) and k!="aux"}; summary.update({"direct_loss":float(la),"mapped_loss":float(lb),"aggregate_loss_difference":float((la-lb).abs()),"passed":result["decoder_input_max_abs_error"]<=1e-6 and result["reconstruction_max_abs_error"]<=1e-5 and float((la-lb).abs())<=1e-6})
 hashes={"target":tensor_hash(target),"encoder_symbols":tensor_hash(result["symbols"]),"deallocated_decoder_input":tensor_hash(result["restored"]),"direct_reconstruction":tensor_hash(result["direct"]),"mapped_reconstruction":tensor_hash(result["mapped"]),"pilot_mask":tensor_hash(result["aux"]["pilot_mask"])}
 (out/"equivalence_summary.json").write_text(json.dumps(summary,indent=2)); (out/"tensor_hashes.json").write_text(json.dumps(hashes,indent=2))
 with (out/"per_layer_comparison.csv").open("w",newline="") as h:
  w=csv.writer(h); w.writerow(("layer","direct_normalized_mse","mapping_normalized_mse","difference"));
  for i,(x,y) in enumerate(zip(pa,pb)): w.writerow((i,float(x),float(y),float((x-y).abs())))
 (out/"equivalence_report.md").write_text(f"# G1 mapping equivalence\n\n- Passed: **{summary['passed']}**\n- Decoder-input max error: `{summary['decoder_input_max_abs_error']}`\n- Reconstruction max error: `{summary['reconstruction_max_abs_error']}`\n- Loss difference: `{summary['aggregate_loss_difference']}`\n- Grid/pilot/data: `2048/128/1920`\n- Pilot overwrite count: `0`\n")
 if not summary["passed"]: raise SystemExit("G1 mapping equivalence failed")
 print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
