from __future__ import annotations
import argparse,csv,json
from pathlib import Path

def write_csv(path,rows):
    fields=sorted({k for r in rows for k in r}) if rows else []
    with path.open("w",newline="") as f:
        if fields: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",required=True); a=p.parse_args(); root=Path(a.root); root.mkdir(parents=True,exist_ok=True)
    epochs=[]; layers=[]; subsets=[]
    for label in ("16","64","256","full"):
        path=root/f"subset_{label}"/"summary.json"
        if not path.exists(): continue
        d=json.loads(path.read_text())
        for h in d["history"]:
            for group,m in h["validation"].items():
                row={"subset_size":label,"epoch":h["epoch"],"optimizer_step":h["optimizer_step"],"group":group,
                     "normalized_mse":m["aggregate"]["normalized_mse"],"power_ratio":m["aggregate"]["power_ratio"],"correlation":m["aggregate"]["pearson_correlation"],
                     "improvement_over_zero":m["relative_improvement_over_zero"],"improvement_over_global_mean":m["relative_improvement_over_global_mean"],"improvement_over_layerwise_mean":m["relative_improvement_over_layerwise_mean"]}
                epochs.append(row)
                for x in m["per_layer"]: layers.append({"subset_size":label,"epoch":h["epoch"],"group":group,**x})
        final=d["final"]; unseen=final["validation"]["unseen_speaker_unseen_utterance_unseen_channel"]
        subsets.append({"subset_size":label,"max_epochs":d["provenance"]["max_epochs"],"train_loss":final["train_loss"],
          "same_speaker_loss":final["validation"]["same_speaker_unseen_utterance_unseen_channel"]["aggregate"]["normalized_mse"],
          "unseen_speaker_loss":unseen["aggregate"]["normalized_mse"],"unseen_layer0_correlation":unseen["layer0_summary"]["pearson_correlation"],
          "unseen_layers1_to_7_correlation":unseen["layers1_to_7_summary"]["pearson_correlation"],
          "gate_passed":d["gate"]["passed"],"plateau":d["plateau"],"train_slope":d["slopes"]["train_loss"],"unseen_slope":d["slopes"]["unseen_loss"]})
    write_csv(root/"aggregate_by_epoch.csv",epochs); write_csv(root/"aggregate_by_subset.csv",subsets); write_csv(root/"per_layer_by_epoch.csv",layers)
    classifications=[]
    for s in subsets:
        if s["unseen_slope"] < -1e-4: classifications.append("insufficient optimization budget")
        if s["train_loss"] < .95 and s["unseen_speaker_loss"] >= .95: classifications.append("small-subset memorization")
        if s["same_speaker_loss"] >= .95: classifications.append("utterance generalization failure")
        if s["same_speaker_loss"] < .95 <= s["unseen_speaker_loss"]: classifications.append("speaker generalization failure")
        if s["unseen_layer0_correlation"] > .1 and s["unseen_layers1_to_7_correlation"] < .05: classifications.append("layer-1-to-7 collapse")
        if s["train_loss"] >= .95 and s["plateau"] and s["max_epochs"]>=16: classifications.append("current encoder-decoder optimization/capacity limitation")
    classifications=sorted(set(classifications)); final_class="mixed cause" if len(classifications)>1 else (classifications[0] if classifications else "insufficient evidence")
    manifest={"engine_version":"g0_exposure_normalized_v1","subsets_completed":[x["subset_size"] for x in subsets],"classification":final_class,"evidence":classifications,"g1_g3_o6_j_blocked":True}
    (root/"exposure_manifest.json").write_text(json.dumps(manifest,indent=2))
    lines=["# G0 Exposure-Normalized Report","",f"Classification: **{final_class}**","",f"Evidence: `{classifications}`","","G1–G3, O6, and J stages remain blocked.","","## Subsets",""]
    lines += [f"- {x['subset_size']}: train={x['train_loss']:.6f}, same-speaker={x['same_speaker_loss']:.6f}, unseen-speaker={x['unseen_speaker_loss']:.6f}, gate={'PASS' if x['gate_passed'] else 'FAIL'}" for x in subsets]
    (root/"exposure_normalized_report.md").write_text("\n".join(lines)+"\n"); print(json.dumps(manifest,indent=2))
    if subsets and not any(x["gate_passed"] for x in subsets if x["subset_size"] in {"256","full"}):
        (root/"architecture_followup_proposal.md").write_text(
            "# Post-G0 Architecture Diagnostic Proposal\n\n"
            "If sufficient exposure is confirmed, compare without changing this experiment:\n\n"
            "1. Current flatten MLP under the accepted exposure budget.\n"
            "2. Corpus-level latent normalization with identical loss reporting.\n"
            "3. Temporal structured encoder/decoder preserving the 50-frame axis.\n"
            "4. Linear and PCA reconstruction references at matched real bottleneck dimension.\n\n"
            "Do not classify capacity until train loss and slopes support it.\n"
        )
if __name__=="__main__": main()
