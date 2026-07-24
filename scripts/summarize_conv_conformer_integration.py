from __future__ import annotations
import csv,json
from pathlib import Path

root=Path("runs/stage1_conv_conformer_integration");root.mkdir(parents=True,exist_ok=True)
stages=[("g0_direct",Path("runs/stage1_content_generalization/g0_architecture_screening_v1/conv_conformer_v1/subset_full/summary.json")),("g1_mapping_train",root/"g1_mapping_train/subset_256/summary.json"),("g2_fixed_clean",root/"g2_fixed_clean/subset_256/summary.json"),("g3_random_clean",root/"g3_random_clean/subset_256/summary.json")]
rows=[];first=None
for stage,path in stages:
 if not path.exists(): rows.append({"stage":stage,"status":"not_executed"});continue
 s=json.loads(path.read_text()); validation=s.get("validation",s.get("final",{}).get("validation",{})); unseen=validation.get("unseen_speaker_unseen_utterance_unseen_channel",{}); same=validation.get("same_speaker_unseen_utterance_unseen_channel",{}); passed=s.get("gate",{}).get("stage_pass",s.get("gate",{}).get("passed"))
 row={"stage":stage,"status":"passed" if passed else "failed","same_speaker_loss":same.get("aggregate",{}).get("normalized_mse"),"unseen_speaker_loss":unseen.get("aggregate",{}).get("normalized_mse"),"layer0_loss":unseen.get("per_layer",[{}])[0].get("normalized_mse") if unseen else None,"enhancement_loss":unseen.get("layers1_to_7_summary",{}).get("normalized_mse")};rows.append(row)
 if stage!="g0_direct" and passed is False and first is None:first=stage
(root/"integration_manifest.json").write_text(json.dumps({"version":"conv_conformer_integration_v1","architecture":"conv_conformer_v1","stages":[x[0] for x in stages]},indent=2));(root/"stage_comparison.json").write_text(json.dumps(rows,indent=2));(root/"first_failure.json").write_text(json.dumps({"first_failing_stage":first},indent=2))
with (root/"stage_comparison.csv").open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
if first:
 next_cmd="Inspect the first failed stage before continuing."
elif not (root/"g1_mapping_train/subset_256/summary.json").exists():
 next_cmd="bash scripts/run_g1_conv_conformer_external.sh --device cuda --subset-size 256 --max-epochs 64"
elif not (root/"g2_fixed_clean/subset_256/summary.json").exists():
 next_cmd="bash scripts/run_g2_conv_conformer_external.sh --device cuda --subset-size 256 --max-epochs 64"
elif not (root/"g3_random_clean/subset_256/summary.json").exists():
 next_cmd="bash scripts/run_g3_conv_conformer_external.sh --device cuda --subset-size 256 --max-epochs 64"
else:
 next_cmd="All G1-G3 stages have artifacts; inspect final gates before any later curriculum."
(root/"conv_conformer_integration_report.md").write_text("# Conv-Conformer integration\n\nG0 is accepted. G1-E equivalence is stored separately. Unexecuted stages are not classified.\n\n## Stages\n\n"+"\n".join(f"- {r['stage']}: {r['status']}" for r in rows)+f"\n\n## Next external command\n\n`{next_cmd}`\n")
(root/"external_commands.md").write_text("# External commands\n\n- G1: `bash scripts/run_g1_conv_conformer_external.sh --device cuda --subset-size 256 --max-epochs 64`\n- G2: `bash scripts/run_g2_conv_conformer_external.sh --device cuda --subset-size 256 --max-epochs 64`\n- G3: `bash scripts/run_g3_conv_conformer_external.sh --device cuda --subset-size 256 --max-epochs 64`\n- Sequence: `bash scripts/run_g1_g3_sequence_external.sh --device cuda --subset-size 256 --max-epochs 64 --continue-on-pass`\n")
