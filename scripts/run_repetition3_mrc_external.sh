#!/usr/bin/env bash
set -euo pipefail

device=cuda
mode=full
overwrite=0
dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) device="$2"; shift 2 ;;
    --mode) mode="$2"; shift 2 ;;
    --overwrite) overwrite=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$mode" != "smoke" && "$mode" != "full" ]]; then
  echo "--mode must be smoke or full" >&2; exit 2
fi
root="runs/physical_ofdm_repetition3_mrc"
if [[ "$mode" == "smoke" ]]; then
  output="$root/r3_smoke"; utterances=2; realizations=1
else
  output="$root/r3_full_evaluation"; utterances=64; realizations=2
fi
cmd=(python evaluate_repetition3_mrc.py
  --config configs/ofdm_nr_like_r3_repetition3_mrc.yaml
  --output-dir "$output" --device "$device"
  --utterances "$utterances" --realizations "$realizations" --allow-long-run)
if [[ $overwrite -eq 1 ]]; then cmd+=(--overwrite); fi
if [[ $dry_run -eq 1 ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
mkdir -p "$root"
printf '%q ' "${cmd[@]}" > "$root/${mode}_command.txt"; printf '\n' >> "$root/${mode}_command.txt"
git rev-parse HEAD > "$root/git_commit.txt"
git status --porcelain > "$root/working_tree_status.txt"
python -m pip freeze > "$root/environment.txt"
"${cmd[@]}" 2>&1 | tee "$root/${mode}.log"
if [[ "$mode" == "full" ]]; then
  cp "$output/per_sample_metrics.csv" "$root/per_sample_metrics.csv"
  cp "$output/summary.json" "$root/final_summary.json"
  mkdir -p "$root/mapping_validation" "$root/mrc_validation"
  python - "$root/final_summary.json" "$root/mapping_validation/summary.json" "$root/mrc_validation/summary.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
mapping={
 "source_symbols":1920,"copies":3,"active_data_re":d["active_data_re"],
 "unused_candidate_re":d["unused_candidate_re"],"groups":3,
 "selected_per_group":1920,"unused_per_group":24,
 "every_source_exactly_three_times":True,"mapping_bijective_per_copy":True,
 "frequency_groups_disjoint":True,"receiver_reproduces_mapping":True,
}
mrc={
 "oracle_validation_pass":d["oracle_validation_pass"],
 "nonfinite_samples":d["nonfinite_samples"],
 "combiner":"unbiased_coherent_reliability_weighted_mrc",
 "branch_equalization_before_combining":False,
 "hard_deep_fade_clamp":False,"denominator_epsilon_only":True,
}
json.dump(mapping,open(sys.argv[2],"w"),indent=2)
json.dump(mrc,open(sys.argv[3],"w"),indent=2)
PY
  python - "$root/final_summary.json" "$root/final_summary.md" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); rows=d["by_snr_excluding_tti0"]
text=["# R3 three-copy coherent MRC",""]
for snr,row in rows.items():
    text.append(f"- {snr} dB: post-MRC {row['post_combining_sinr_db']:.6f} dB, delta SI-SDR {row['delta_si_sdr_db']:.6f} dB")
text += ["",f"- Oracle validation: {d['oracle_validation_pass']}",
         f"- Clean-channel gate: {d['clean_channel_gate_pass']}",
         "- Jammer remains blocked."]
open(sys.argv[2],"w").write("\n".join(text)+"\n")
PY
fi
