#!/usr/bin/env bash
set -euo pipefail

device=cuda
profile=r3
mode=full
overwrite=0
dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) device="$2"; shift 2 ;;
    --profile) profile="$2"; shift 2 ;;
    --mode) mode="$2"; shift 2 ;;
    --overwrite) overwrite=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$profile" != "r2" && "$profile" != "r3" ]]; then
  echo "--profile must be r2 or r3" >&2; exit 2
fi
if [[ "$mode" != "smoke" && "$mode" != "full" ]]; then
  echo "--mode must be smoke or full" >&2; exit 2
fi
if [[ "$mode" == "full" && "$profile" != "r3" ]]; then
  echo "full evaluation is restricted to primary R3 profile" >&2; exit 2
fi

config="configs/ofdm_nr_like_${profile}.yaml"
if [[ "$mode" == "smoke" ]]; then
  output="runs/physical_ofdm_profiles/${profile}_smoke"
  utterances=2
  realizations=1
else
  output="runs/physical_ofdm_profiles/r3_full_evaluation"
  utterances=64
  realizations=2
fi
cmd=(python evaluate_physical_fdd.py --config "$config"
  --checkpoint runs/waveform_aware_wireless/clean_channel_training/best_waveform_si_sdr.pt
  --output-dir "$output" --device "$device" --utterances "$utterances"
  --realizations "$realizations" --allow-long-run)
if [[ $overwrite -eq 1 ]]; then cmd+=(--overwrite); fi
if [[ $dry_run -eq 1 ]]; then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi

root="runs/physical_ofdm_profiles"
mkdir -p "$root"
printf '%q ' "${cmd[@]}" > "$root/${profile}_${mode}_command.txt"
printf '\n' >> "$root/${profile}_${mode}_command.txt"
git rev-parse HEAD > "$root/git_commit.txt"
git status --porcelain > "$root/working_tree_status.txt"
python -m pip freeze > "$root/environment.txt"
"${cmd[@]}" 2>&1 | tee "$root/${profile}_${mode}.log"
if [[ "$mode" == "full" ]]; then
  cp "$output/summary.json" "$root/final_summary.json"
  python - "$root/final_summary.json" "$root/final_summary.md" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
p=d["profile"]; rows=d["by_snr_excluding_tti0"]; gate=d["five_db_target"]
text=[
"# Physical R3 OFDM evaluation","",
f"- Profile: {p['profile_name']}",
f"- Physical TTI: {p['tti_duration_s']*1e3:.6f} ms",
f"- Active/candidate/selected RE: {p['active_source_re']}/{p['candidate_data_re']}/1920",
f"- Unused candidate RE: {p['unused_candidate_re']}","",
"## Post-MMSE source-symbol SINR","",
]
for snr,row in rows.items(): text.append(f"- Nominal {snr} dB: {row['post_mmse_sinr_db']:.6f} dB")
text += ["",f"- 5 dB target passed: {gate['passed']}",
         f"- Remaining gap: {gate['remaining_gap_db']:.6f} dB",
         "- Jammer remains blocked."]
open(sys.argv[2],"w").write("\\n".join(text)+"\\n")
PY
fi
