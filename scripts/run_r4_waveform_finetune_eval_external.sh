#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

device=cuda
checkpoint=runs/waveform_aware_wireless/r4_repetition3_mrc_finetune/best_clean_gate.pt
output_dir=runs/waveform_aware_wireless/r4_repetition3_mrc_finetune/final_evaluation
overwrite=0
dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) device="$2"; shift 2 ;;
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --overwrite) overwrite=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

args=(--config configs/train_r4_waveform_finetune.yaml --physical-config configs/ofdm_nr_like_r4_repetition3_mrc.yaml --checkpoint "$checkpoint" --output-dir "$output_dir" --device "$device" --utterances 64 --realizations 2)
(( overwrite )) && args+=(--overwrite)
if (( dry_run )); then args+=(--dry-run); else args+=(--allow-long-run); fi
mkdir -p "$(dirname "$output_dir")"
python evaluate_r4_waveform_finetune.py "${args[@]}" 2>&1 | tee "${output_dir%/}_external.log"
