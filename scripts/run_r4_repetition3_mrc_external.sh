#!/usr/bin/env bash
set -euo pipefail

device=cuda
utterances=64
realizations=2
overwrite=0
dry_run=0
output_dir=runs/physical_ofdm_r4_repetition3_mrc/r4_full_evaluation
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) device="$2"; shift 2 ;;
    --utterances) utterances="$2"; shift 2 ;;
    --realizations) realizations="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --overwrite) overwrite=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
args=(--config configs/ofdm_nr_like_r4_repetition3_mrc.yaml --output-dir "$output_dir" --device "$device" --utterances "$utterances" --realizations "$realizations")
(( overwrite )) && args+=(--overwrite)
if (( dry_run )); then
  args+=(--dry-run)
else
  args+=(--allow-long-run)
fi
mkdir -p runs/physical_ofdm_r4_repetition3_mrc
python evaluate_r4_repetition3_mrc.py "${args[@]}" 2>&1 | tee runs/physical_ofdm_r4_repetition3_mrc/external.log
