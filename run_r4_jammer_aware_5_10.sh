#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
case "$mode" in
  baseline)
    allocation_args=()
    output_dir="runs/waveform_aware_wireless/r4_jammer_aware_baseline_5_10"
    ;;
  sinr)
    allocation_args=(--jammer-aware-allocation-mode delayed_rx_interference)
    output_dir="runs/waveform_aware_wireless/r4_jammer_aware_sinr_5_10"
    ;;
  *)
    echo "usage: bash run_r4_jammer_aware_5_10.sh {baseline|sinr}" >&2
    exit 2
    ;;
esac

python evaluate_r4_jammer_refiner_checkpoints.py \
  --config configs/eval_r4_jammer_refiner_fixed.yaml \
  --checkpoint-dir runs/waveform_aware_wireless/r4_jammer_refiner/full_medium \
  --checkpoint-names last.pt \
  --snr-db 5 10 15 \
  --jsr-db 5 10 \
  --device cuda \
  --allocation-sequence-mode same_condition_warmup_tti_pair \
  "${allocation_args[@]}" \
  --output-dir "$output_dir" \
  --overwrite
