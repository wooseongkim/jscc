#!/usr/bin/env bash
set -euo pipefail

# x_best repetition/power is loaded by the fixed-evaluation config.  This
# script changes only RE placement: CSI reliability only, no residual-risk or
# interference input.  The two comparison runs must already exist.
python evaluate_r4_jammer_refiner_checkpoints.py \
  --config configs/eval_r4_jammer_refiner_fixed.yaml \
  --checkpoint-dir runs/waveform_aware_wireless/r4_jammer_refiner/full_medium \
  --checkpoint-names last.pt \
  --snr-db 5 10 15 \
  --jsr-db 5 10 \
  --device cuda \
  --allocation-sequence-mode same_condition_warmup_tti_pair \
  --jammer-aware-allocation-mode csi_only \
  --compare-existing-dir runs/waveform_aware_wireless/r4_jammer_aware_baseline_5_10 \
  --compare-interference-dir runs/waveform_aware_wireless/r4_jammer_aware_sinr_5_10 \
  --output-dir runs/waveform_aware_wireless/r4_csi_only_5_10 \
  --overwrite
