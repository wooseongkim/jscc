#!/usr/bin/env bash
# Run the fixed held-out CUDA checkpoint comparison without a long shell line.
set -euo pipefail

python evaluate_r4_jammer_refiner_checkpoints.py \
  --config configs/eval_r4_jammer_refiner_fixed.yaml \
  --checkpoint-dir runs/waveform_aware_wireless/r4_jammer_refiner/full_medium \
  --checkpoint-names last.pt best_validation_si_sdr.pt best_validation_latent.pt \
  --device cuda \
  --output-dir runs/waveform_aware_wireless/r4_jammer_refiner/fixed_validation_cuda \
  --overwrite
