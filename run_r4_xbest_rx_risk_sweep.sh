#!/usr/bin/env bash
# Fixed x_best UEP profile; only RX-estimated RE risk alpha is swept.
set -euo pipefail

python evaluate_r4_jammer_refiner_checkpoints.py \
  --config configs/eval_r4_jammer_refiner_fixed.yaml \
  --checkpoint-dir runs/waveform_aware_wireless/r4_jammer_refiner/full_medium \
  --checkpoint-names last.pt \
  --allocation-risk-mode delayed_rx_residual \
  --risk-alpha-sweep 0 0.25 0.5 1.0 2.0 \
  --device cuda \
  --output-dir runs/waveform_aware_wireless/r4_jammer_refiner/xbest_rx_risk_sweep_cuda \
  --overwrite
