#!/usr/bin/env bash
# Re-evaluate the three frozen x_best placement modes with raw-waveform
# ESTOI, frozen-Whisper WER, ViSQOL MOS-LQO, and the SI-SDR<-10 dB fraction.
# This script does not train, tune, or modify the frozen x_best UEP profile.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CFG="configs/eval_r4_jammer_refiner_fixed.yaml"
CKPT_DIR="runs/waveform_aware_wireless/r4_jammer_refiner/full_medium"
CKPT_NAME="last.pt"
ROOT_OUT="runs/waveform_aware_wireless"
COMMON=(
  --config "$CFG"
  --checkpoint-dir "$CKPT_DIR"
  --checkpoint-names "$CKPT_NAME"
  --snr-db 5 10 15
  --jsr-db 5 10
  --device cuda
  --allocation-sequence-mode same_condition_warmup_tti_pair
  --enable-estoi
  --enable-wer
  --enable-visqol
  --overwrite
)

EXISTING_OUT="$ROOT_OUT/r4_xbest_existing_speech_metrics"
CSI_OUT="$ROOT_OUT/r4_xbest_csi_only_speech_metrics"
INTERFERENCE_OUT="$ROOT_OUT/r4_xbest_interference_speech_metrics"

# 864 paired conditions each: 16 utterances x 2 realizations x 3 SNR x
# (no_jammer + 4 jammer types x 2 JSR).  Raw output only is the paper metric.
"$PYTHON_BIN" evaluate_r4_jammer_refiner_checkpoints.py \
  "${COMMON[@]}" \
  --jammer-aware-allocation-mode none \
  --output-dir "$EXISTING_OUT"

"$PYTHON_BIN" evaluate_r4_jammer_refiner_checkpoints.py \
  "${COMMON[@]}" \
  --jammer-aware-allocation-mode csi_only \
  --output-dir "$CSI_OUT"

"$PYTHON_BIN" evaluate_r4_jammer_refiner_checkpoints.py \
  "${COMMON[@]}" \
  --jammer-aware-allocation-mode delayed_rx_interference \
  --output-dir "$INTERFERENCE_OUT" \
  --compare-existing-dir "$EXISTING_OUT" \
  --compare-interference-dir "$INTERFERENCE_OUT"

echo "Per-mode quality summaries:"
printf '  %s\n' "$EXISTING_OUT/speech_quality_summary.csv" "$CSI_OUT/speech_quality_summary.csv" "$INTERFERENCE_OUT/speech_quality_summary.csv"
echo "Strict three-way SI-SDR + ESTOI/WER/ViSQOL comparison:"
printf '  %s\n' "$INTERFERENCE_OUT/summary_by_condition.csv"
