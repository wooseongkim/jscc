#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/codec_only_baseline.yaml}"
MANIFEST="${MANIFEST:-manifests/mini_librispeech/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/codec_only/test_100_final_protocol_comparison}"
PYTHON="${PYTHON:-python}"

"${PYTHON}" eval_codec_only.py \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --split test \
  --max_items 100 \
  --batch_size 4 \
  --decode_mode both \
  --compare_official true \
  --metric_align peak_xcorr \
  --max_lag_samples 1000 \
  --snr_scale_match true \
  --metric_zero_mean true \
  --protocol_comparison true \
  --worst_k 10 \
  --waveform_samples 16000 32000 48000 \
  --output_dir "${OUTPUT_DIR}"
