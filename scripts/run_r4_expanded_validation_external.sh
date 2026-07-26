#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
if [[ "${1:-}" == "--device" ]]; then
  DEVICE="$2"
  shift 2
fi

python evaluate_r4_expanded_validation.py \
  --config configs/eval_r4_expanded_validation.yaml \
  --device "$DEVICE" \
  --allow-long-run \
  "$@"

