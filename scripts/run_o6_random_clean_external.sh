#!/usr/bin/env bash
set -euo pipefail
DEVICE=auto; DRY=0; OVERWRITE=0; RESUME=""; ROOT="runs/stage1_random_distribution/o6_random_clean"
while (($#)); do case "$1" in
  --device) DEVICE=$2; shift 2;; --dry-run) DRY=1; shift;; --overwrite) OVERWRITE=1; shift;;
  --resume) RESUME=$2; shift 2;; --output-dir) ROOT=$2; shift 2;; *) echo "unknown option: $1" >&2; exit 2;; esac; done
if [[ -e "$ROOT" && $OVERWRITE -eq 0 && -z "$RESUME" ]]; then echo "refusing existing result directory: $ROOT" >&2; exit 1; fi
CMD=(python diagnose_stage1_random_distribution.py --config configs/train_stage1_fixed_tx_uniform.yaml --stage o6_random_clean --steps 1000 --seed 23 --output-dir "$ROOT" --device "$DEVICE" --checkpoint-every 250 --validation-every 100 --allow-long-run --initialization-mode curriculum_resume)
[[ -n "$RESUME" ]] && CMD+=(--resume "$RESUME")
[[ $OVERWRITE -eq 1 ]] && CMD+=(--overwrite)
if [[ $DRY -eq 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
PENDING_LOG="${ROOT}.run.log"
"${CMD[@]}" 2>&1 | tee "$PENDING_LOG"
mv "$PENDING_LOG" "$ROOT/run.log"
