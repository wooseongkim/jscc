#!/usr/bin/env bash
set -euo pipefail
DEVICE=auto; DRY=0; OVERWRITE=0; RESUME="runs/stage1_random_distribution/o6_random_clean/diagnostic_last.pt"; ROOT="runs/stage1_random_distribution/o6_random_clean"
while (($#)); do case "$1" in --device) DEVICE=$2; shift 2;; --dry-run) DRY=1; shift;; --overwrite) OVERWRITE=1; shift;; --resume) RESUME=$2; shift 2;; *) echo "unknown option: $1" >&2; exit 2;; esac; done
CMD=(python diagnose_stage1_random_distribution.py --config configs/train_stage1_fixed_tx_uniform.yaml --stage o6_random_clean --steps 3000 --seed 23 --output-dir "$ROOT" --device "$DEVICE" --checkpoint-every 250 --validation-every 100 --allow-long-run --initialization-mode curriculum_resume --resume "$RESUME")
[[ $OVERWRITE -eq 1 ]] && CMD+=(--overwrite)
if [[ $DRY -eq 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
[[ -f "$RESUME" ]] || { echo "missing resume checkpoint: $RESUME" >&2; exit 1; }
"${CMD[@]}" 2>&1 | tee -a "$ROOT/extension.log"
