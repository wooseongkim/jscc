#!/usr/bin/env bash
set -euo pipefail
DRY=0; EXECUTE=0; DEVICE=auto; OVERWRITE=0; RESUME=""
while (($#)); do case "$1" in --dry-run) DRY=1; shift;; --execute-ready-training) EXECUTE=1; shift;; --device) DEVICE=$2; shift 2;; --overwrite) OVERWRITE=1; shift;; --resume) RESUME=$2; shift 2;; *) echo "unknown option: $1" >&2; exit 2;; esac; done
ROOT=runs/stage1_random_distribution
python evaluate_stage1_readiness.py --root "$ROOT" --fixed-path-passed --tests-passed >/dev/null
READY=$(python -c 'import json; print(str(json.load(open("runs/stage1_random_distribution/uniform_stage1_readiness.json"))["ready"]).lower())')
[[ $READY == true ]] || { echo "Uniform Stage-1 readiness is false; training command is hidden"; exit 1; }
CMD=(python train_stage1_fixed_tx.py --config configs/train_stage1_fixed_tx_uniform.yaml --steps 20000 --output_dir runs/stage1_uniform_pilot_reserved_v1_scientific)
if [[ $DRY -eq 1 || $EXECUTE -eq 0 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
[[ $OVERWRITE -eq 1 ]] || [[ ! -e runs/stage1_uniform_pilot_reserved_v1_scientific ]] || { echo "refusing existing scientific output" >&2; exit 1; }
"${CMD[@]}" 2>&1 | tee runs/stage1_uniform_pilot_reserved_v1_scientific.log
