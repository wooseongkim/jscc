#!/usr/bin/env bash
set -euo pipefail
BATCH_SIZE=4; MAX_EPOCHS=64; DEVICE=auto; RESUME=""; OVERWRITE=0; DRY=0; START_SUBSET=16
ROOT="runs/stage1_content_generalization/g0_exposure_normalized_v1"
while (($#)); do case "$1" in
 --batch-size) BATCH_SIZE=$2; shift 2;; --max-epochs) MAX_EPOCHS=$2; shift 2;; --device) DEVICE=$2; shift 2;;
 --resume) RESUME=$2; shift 2;; --overwrite) OVERWRITE=1; shift;; --dry-run) DRY=1; shift;; --start-subset) START_SUBSET=$2; shift 2;;
 --root) ROOT=$2; shift 2;; *) echo "unknown option: $1" >&2; exit 2;; esac; done
SUBSETS=(16 64 256 full); start=-1
for i in "${!SUBSETS[@]}"; do [[ ${SUBSETS[$i]} == "$START_SUBSET" ]] && start=$i; done
[[ $start -ge 0 ]] || { echo "invalid --start-subset" >&2; exit 2; }
for ((i=start;i<${#SUBSETS[@]};i++)); do
 subset=${SUBSETS[$i]}; out="$ROOT/subset_$subset"
 if [[ -e "$out" && $OVERWRITE -eq 0 && -z "$RESUME" ]]; then echo "refusing existing result directory: $out" >&2; exit 1; fi
 CMD=(python diagnose_g0_exposure_normalized.py --config configs/train_stage1_fixed_tx_uniform.yaml --subset-size "$subset" --batch-size "$BATCH_SIZE" --max-epochs "$MAX_EPOCHS" --seed 23 --output-dir "$out" --device "$DEVICE" --allow-long-run)
 [[ $OVERWRITE -eq 1 ]] && CMD+=(--overwrite)
 [[ -n "$RESUME" && $i -eq $start ]] && CMD+=(--resume "$RESUME")
 if [[ $DRY -eq 1 ]]; then printf '%q ' "${CMD[@]}"; echo; continue; fi
 mkdir -p "$(dirname "$out")"; PENDING_LOG="${out}.run.log"; "${CMD[@]}" 2>&1 | tee "$PENDING_LOG"; mv "$PENDING_LOG" "$out/run.log"
 python scripts/summarize_g0_exposure.py --root "$ROOT"; RESUME=""
done
