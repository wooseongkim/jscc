#!/usr/bin/env bash
set -euo pipefail
DEVICE=auto; DRY=0; OVERWRITE=0; RESUME=""; STEPS=1000
ROOT="runs/stage1_content_generalization"; START_STAGE=g0_direct; START_SUBSET=16
while (($#)); do case "$1" in
  --device) DEVICE=$2; shift 2;; --dry-run) DRY=1; shift;; --overwrite) OVERWRITE=1; shift;;
  --resume) RESUME=$2; shift 2;; --steps) STEPS=$2; shift 2;; --root) ROOT=$2; shift 2;;
  --start-stage) START_STAGE=$2; shift 2;; --start-subset) START_SUBSET=$2; shift 2;;
  *) echo "unknown option: $1" >&2; exit 2;; esac; done
STAGES=(g0_direct g1_pilot_reserved_identity g2_fixed_clean g3_random_clean)
SUBSETS=(16 64 256 full)
stage_start=-1; subset_start=-1
for i in "${!STAGES[@]}"; do [[ ${STAGES[$i]} == "$START_STAGE" ]] && stage_start=$i; done
for i in "${!SUBSETS[@]}"; do [[ ${SUBSETS[$i]} == "$START_SUBSET" ]] && subset_start=$i; done
[[ $stage_start -ge 0 && $subset_start -ge 0 ]] || { echo "invalid start stage/subset" >&2; exit 2; }
for ((si=stage_start; si<${#STAGES[@]}; si++)); do
  stage=${STAGES[$si]}; local_start=0; [[ $si -eq $stage_start ]] && local_start=$subset_start
  stage_passed=0
  for ((ui=local_start; ui<${#SUBSETS[@]}; ui++)); do
    subset=${SUBSETS[$ui]}; out="$ROOT/$stage/subset_$subset"
    if [[ -e "$out" && $OVERWRITE -eq 0 && -z "$RESUME" ]]; then echo "refusing existing result directory: $out" >&2; exit 1; fi
    CMD=(python diagnose_stage1_content_generalization.py --config configs/train_stage1_fixed_tx_uniform.yaml --stage "$stage" --subset-size "$subset" --steps "$STEPS" --seed 23 --output-dir "$out" --device "$DEVICE" --checkpoint-every 250 --validation-every 100 --allow-long-run)
    [[ $OVERWRITE -eq 1 ]] && CMD+=(--overwrite)
    if [[ -n "$RESUME" && $si -eq $stage_start && $ui -eq $local_start ]]; then CMD+=(--resume "$RESUME"); fi
    if [[ $DRY -eq 1 ]]; then printf '%q ' "${CMD[@]}"; echo; continue; fi
    mkdir -p "$(dirname "$out")"; PENDING_LOG="${out}.run.log"
    "${CMD[@]}" 2>&1 | tee "$PENDING_LOG"; mv "$PENDING_LOG" "$out/run.log"
    decision=$(python - "$out/summary.json" "$stage" "$subset" <<'PY'
import json,sys
from speech_jscc.diagnostics.content_generalization import ladder_decision
print(ladder_decision(sys.argv[2],sys.argv[3],bool(json.load(open(sys.argv[1]))["gate"]["passed"])))
PY
)
    python scripts/summarize_content_generalization.py --root "$ROOT"
    if [[ $decision == next_stage ]]; then stage_passed=1; break; fi
    if [[ $decision == stop_first_failing_stage ]]; then echo "stop_first_failing_stage: $stage" >&2; exit 1; fi
    RESUME=""
  done
  [[ $DRY -eq 1 ]] && continue
  [[ $stage_passed -eq 1 ]] || { echo "stage did not pass: $stage" >&2; exit 1; }
  subset_start=0
done
