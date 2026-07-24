#!/usr/bin/env bash
set -euo pipefail
DEVICE=auto; DRY=0; OVERWRITE=0; CONTINUE=0; STAGE=j1_weak_barrage; RESUME=""; ROOT="runs/stage1_random_distribution"
while (($#)); do case "$1" in
  --device) DEVICE=$2; shift 2;; --dry-run) DRY=1; shift;; --overwrite) OVERWRITE=1; shift;;
  --continue_on_pass) CONTINUE=1; shift;; --stage) STAGE=$2; shift 2;; --resume) RESUME=$2; shift 2;;
  --root) ROOT=$2; shift 2;; *) echo "unknown option: $1" >&2; exit 2;; esac; done
STAGES=(j1_weak_barrage j2_moderate_barrage j3_strong_barrage j4_mixed_sparse j5_full_mixture)
PARENTS=(o6_random_clean j1_weak_barrage j2_moderate_barrage j3_strong_barrage j4_mixed_sparse)
start=-1; for i in "${!STAGES[@]}"; do [[ ${STAGES[$i]} == "$STAGE" ]] && start=$i; done
[[ $start -ge 0 ]] || { echo "unknown stage: $STAGE" >&2; exit 2; }
for ((i=start; i<${#STAGES[@]}; i++)); do
  stage=${STAGES[$i]}; parent=${PARENTS[$i]}; out="$ROOT/$stage"; checkpoint="$ROOT/$parent/diagnostic_last.pt"
  if [[ -e "$out" && $OVERWRITE -eq 0 && -z "$RESUME" ]]; then echo "refusing existing result directory: $out" >&2; exit 1; fi
  CMD=(python diagnose_stage1_random_distribution.py --config configs/train_stage1_fixed_tx_uniform.yaml --stage "$stage" --steps 1000 --seed 23 --output-dir "$out" --device "$DEVICE" --checkpoint-every 250 --validation-every 100 --allow-long-run --initialization-mode curriculum_resume)
  if [[ -n "$RESUME" && $i -eq $start ]]; then CMD+=(--resume "$RESUME"); else CMD+=(--parent-checkpoint "$checkpoint"); fi
  [[ $OVERWRITE -eq 1 ]] && CMD+=(--overwrite)
  if [[ $DRY -eq 1 ]]; then printf '%q ' "${CMD[@]}"; echo; [[ $CONTINUE -eq 1 ]] && continue || break; fi
  if [[ -n "$RESUME" && $i -eq $start ]]; then
    [[ -f "$RESUME" ]] || { echo "missing resume checkpoint: $RESUME" >&2; exit 1; }
  else
    [[ -f "$checkpoint" ]] || { echo "missing parent checkpoint: $checkpoint" >&2; exit 1; }
    parent_summary="$ROOT/$parent/summary.json"
    [[ -f "$parent_summary" ]] || { echo "missing parent summary: $parent_summary" >&2; exit 1; }
    python - "$parent_summary" <<'PY'
import json,sys
if not json.load(open(sys.argv[1]))["gate"]["passed"]:
    print(f"parent stage gate failed: {sys.argv[1]}", file=sys.stderr)
    raise SystemExit(1)
PY
  fi
  PENDING_LOG="${out}.run.log"
  "${CMD[@]}" 2>&1 | tee "$PENDING_LOG"
  mv "$PENDING_LOG" "$out/run.log"
  python - "$out/summary.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["gate"]["passed"] else 1)
PY
  [[ $CONTINUE -eq 1 ]] || { echo "stage passed; rerun with --continue_on_pass to advance"; break; }
done
