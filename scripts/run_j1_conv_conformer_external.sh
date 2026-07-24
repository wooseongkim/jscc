#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

device=cuda
subset=256
batch=4
epochs=64
workers=0
resume=""
overwrite=0
dry_run=0
root=runs/stage1_conv_conformer_jammer/j1_weak_random_barrage

while (($#)); do
  case "$1" in
    --device) device="$2"; shift 2 ;;
    --subset-size) subset="$2"; shift 2 ;;
    --batch-size) batch="$2"; shift 2 ;;
    --max-epochs) epochs="$2"; shift 2 ;;
    --num-workers) workers="$2"; shift 2 ;;
    --resume) resume="$2"; shift 2 ;;
    --overwrite) overwrite=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

out="$root/subset_$subset"
g3_summary="runs/stage1_conv_conformer_integration/g3_random_clean/subset_$subset/summary.json"
cmd=(python diagnose_stage1_content_generalization.py
  --config configs/conv_conformer_integration_v1.yaml
  --stage j1_weak_random_barrage
  --subset-size "$subset"
  --max-epochs "$epochs"
  --batch-size "$batch"
  --num-workers "$workers"
  --seed 23
  --output-dir "$out"
  --device "$device"
  --checkpoint-every 250
  --validation-every 100
  --allow-long-run)
[[ -n "$resume" ]] && cmd+=(--resume "$resume")
((overwrite)) && cmd+=(--overwrite)

if ((dry_run)); then
  printf '%q ' "${cmd[@]}"
  echo
  exit 0
fi

python - "$g3_summary" <<'PY'
import json, sys
path = sys.argv[1]
payload = json.load(open(path))
assert payload["gate"]["stage_pass"], "G3 stage_pass is required before J1"
PY

[[ ! -e "$out" || $overwrite -eq 1 || -n "$resume" ]] || {
  echo "refusing existing output directory: $out" >&2
  exit 1
}
mkdir -p "$root"
temporary_log="$root/.subset_${subset}.run.log"
"${cmd[@]}" 2>&1 | tee "$temporary_log"
mv "$temporary_log" "$out/run.log"

echo "J1 completed. Inspect: $out/summary.json"
echo "This script never starts a later jammer stage."
