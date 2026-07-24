#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
device=auto;batch=4;workers=0;subset=256;epochs=64;resume="";overwrite=0;dry=0;out=runs/stage1_conv_conformer_integration/g1_mapping_train
while (($#));do case "$1" in --device)device="$2";shift 2;;--batch-size)batch="$2";shift 2;;--num-workers)workers="$2";shift 2;;--subset-size)subset="$2";shift 2;;--max-epochs)epochs="$2";shift 2;;--resume)resume="$2";shift 2;;--overwrite)overwrite=1;shift;;--dry-run)dry=1;shift;;*)echo "unknown argument: $1" >&2;exit 2;;esac;done
cmd=(python diagnose_stage1_content_generalization.py --config configs/conv_conformer_integration_v1.yaml --stage g1_pilot_reserved_identity --subset-size "$subset" --max-epochs "$epochs" --batch-size "$batch" --num-workers "$workers" --seed 23 --output-dir "$out/subset_$subset" --device "$device" --checkpoint-every 250 --validation-every 100 --allow-long-run); [[ -n "$resume" ]]&&cmd+=(--resume "$resume");((overwrite))&&cmd+=(--overwrite)
if ((dry));then printf '%q ' "${cmd[@]}";echo;exit 0;fi
python - <<'PY'
import json
p=json.load(open("runs/stage1_conv_conformer_integration/g1_mapping_equivalence/equivalence_summary.json"))
assert p["passed"],"G1-E equivalence pass is required"
PY
target="$out/subset_$subset";[[ ! -e "$target"||$overwrite -eq 1||-n "$resume" ]]||{ echo "refusing existing output directory: $target" >&2;exit 1;};mkdir -p "$out";"${cmd[@]}" 2>&1|tee "${target}.run.log";mv "${target}.run.log" "$target/run.log"
