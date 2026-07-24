#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
device=auto;batch=4;workers=0;subset=256;epochs=64;resume="";overwrite=0;dry=0;root=runs/stage1_conv_conformer_integration;prev="$root/g1_mapping_train/subset_$subset/summary.json";out="$root/g2_fixed_clean/subset_$subset"
while (($#));do case "$1" in --device)device="$2";shift 2;;--batch-size)batch="$2";shift 2;;--num-workers)workers="$2";shift 2;;--subset-size)subset="$2";prev="$root/g1_mapping_train/subset_$2/summary.json";out="$root/g2_fixed_clean/subset_$2";shift 2;;--max-epochs)epochs="$2";shift 2;;--resume)resume="$2";shift 2;;--overwrite)overwrite=1;shift;;--dry-run)dry=1;shift;;*)echo "unknown argument: $1" >&2;exit 2;;esac;done
cmd=(python diagnose_stage1_content_generalization.py --config configs/conv_conformer_integration_v1.yaml --stage g2_fixed_clean --subset-size "$subset" --max-epochs "$epochs" --batch-size "$batch" --num-workers "$workers" --seed 23 --output-dir "$out" --device "$device" --checkpoint-every 250 --validation-every 100 --allow-long-run);[[ -n "$resume" ]]&&cmd+=(--resume "$resume");((overwrite))&&cmd+=(--overwrite)
if ((dry));then printf '%q ' "${cmd[@]}";echo;exit 0;fi
python - "$prev" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]));assert p["gate"]["stage_pass"],"G1 stage_pass is required"
PY
[[ ! -e "$out"||$overwrite -eq 1||-n "$resume" ]]||{ echo "refusing existing output directory: $out" >&2;exit 1;};mkdir -p "$(dirname "$out")";"${cmd[@]}" 2>&1|tee "${out}.run.log";mv "${out}.run.log" "$out/run.log"
