#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
device=cuda;steps=4096;overwrite=0;dry=0;resume="";batch=4
root=runs/stage1_conv_conformer_jammer/j2_strong_barrage_boundary;range="$root/selected_training_range.json";decision="$root/initialization_decision.json"
while (($#));do case "$1" in --device)device="$2";shift 2;;--steps)steps="$2";shift 2;;--batch-size)batch="$2";shift 2;;--resume)resume="$2";shift 2;;--overwrite)overwrite=1;shift;;--dry-run)dry=1;shift;;*)echo "unknown argument: $1" >&2;exit 2;;esac;done
if ((dry));then echo "python train_j2_conv_conformer.py --stage j2_strong_barrage --steps $steps --selected-range $range --initialization-mode <from-$decision> --output-dir $root/training/<selected-mode>";exit 0;fi
[[ -f "$range"&&-f "$decision" ]]||{ echo "boundary range and initialization decision are required" >&2;exit 1;}
mode="$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["selected_initialization"])' "$decision")";out="$root/training/$mode"
cmd=(python train_j2_conv_conformer.py --config configs/conv_conformer_j2_strong_barrage.yaml --stage j2_strong_barrage --subset-size 256 --steps "$steps" --batch-size "$batch" --seed 23 --output-dir "$out" --device "$device" --selected-range "$range" --initialization-mode "$mode" --j1-summary runs/stage1_conv_conformer_jammer/j1_weak_random_barrage/subset_256/summary.json --checkpoint-every 500 --validation-every 100 --allow-long-run)
[[ "$mode" == j1_transfer ]]&&cmd+=(--parent-checkpoint runs/stage1_conv_conformer_jammer/j1_weak_random_barrage/subset_256/diagnostic_last.pt);[[ -n "$resume" ]]&&cmd+=(--resume "$resume");((overwrite))&&cmd+=(--overwrite)
[[ ! -e "$out"||$overwrite -eq 1||-n "$resume" ]]||{ echo "refusing existing output directory: $out" >&2;exit 1;};mkdir -p "$(dirname "$out")";log="${out}.run.log";"${cmd[@]}" 2>&1|tee "$log";mv "$log" "$out/run.log"
