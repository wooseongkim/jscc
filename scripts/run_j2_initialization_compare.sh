#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
device=cuda;steps=512;overwrite=0;dry=0;resume=""
root=runs/stage1_conv_conformer_jammer/j2_strong_barrage_boundary;range="$root/selected_training_range.json"
while (($#));do case "$1" in --device)device="$2";shift 2;;--steps)steps="$2";shift 2;;--resume)resume="$2";shift 2;;--overwrite)overwrite=1;shift;;--dry-run)dry=1;shift;;*)echo "unknown argument: $1" >&2;exit 2;;esac;done
j1s=runs/stage1_conv_conformer_jammer/j1_weak_random_barrage/subset_256/summary.json;j1c=runs/stage1_conv_conformer_jammer/j1_weak_random_barrage/subset_256/diagnostic_last.pt
run_one(){ mode="$1";out="$root/initialization_compare/$mode";cmd=(python train_j2_conv_conformer.py --config configs/conv_conformer_j2_strong_barrage.yaml --stage j2_strong_barrage --subset-size 256 --steps "$steps" --batch-size 4 --seed 23 --output-dir "$out" --device "$device" --selected-range "$range" --initialization-mode "$mode" --j1-summary "$j1s" --checkpoint-every 250 --validation-every 100 --allow-long-run);[[ "$mode" == j1_transfer ]]&&cmd+=(--parent-checkpoint "$j1c");[[ -n "$resume" ]]&&cmd+=(--resume "$resume");((overwrite))&&cmd+=(--overwrite);if ((dry));then printf '%q ' "${cmd[@]}";echo;return;fi;[[ ! -e "$out"||$overwrite -eq 1||-n "$resume" ]]||{ echo "refusing existing output directory: $out" >&2;return 1;};mkdir -p "$(dirname "$out")";log="${out}.run.log";"${cmd[@]}" 2>&1|tee "$log";mv "$log" "$out/run.log";}
run_one fresh;run_one j1_transfer
if ((!dry));then python select_j2_initialization.py --fresh-summary "$root/initialization_compare/fresh/summary.json" --transfer-summary "$root/initialization_compare/j1_transfer/summary.json" --output "$root/initialization_decision.json";fi
