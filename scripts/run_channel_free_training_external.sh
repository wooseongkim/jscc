#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
device=cuda;steps=4096;batch=4;overwrite=0;dry=0;resume="";out=runs/stage1_conv_conformer_clean_end_to_end/channel_free_training/fresh
while (($#));do case "$1" in --device)device="$2";shift 2;;--steps)steps="$2";shift 2;;--batch-size)batch="$2";shift 2;;--resume)resume="$2";shift 2;;--overwrite)overwrite=1;shift;;--dry-run)dry=1;shift;;*)echo "unknown argument: $1" >&2;exit 2;;esac;done
cmd=(python train_channel_free_conv_conformer.py --config configs/conv_conformer_clean_end_to_end.yaml --pre-status runs/stage1_conv_conformer_clean_end_to_end/root_cause_summary.json --output-dir "$out" --device "$device" --subset-size 256 --steps "$steps" --batch-size "$batch" --allow-long-run);[[ -n "$resume" ]]&&cmd+=(--resume "$resume");((overwrite))&&cmd+=(--overwrite)
if ((dry));then printf '%q ' "${cmd[@]}";echo;exit 0;fi
[[ ! -e "$out"||$overwrite -eq 1||-n "$resume" ]]||{ echo "refusing existing output directory: $out" >&2;exit 1;};mkdir -p "$(dirname "$out")";log="${out}.run.log";"${cmd[@]}" 2>&1|tee "$log";mv "$log" "$out/run.log"
