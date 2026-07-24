#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
device=cuda;samples=64;overwrite=0;dry=0;out=runs/stage1_conv_conformer_clean_end_to_end/clean_channel_ladder
while (($#));do case "$1" in --device)device="$2";shift 2;;--samples)samples="$2";shift 2;;--overwrite)overwrite=1;shift;;--dry-run)dry=1;shift;;*)echo "unknown argument: $1" >&2;exit 2;;esac;done
cmd=(python eval_clean_channel_ladder.py --config configs/conv_conformer_clean_end_to_end.yaml --checkpoint runs/stage1_conv_conformer_clean_end_to_end/channel_free_training/fresh/best_waveform.pt --output-dir "$out" --device "$device" --samples "$samples" --allow-long-run);((overwrite))&&cmd+=(--overwrite)
if ((dry));then printf '%q ' "${cmd[@]}";echo;exit 0;fi
[[ -f runs/stage1_conv_conformer_clean_end_to_end/channel_free_training/fresh/best_waveform.pt ]]||{ echo "best waveform checkpoint missing" >&2;exit 1;};[[ ! -e "$out"||$overwrite -eq 1 ]]||{ echo "refusing existing output directory: $out" >&2;exit 1;};mkdir -p "$(dirname "$out")";log="${out}.run.log";"${cmd[@]}" 2>&1|tee "$log";mv "$log" "$out/run.log"
