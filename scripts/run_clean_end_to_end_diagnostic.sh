#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
device=cuda;samples=16;overwrite=0;dry=0;root=runs/stage1_conv_conformer_clean_end_to_end
while (($#));do case "$1" in --device)device="$2";shift 2;;--samples)samples="$2";shift 2;;--overwrite)overwrite=1;shift;;--dry-run)dry=1;shift;;*)echo "unknown argument: $1" >&2;exit 2;;esac;done
cmd=(python diagnose_clean_end_to_end.py --config configs/conv_conformer_clean_end_to_end.yaml --j4-checkpoint runs/stage1_conv_conformer_jammer/j4_random_burst/training/j3_transfer/diagnostic_last.pt --j5-checkpoint runs/stage1_conv_conformer_jammer/j5_pilot_targeted/training/j4_transfer/diagnostic_last.pt --output-root "$root" --device "$device" --samples "$samples" --allow-long-run);((overwrite))&&cmd+=(--overwrite)
if ((dry));then printf '%q ' "${cmd[@]}";echo;exit 0;fi
[[ ! -e "$root/pre_diagnostic_status.json"||$overwrite -eq 1 ]]||{ echo "refusing existing diagnostic root: $root" >&2;exit 1;};mkdir -p "$(dirname "$root")";log="${root}.run.log";"${cmd[@]}" 2>&1|tee "$log";mv "$log" "$root/run.log"
