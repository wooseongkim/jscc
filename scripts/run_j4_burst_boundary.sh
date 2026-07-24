#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/jscc-matplotlib";mkdir -p "$MPLCONFIGDIR"
device=cuda;realizations=16;overwrite=0;dry=0;root=runs/stage1_conv_conformer_jammer/j4_random_burst;out="$root/boundary_sweep"
while (($#));do case "$1" in --device)device="$2";shift 2;;--realizations)realizations="$2";shift 2;;--overwrite)overwrite=1;shift;;--dry-run)dry=1;shift;;*)echo "unknown argument: $1" >&2;exit 2;;esac;done
cmd=(python diagnose_j4_burst_boundary.py --config configs/conv_conformer_j4_random_burst.yaml --j3-manifest runs/stage1_conv_conformer_jammer/j3_random_narrowband/training/j2_transfer/accepted_manifest.json --j3-checkpoint runs/stage1_conv_conformer_jammer/j3_random_narrowband/training/j2_transfer/diagnostic_last.pt --output-dir "$out" --device "$device" --realizations "$realizations" --allow-long-run);((overwrite))&&cmd+=(--overwrite)
if ((dry));then printf '%q ' "${cmd[@]}";echo;exit 0;fi
[[ ! -e "$out"||$overwrite -eq 1 ]]||{ echo "refusing existing output directory: $out" >&2;exit 1;};mkdir -p "$root";log="$root/.boundary.log";"${cmd[@]}" 2>&1|tee "$log";mv "$log" "$out/run.log"
