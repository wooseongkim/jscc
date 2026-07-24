#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
device=cuda; steps=4096; batch=4; overwrite=0; dry=0; resume=""
root=runs/stage1_conv_conformer_jammer/j5_pilot_targeted; out="$root/training/j4_transfer"; dist="$root/selected_training_distribution.json"
while (($#)); do case "$1" in
 --device) device="$2"; shift 2;; --steps) steps="$2"; shift 2;; --batch-size) batch="$2"; shift 2;;
 --resume) resume="$2"; shift 2;; --overwrite) overwrite=1; shift;; --dry-run) dry=1; shift;; *) echo "unknown argument: $1" >&2; exit 2;; esac; done
cmd=(python train_j5_conv_conformer.py --config configs/conv_conformer_j5_pilot_targeted.yaml --selected-distribution "$dist" --j4-manifest runs/stage1_conv_conformer_jammer/j4_random_burst/accepted_manifest.json --parent-checkpoint runs/stage1_conv_conformer_jammer/j4_random_burst/training/j3_transfer/diagnostic_last.pt --output-dir "$out" --device "$device" --steps "$steps" --batch-size "$batch" --allow-long-run)
[[ -n "$resume" ]] && cmd+=(--resume "$resume"); ((overwrite)) && cmd+=(--overwrite)
if ((dry)); then printf '%q ' "${cmd[@]}"; echo; exit 0; fi
[[ -f "$dist" ]] || { echo "boundary-selected distribution missing: $dist" >&2; exit 1; }
[[ ! -e "$out" || $overwrite -eq 1 || -n "$resume" ]] || { echo "refusing existing output directory: $out" >&2; exit 1; }
mkdir -p "$(dirname "$out")"; log="${out}.run.log"; "${cmd[@]}" 2>&1 | tee "$log"; mv "$log" "$out/run.log"
