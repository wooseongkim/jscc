#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
device=cuda; realizations=16; overwrite=0; dry=0
root=runs/stage1_conv_conformer_jammer/j5_pilot_targeted; out="$root/boundary_sweep"
while (($#)); do case "$1" in
  --device) device="$2"; shift 2;; --realizations) realizations="$2"; shift 2;;
  --overwrite) overwrite=1; shift;; --dry-run) dry=1; shift;; *) echo "unknown argument: $1" >&2; exit 2;; esac; done
cmd=(python diagnose_j5_pilot_boundary.py --config configs/conv_conformer_j5_pilot_targeted.yaml --j4-manifest runs/stage1_conv_conformer_jammer/j4_random_burst/accepted_manifest.json --j4-checkpoint runs/stage1_conv_conformer_jammer/j4_random_burst/training/j3_transfer/diagnostic_last.pt --output-dir "$out" --device "$device" --realizations "$realizations" --allow-long-run)
((overwrite)) && cmd+=(--overwrite)
if ((dry)); then printf '%q ' "${cmd[@]}"; echo; exit 0; fi
[[ ! -e "$out" || $overwrite -eq 1 ]] || { echo "refusing existing output directory: $out" >&2; exit 1; }
mkdir -p "$root"; log="$root/.boundary.run.log"; "${cmd[@]}" 2>&1 | tee "$log"; mv "$log" "$out/run.log"
