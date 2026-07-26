#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

device=cuda
output_dir=runs/waveform_aware_wireless/r4_repetition3_mrc_finetune
resume=
overwrite=0
dry_run=0
smoke_steps=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) device="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --resume) resume="$2"; shift 2 ;;
    --smoke-steps) smoke_steps="$2"; shift 2 ;;
    --overwrite) overwrite=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

args=(--config configs/train_r4_waveform_finetune.yaml --device "$device" --output-dir "$output_dir")
[[ -n "$resume" ]] && args+=(--resume "$resume")
[[ -n "$smoke_steps" ]] && args+=(--smoke-steps "$smoke_steps")
(( overwrite )) && args+=(--overwrite)
if (( dry_run )); then
  args+=(--dry-run)
elif [[ -z "$smoke_steps" ]]; then
  args+=(--allow-long-run)
fi
mkdir -p "$(dirname "$output_dir")"
python train_r4_waveform_finetune.py "${args[@]}" 2>&1 | tee "${output_dir%/}_external.log"

