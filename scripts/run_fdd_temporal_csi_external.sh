#!/usr/bin/env bash
set -euo pipefail

device=cuda
overwrite=0
dry_run=0
utterances=64
realizations=2
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) device="$2"; shift 2 ;;
    --utterances) utterances="$2"; shift 2 ;;
    --realizations) realizations="$2"; shift 2 ;;
    --overwrite) overwrite=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

root="runs/mmse_csi_interleaving"
checkpoint="runs/waveform_aware_wireless/clean_channel_training/best_waveform_si_sdr.pt"
cmd=(python evaluate_fdd_temporal_csi.py
  --config configs/fdd_temporal_csi.yaml
  --checkpoint "$checkpoint"
  --output-root "$root"
  --device "$device"
  --utterances "$utterances"
  --realizations "$realizations"
  --allow-long-run)
if [[ $overwrite -eq 1 ]]; then cmd+=(--overwrite); fi
if [[ $dry_run -eq 1 ]]; then
  printf '%q ' "${cmd[@]}"; printf '\n'
  exit 0
fi

mkdir -p "$root"
{
  printf '%q ' "${cmd[@]}"; printf '\n'
} > "$root/command.txt"
git rev-parse HEAD > "$root/git_commit.txt"
git status --porcelain > "$root/working_tree_status.txt"
python -m pip freeze > "$root/environment.txt"
cp configs/fdd_temporal_csi.yaml "$root/resolved_config.yaml"
"${cmd[@]}" 2>&1 | tee "$root/run.log"
