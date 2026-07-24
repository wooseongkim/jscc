#!/usr/bin/env bash
set -euo pipefail

device=cuda
overwrite=0
dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) device="$2"; shift 2 ;;
    --overwrite) overwrite=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

cmd=(python compare_oracle_csi.py
  --checkpoint runs/waveform_aware_wireless/clean_channel_training/best_waveform_si_sdr.pt
  --output-dir runs/oracle_csi_comparison
  --device "$device" --utterances 64 --realizations 2 --allow-long-run)
if [[ $overwrite -eq 1 ]]; then cmd+=(--overwrite); fi
if [[ $dry_run -eq 1 ]]; then
  printf '%q ' "${cmd[@]}"; printf '\n'
  exit 0
fi
mkdir -p runs/oracle_csi_comparison
"${cmd[@]}" 2>&1 | tee runs/oracle_csi_comparison/run.log
