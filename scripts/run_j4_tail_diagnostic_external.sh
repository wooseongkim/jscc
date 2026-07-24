#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
device=cuda; output=runs/stage1_conv_conformer_jammer/j4_random_burst/tail_diagnostic; overwrite=; dry_run=
while (($#)); do
  case "$1" in
    --device) device="$2"; shift 2;;
    --output-dir) output="$2"; shift 2;;
    --overwrite) overwrite=--overwrite; shift;;
    --dry-run) dry_run=--dry-run; shift;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
if [[ -d "$output" && -z "$overwrite" && -z "$dry_run" ]]; then echo "refusing existing output directory: $output" >&2; exit 1; fi
command=(python diagnose_j4_tail_failure.py
  --config configs/conv_conformer_j4_random_burst.yaml
  --diagnostic-config configs/j4_failure_diagnostic.yaml
  --j3-checkpoint runs/stage1_conv_conformer_jammer/j3_random_narrowband/training/j2_transfer/diagnostic_last.pt
  --j4-checkpoint runs/stage1_conv_conformer_jammer/j4_random_burst/training/j3_transfer/diagnostic_last.pt
  --j4-summary runs/stage1_conv_conformer_jammer/j4_random_burst/training/j3_transfer/summary.json
  --selected-distribution runs/stage1_conv_conformer_jammer/j4_random_burst/selected_training_distribution.json
  --output-dir "$output" --device "$device" --unseen-utterances 64 --realizations-per-utterance 2 --allow-long-run)
[[ -n "$overwrite" ]] && command+=("$overwrite")
[[ -n "$dry_run" ]] && command+=("$dry_run")
printf '%q ' "${command[@]}"; printf '\n'
if [[ -n "$dry_run" ]]; then
  "${command[@]}"
else
  mkdir -p "$(dirname "$output")"
  temporary_log="${output}.run.log"
  "${command[@]}" 2>&1 | tee "$temporary_log"
  mv "$temporary_log" "$output/run.log"
fi
