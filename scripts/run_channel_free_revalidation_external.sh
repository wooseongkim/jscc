#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

experiment=""
device="cuda"
batch_size="4"
steps=""
initialization="fresh"
resume=""
overwrite=0
dry_run=0
root="runs/channel_free_revalidation"

while (($#)); do
  case "$1" in
    --experiment) experiment="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --steps) steps="$2"; shift 2 ;;
    --initialization) initialization="$2"; shift 2 ;;
    --resume) resume="$2"; shift 2 ;;
    --overwrite) overwrite=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$experiment" ]]; then
  echo "--experiment is required: baseline, cf1, cf2, cf3, cf4, cf5, or eval" >&2
  exit 2
fi

run_logged() {
  local output="$1"; shift
  local -a command=("$@")
  if ((dry_run)); then
    printf '%q ' "${command[@]}"
    echo
    return
  fi
  if [[ -e "$output" && $overwrite -ne 1 && -z "$resume" ]]; then
    echo "refusing existing output directory: $output" >&2
    return 1
  fi
  mkdir -p "$root"
  local log="${output}.run.log"
  "${command[@]}" 2>&1 | tee "$log"
  mv "$log" "$output/run.log"
}

run_train() {
  local cf="$1"
  local name="$2"
  local output="$root/$name"
  local -a command=(python train_channel_free_conv_conformer.py
    --config configs/channel_free_revalidation.yaml
    --experiment "$cf"
    --output-dir "$output"
    --device "$device"
    --batch-size "$batch_size"
    --initialization "$initialization"
    --allow-long-run)
  [[ -n "$steps" ]] && command+=(--steps "$steps")
  [[ -n "$resume" ]] && command+=(--resume "$resume")
  ((overwrite)) && command+=(--overwrite)
  ((dry_run)) && command+=(--dry-run)
  run_logged "$output" "${command[@]}"
}

case "$experiment" in
  baseline)
    for kind in official continuous_sum; do
      output="$root/baseline_${kind}"
      command=(python eval_channel_free_conv_conformer.py
        --config configs/channel_free_revalidation.yaml --mode baseline
        --baseline-kind "$kind" --output-dir "$output"
        --device "$device" --samples 64 --allow-long-run)
      ((overwrite)) && command+=(--overwrite)
      ((dry_run)) && command+=(--dry-run)
      run_logged "$output" "${command[@]}"
    done
    ;;
  cf1) run_train cf1 cf1_30frames_1920 ;;
  cf2) run_train cf2 cf2_50frames_1920 ;;
  cf3) run_train cf3 cf3_50frames_3200 ;;
  cf4) run_train cf4 cf4_large_model ;;
  cf5)
    [[ -n "$resume" ]] || {
      echo "CF-5 requires --resume pointing to the selected CF-1–CF-4 checkpoint" >&2
      exit 2
    }
    [[ -n "$steps" ]] || steps=32000
    source_cf="${CF5_SOURCE_EXPERIMENT:-cf4}"
    run_train "$source_cf" cf5_long_training
    ;;
  eval)
    output="$root/final_comparison"
    command=(python eval_channel_free_conv_conformer.py
      --config configs/channel_free_revalidation.yaml --mode final
      --output-dir "$output" --device "$device" --samples 64 --allow-long-run)
    for item in \
      "cf1=$root/cf1_30frames_1920/best_waveform_si_sdr.pt" \
      "cf2=$root/cf2_50frames_1920/best_waveform_si_sdr.pt" \
      "cf3=$root/cf3_50frames_3200/best_waveform_si_sdr.pt" \
      "cf4=$root/cf4_large_model/best_waveform_si_sdr.pt" \
      "cf5=$root/cf5_long_training/best_waveform_si_sdr.pt"; do
      [[ -f "${item#*=}" ]] && command+=(--checkpoint "$item")
    done
    ((overwrite)) && command+=(--overwrite)
    ((dry_run)) && command+=(--dry-run)
    run_logged "$output" "${command[@]}"
    ;;
  *)
    echo "--experiment must be baseline, cf1, cf2, cf3, cf4, cf5, or eval" >&2
    exit 2
    ;;
esac
