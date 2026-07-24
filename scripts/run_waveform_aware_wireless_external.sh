#!/usr/bin/env bash
set -euo pipefail

device=cuda
overwrite=0
dry_run=0
resume=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) device="$2"; shift 2 ;;
    --overwrite) overwrite=1; shift ;;
    --resume) resume="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

checkpoint="runs/channel_free_revalidation/cf2_50frames_1920/best_waveform_si_sdr.pt"
root="runs/waveform_aware_wireless"
eval_cmd=(python eval_waveform_aware_wireless.py --mode all --checkpoint "$checkpoint"
  --output-root "$root" --device "$device" --utterances 64
  --realizations-per-utterance 2 --allow-long-run)
train_cmd=(python train_waveform_aware_clean_channel.py --checkpoint "$checkpoint"
  --zero-shot-summary "$root/clean_channel_zero_shot/summary.json"
  --output-dir "$root/clean_channel_training" --device "$device" --steps 20000
  --allow-long-run)
if [[ $overwrite -eq 1 ]]; then
  eval_cmd+=(--overwrite)
  train_cmd+=(--overwrite)
fi
if [[ -n "$resume" ]]; then
  train_cmd+=(--resume "$resume")
fi

if [[ $dry_run -eq 1 ]]; then
  printf '%q ' "${eval_cmd[@]}"; printf '\n'
  printf '%q ' "${train_cmd[@]}"; printf '\n'
  echo "conditional checkpoints: best_summed_latent_nmse.pt best_waveform_si_sdr.pt last.pt"
  exit 0
fi

mkdir -p "$root"
if [[ -z "$resume" ]]; then
  "${eval_cmd[@]}" 2>&1 | tee "$root/evaluation.log"
elif [[ ! -f "$root/clean_channel_zero_shot/summary.json" ]]; then
  echo "resume requires existing zero-shot summary: $root/clean_channel_zero_shot/summary.json" >&2
  exit 1
fi

if python - "$root/clean_channel_zero_shot/summary.json" <<'PY'
import json,sys
passed=bool(json.load(open(sys.argv[1]))["random"]["gate"]["passed"])
raise SystemExit(0 if not passed else 1)
PY
then
  "${train_cmd[@]}" 2>&1 | tee "$root/clean_channel_training.log"
else
  echo "random clean zero-shot passed; fine-tuning is not required"
fi
