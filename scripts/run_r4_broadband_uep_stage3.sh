#!/usr/bin/env bash
# Run the frozen Stage-3 full expanded-selection evaluation: U0 plus four
# profiles chosen from the completed Stage-2 confirmation.
set -euo pipefail

run_root="runs/waveform_aware_wireless/r4_broadband_uep_optimization"
manifest_path="$run_root/stage2_confirmation/stage3_top4_profiles.json"
output_path="$run_root/stage3_full_selection"

if [[ ! -f "$manifest_path" ]]; then
  echo "missing Stage-3 profile manifest: $manifest_path" >&2
  exit 2
fi

mapfile -t profile_ids < <(
  python - "$manifest_path" <<'PY'
from pathlib import Path
import json
import sys

profiles = json.loads(Path(sys.argv[1]).read_text())["profiles"]
for profile_id in sorted(key for key in profiles if key != "U0"):
    print(profile_id)
PY
)

if [[ ${#profile_ids[@]} -ne 4 ]]; then
  echo "expected four non-U0 Stage-3 profiles; found ${#profile_ids[@]}" >&2
  exit 2
fi

exec python evaluate_r4_broadband_uep_profiles.py \
  --config configs/eval_r4_broadband_uep_profiles.yaml \
  --checkpoint runs/waveform_aware_wireless/r4_si_sdr_finetune/si_sdr_medium/local_step_003000.pt \
  --split expanded_selection \
  --profile-json "$manifest_path" \
  --profiles U0 "${profile_ids[@]}" \
  --jammer-type broadband_awgn \
  --jsr-db no_jammer -5 0 5 10 \
  --snr-db 5 10 15 \
  --max-utterances 48 \
  --max-realizations 2 \
  --device cuda \
  --allow-long-run \
  --output-dir "$output_path" \
  --overwrite
