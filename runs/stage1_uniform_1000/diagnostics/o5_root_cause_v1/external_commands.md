# External commands

```bash
#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
root="runs/stage1_uniform_1000/diagnostics/o5_root_cause_v1"
overwrite=0
if [[ "${1:-}" == "--overwrite" ]]; then overwrite=1; shift; fi
if [[ -e "$root" && "$overwrite" -ne 1 ]]; then echo "Refusing existing $root; pass --overwrite" >&2; exit 2; fi
mkdir -p "$root"
conditions=(clean_awgn_reference full_barrage_estimated_csi full_barrage_oracle_csi data_only_barrage_estimated_csi data_only_barrage_oracle_csi pilot_only_jammer_estimated_csi full_barrage_oracle_subtraction)
for condition in "${conditions[@]}"; do
  out="$root/$condition"
  extra=(); [[ "$overwrite" -eq 1 ]] && extra+=(--overwrite)
  log_tmp="$root/.${condition}.run.log"
  python diagnose_o5_root_cause.py --config configs/train_stage1_fixed_tx_uniform.yaml --condition "$condition" --steps 500 --seed 23 --output_dir "$out" --allow_long_run "${extra[@]}" 2>&1 | tee "$log_tmp"
  mv "$log_tmp" "$out/run.log"
done
python scripts/summarize_o5_root_cause.py --root "$root"

# Optional sensitivity matrix (print only; do not mix with primary results).
for jsr in -10 -5 0; do
  for condition in full_barrage_estimated_csi full_barrage_oracle_csi data_only_barrage_estimated_csi; do
    echo "python diagnose_o5_root_cause.py --config configs/train_stage1_fixed_tx_uniform.yaml --condition $condition --steps 500 --seed 23 --requested_jsr_db $jsr --output_dir $root/sensitivity_jsr${jsr}/$condition --allow_long_run"
  done
done

#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
root="runs/stage1_uniform_1000/diagnostics/o5_root_cause_v1"
execute=0
if [[ "${1:-}" == "--execute" ]]; then execute=1; shift; fi
emit() { echo "$*"; [[ "$execute" -eq 1 ]] && bash -lc "$*"; }
for condition in full_barrage_estimated_csi full_barrage_oracle_csi data_only_barrage_estimated_csi; do
  summary="$root/$condition/summary.json"
  [[ -f "$summary" ]] || { echo "missing $summary" >&2; continue; }
  status="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["plateau_status"])' "$summary")"
  if [[ "$condition" != "full_barrage_estimated_csi" && "$status" != "optimization_still_progressing" ]]; then
    echo "skip $condition extension: plateau_status=$status"
    continue
  fi
  cmd="python diagnose_o5_root_cause.py --config configs/train_stage1_fixed_tx_uniform.yaml --condition $condition --steps 1000 --seed 23 --output_dir $root/$condition --resume $root/$condition/diagnostic_last.pt --allow_long_run"
  emit "$cmd"
done
c1_summary="$root/full_barrage_estimated_csi/summary.json"
if [[ -f "$c1_summary" ]] && [[ "$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["plateau_status"])' "$c1_summary")" == "optimization_still_progressing" ]]; then
  emit "python diagnose_o5_root_cause.py --config configs/train_stage1_fixed_tx_uniform.yaml --condition full_barrage_estimated_csi --steps 3000 --seed 23 --output_dir $root/full_barrage_estimated_csi --resume $root/full_barrage_estimated_csi/diagnostic_last.pt --allow_long_run"
else
  echo "C1 3000-step extension not recommended without optimization_still_progressing status."
fi

```
