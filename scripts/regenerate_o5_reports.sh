#!/usr/bin/env bash
set -euo pipefail

ROOT="runs/stage1_uniform_1000/diagnostics/o5_root_cause_v1"
if [[ ${1:-} == "--root" ]]; then
  ROOT=${2:?missing root path}
fi
python scripts/summarize_o5_root_cause.py --root "$ROOT" 2>&1 | tee "$ROOT/report_regeneration.log"
