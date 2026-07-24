#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

overwrite=0
with_extensions=0
for argument in "$@"; do
  case "$argument" in
    --overwrite) overwrite=1 ;;
    --with-extensions) with_extensions=1 ;;
    *) echo "Usage: $0 [--overwrite] [--with-extensions]" >&2; exit 2 ;;
  esac
done

primary_args=()
[[ "$overwrite" -eq 1 ]] && primary_args+=(--overwrite)

echo "[1/2] Running the sequential C0-C6 500-step matrix."
# Child long-run commands carry --allow_long_run explicitly.
bash scripts/run_o5_root_cause_external.sh "${primary_args[@]}"

if [[ "$with_extensions" -eq 1 ]]; then
  echo "[2/2] Running only extensions recommended by the completed summaries."
  bash scripts/run_o5_extension_external.sh --execute
else
  echo "[2/2] Extensions were not requested. Review recommendations with:"
  echo "  bash scripts/run_o5_extension_external.sh"
  echo "Re-run this batch with --with-extensions to execute recommended resumes."
fi
