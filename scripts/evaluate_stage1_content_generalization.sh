#!/usr/bin/env bash
set -euo pipefail
ROOT="runs/stage1_content_generalization"; DRY=0
while (($#)); do case "$1" in --root) ROOT=$2; shift 2;; --dry-run) DRY=1; shift;; --device|--resume) shift 2;; --overwrite) shift;; *) echo "unknown option: $1" >&2; exit 2;; esac; done
CMD=(python scripts/summarize_content_generalization.py --root "$ROOT")
if [[ $DRY -eq 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
"${CMD[@]}"
