#!/usr/bin/env bash
set -euo pipefail
DRY=0; ROOT="runs/stage1_random_distribution"
while (($#)); do case "$1" in --dry-run) DRY=1; shift;; --root) ROOT=$2; shift 2;; --device|--resume) shift 2;; --overwrite) shift;; *) echo "unknown option: $1" >&2; exit 2;; esac; done
CMD=(python evaluate_stage1_readiness.py --root "$ROOT" --fixed-path-passed)
if [[ $DRY -eq 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
"${CMD[@]}"
