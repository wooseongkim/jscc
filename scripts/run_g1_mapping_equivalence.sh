#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
device=auto; overwrite=0; dry=0
while (($#)); do case "$1" in --device) device="$2";shift 2;;--overwrite) overwrite=1;shift;;--dry-run) dry=1;shift;;*) echo "unknown argument: $1" >&2;exit 2;;esac;done
cmd=(python diagnose_g1_mapping_equivalence.py --device "$device"); ((overwrite))&&cmd+=(--overwrite)
if ((dry));then printf '%q ' "${cmd[@]}";echo;exit 0;fi
"${cmd[@]}"
