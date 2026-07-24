#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dry=0;continue=0;device=auto;subset=256;batch=4;workers=0;epochs=64;overwrite=0;resume=""
while (($#));do case "$1" in --dry-run)dry=1;shift;;--continue-on-pass)continue=1;shift;;--device)device="$2";shift 2;;--subset-size)subset="$2";shift 2;;--batch-size)batch="$2";shift 2;;--num-workers)workers="$2";shift 2;;--max-epochs)epochs="$2";shift 2;;--overwrite)overwrite=1;shift;;--resume)resume="$2";shift 2;;*)echo "unknown argument: $1" >&2;exit 2;;esac;done
common=(--device "$device" --subset-size "$subset" --batch-size "$batch" --num-workers "$workers" --max-epochs "$epochs");((overwrite))&&common+=(--overwrite)
if ((dry));then bash scripts/run_g1_mapping_equivalence.sh --dry-run;bash scripts/run_g1_conv_conformer_external.sh "${common[@]}" --dry-run;bash scripts/run_g2_conv_conformer_external.sh "${common[@]}" --dry-run;bash scripts/run_g3_conv_conformer_external.sh "${common[@]}" --dry-run;exit 0;fi
bash scripts/run_g1_mapping_equivalence.sh;bash scripts/run_g1_conv_conformer_external.sh "${common[@]}";((continue))||exit 0
bash scripts/run_g2_conv_conformer_external.sh "${common[@]}";bash scripts/run_g3_conv_conformer_external.sh "${common[@]}"
