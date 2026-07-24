#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$repo_root"
architecture=""; subset=""; batch=4; epochs=64; device=auto; workers=0; resume=""; overwrite=0; continue_on_pass=0; dry=0
root="runs/stage1_content_generalization/g0_architecture_screening_v1"
while (($#)); do case "$1" in
 --architecture) architecture="$2"; shift 2;; --subset-size) subset="$2"; shift 2;; --batch-size) batch="$2"; shift 2;;
 --max-epochs) epochs="$2"; shift 2;; --device) device="$2"; shift 2;; --num-workers) workers="$2"; shift 2;;
 --resume) resume="$2"; shift 2;; --overwrite) overwrite=1; shift;; --continue-on-pass) continue_on_pass=1; shift;;
 --dry-run) dry=1; shift;; --output-root) root="$2"; shift 2;; *) echo "unknown argument: $1" >&2; exit 2;; esac; done
[[ "$architecture" =~ ^(flat_mlp|normalized_flat_mlp|conv_conformer_v1|per_layer_pca_480)$ ]] || { echo "--architecture is required" >&2; exit 2; }
[[ "$subset" =~ ^(16|64|256|full)$ ]] || { echo "--subset-size is required" >&2; exit 2; }
config="configs/train_stage1_fixed_tx_uniform.yaml"; [[ "$architecture" == conv_conformer_v1 ]] && config="configs/g0_conv_conformer_v1.yaml"
out="$root/$architecture/subset_$subset"; cmd=(python diagnose_g0_architecture_screening.py --config "$config" --architecture "$architecture" --subset-size "$subset" --batch-size "$batch" --max-epochs "$epochs" --device "$device" --num-workers "$workers" --output-dir "$out" --allow-long-run)
[[ -n "$resume" ]] && cmd+=(--resume "$resume"); ((overwrite)) && cmd+=(--overwrite); ((continue_on_pass)) && cmd+=(--continue-on-pass)
if ((dry)); then printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
[[ ! -e "$out" || $overwrite -eq 1 || -n "$resume" ]] || { echo "refusing existing output directory: $out" >&2; exit 1; }
mkdir -p "$(dirname "$out")"; log_tmp="${out}.run.log"; command_tmp="${out}.command.txt"
printf '%q ' "${cmd[@]}" > "$command_tmp"; printf '\n' >> "$command_tmp"
"${cmd[@]}" 2>&1 | tee "$log_tmp"
mv "$log_tmp" "$out/run.log"; mv "$command_tmp" "$out/command.txt"
