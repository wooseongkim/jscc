# O5 Fixed-Jammer Root-Cause Diagnostic Design

## Scope

Build an offline-only diagnostic for the first failing O5 fixed-jammer stage. Preserve `pilot_reserved_v1`, model/channel/estimator/loss/state/checkpoint behavior, and never expose oracle information to neural inputs. Codex runs tests, dry-runs, and at most one 3–5-step smoke; all 500/1000/3000-step work is emitted as external scripts.

## Components

- `src/speech_jscc/diagnostics/o5_root_cause.py`: condition definitions, fixed realization construction, stable hashes, offline forward variants, metrics, scale diagnostics, slopes, checkpoints, resume, and reports.
- `diagnose_o5_root_cause.py`: guarded CLI requiring `--steps <= 5` unless `--allow_long_run`, with non-overwrite output handling and dry-run support.
- `scripts/run_o5_root_cause_external.sh`: safe C0–C6 500-step matrix with separate directories and logs.
- `scripts/run_o5_extension_external.sh`: reads summaries and prints extension commands; executes only with an explicit flag.
- Focused tests in `tests/test_o5_root_cause_diagnostics.py` and `tests/test_o5_external_execution.py`.

## Conditions

- C0 `clean_awgn_reference`: no jammer, estimated DFT-tap LS CSI.
- C1 `full_barrage_estimated_csi`: full-grid barrage, estimated CSI.
- C2 `full_barrage_oracle_csi`: C1 realization, true legitimate CSI only in diagnostic equalizer.
- C3 `data_only_barrage_estimated_csi`: non-pilot-only jammer normalized to total-grid 0 dB JSR, estimated CSI.
- C4 `data_only_barrage_oracle_csi`: C3 realization, diagnostic oracle legitimate CSI.
- C5 `pilot_only_jammer_estimated_csi`: pilot-only jammer normalized to total-grid 0 dB JSR.
- C6 `full_barrage_oracle_subtraction`: C1 received tensor minus exact faded jammer, then normal estimated legitimate-channel path; labeled `diagnostic_only_oracle_jammer_subtraction`.

All conditions share target, initial parameters, legitimate channel, noise, and pilot mask. C1/C2/C6 share jammer channel/waveform/mask; C3/C4 share their jammer realization. Fixed tensors never change during optimization.

## Scientific Metrics

Step 0 and configured intervals record total and per-layer raw/normalized MSE, target/reconstruction power and ratio, cosine, correlation, reconstruction moments, encoder/decoder gradient norms, LR, best/zero loss, and improvement. Channel diagnostics distinguish requested total-grid JSR, measured total-grid JSR, active-resource JSR, overlap counts, pre/post-channel JSR, CSI NMSE, pilot/data EVM, equalized data MSE, estimated/oracle data SINR, equalizer gain, and channel-power tails.

For reconstruction `R` and target `E`, diagnostic-only scaling uses `a*=sum(R*E)/max(sum(R^2),epsilon)` globally and per layer. It reports original/rescaled NMSE, power ratios, and correlations but never changes training loss, inference, or checkpoints.

Plateau analysis fits least-squares slopes over the final 20%. Configurable tolerances classify `optimization_still_progressing`, `plateaued_failure`, and `amplitude_suppression` without assigning a single cause from one metric.

## Checkpoints and Safety

Diagnostic checkpoints are labeled `diagnostic_type: o5_root_cause_fixed_realization` and contain model/optimizer/step, fixed realization tensors or complete specification, RNG states, condition/config/resource metadata, hashes, and history summary. Resume validates hashes and restores exact optimizer/realization state. They are not scientific Stage-1 checkpoints.

The CLI refuses existing output directories unless `--overwrite` or valid `--resume` is supplied. Long runs require `--allow_long_run`. Shell scripts use `set -euo pipefail`, preserve command/environment/git/log files, and never overwrite by default.

## Reporting

The root output contains manifest, paired hashes, aggregate CSV/JSON, Markdown decision report, external commands, plots, and a directory per condition with resolved config, command, environment, JSONL metrics, summary, per-layer CSV, hashes, checkpoint, and log. Dry-run validates and prints without initializing optimization.
