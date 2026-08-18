# R4 Expanded Validation Design

## Goal

Re-evaluate the existing R4 waveform-fine-tuning checkpoints without training,
using a speaker-disjoint selection set and paired physical-channel conditions.
The legacy 64-utterance `dev-clean-2` final evaluation remains immutable and is
never used for checkpoint selection.

## Dataset roles

- Training remains the existing 256-utterance subset selected from
  `manifests/mini_librispeech/train.jsonl`.
- Selection validation is a deterministic, speaker-balanced 48-utterance subset
  of `manifests/mini_librispeech/test.jsonl`. Its metadata records
  `source_split: test-clean` and `assigned_evaluation_role:
  selection_validation`; its results must never be described as test results.
- Legacy final remains the exact ordered 64-path `dev-clean-2` suite produced by
  the original checkpoint configuration and `fixed_paths`, with the original
  evaluation seeds. It is emitted as an immutable reference manifest and is
  evaluated only behind explicit long-run flags after selection is complete.

The evaluator fails before model loading if any utterance or speaker overlaps
between train, selection validation, and legacy final.

## Candidate discovery

Named checkpoints (`best_5db_si_sdr.pt`, `best_clean_gate.pt`,
`best_validation_average.pt`, and `last.pt`) are always included. Additional
step checkpoints are discovered when present. Candidate generation adds
stage-boundary checkpoints, 1,000-step checkpoints, and the top ten historical
light-validation steps only when matching checkpoint files exist. Missing
historical weights are reported rather than reconstructed.

## Paired physical evaluation

The existing `R4WaveformForward` remains the only physical-path implementation.
For each SNR, realization, and ordered utterance trajectory, channel taps, noise
seeds, crop metadata, and bootstrap state are generated once. The receiver CSI
report is channel-observation-derived: pilots are deterministic and independent
of JSCC data, so one initial-checkpoint pass creates the report trajectory used
by the initial model and every candidate. Candidate-generated reports are
audited but never fed back into allocation. This prevents model-dependent CSI
state drift from weakening paired comparisons.

Every candidate and the initial R4 checkpoint therefore use identical waveform,
crop, taps, AWGN seed, pilot observations, delayed CSI input, and resource map.

## Metrics and statistics

Rows use explicit metric names identifying their baseline. Both
utterance-realization rows and realization-averaged utterance rows are retained.
For every checkpoint and SNR, deterministic paired statistics include count,
mean, median, standard deviation, standard error, percentiles, extrema,
improvement/regression rates, and a 2,000-sample paired bootstrap confidence
interval. Final ranking uses utterance-level deltas.

## Selection

Light validation only generates candidates. Full selection validation enforces
the 5 dB clean-codec waveform gate and 10/15 dB no-material-regression rules.
Passing candidates rank by 5 dB utterance-level paired SI-SDR mean, then its
10th percentile, three-SNR average, and variance. If none passes, candidates
rank by normalized minimum gate margin followed by the same 5 dB criteria.

The copied output is `best_expanded_validation.pt`. A nonpassing selection is
marked `selection_status: best_nonpassing_candidate` and
`clean_gate_pass: false`.

## Outputs and safety

The run writes immutable manifests, seed metadata, overlap audit, candidate
inventory, per-sample metrics, checkpoint summaries, paired statistics,
ranking, decision, runtime information, and a report. Existing output
directories are rejected unless `--overwrite` is explicit. Final evaluation
requires both `--run-final-test` and `--allow-long-run` and cannot feed back
into selection.

