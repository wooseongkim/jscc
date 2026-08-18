# R4 Stratified Resource Allocation Design

## Goal

Evaluate whether conservative, fixed layer-group transmit-power profiles improve
the official waveform SI-SDR of the selected R4 repetition-3/MRC checkpoint,
without changing resource count, repetition, bandwidth, channel realization, or
the existing decoder.

## Scope and invariants

The experiment supports exactly three primary profiles: `uniform`,
`core_protection`, and `layer1_focused`.  It treats layers 0--3 as one core
group and gives layer 1 an additional level only in `layer1_focused`; it never
uses an eight-layer ranking or NMSE to choose weights.

Each input has 1,920 source complex symbols: eight contiguous groups of 240
symbols.  The existing global balanced triplet allocator remains responsible
for the bijective mapping to 5,760 data REs and for its delayed-CSI triplet
multiplier.  Every profile uses the same allocation, delayed CSI, fading, LS
estimate, AWGN, seeds, checkpoint, decoder, and official unaligned waveform
metric.  Repetition remains three.

The only profile-dependent operation is a source-order multiplier.  Given raw
layer weights `r_l` and source counts `N_l`, it computes

`w_l = r_l / (sum_l N_l r_l / sum_l N_l)`.

The transmitted complex symbol is scaled by `sqrt(w_l)`.  This makes
`sum_l N_l w_l / sum_l N_l = 1`.  The multiplier is composed with, rather than
replaces, the existing CSI triplet power multiplier; the same composed power is
passed to coherent MRC so the receiver reverses transmit scaling consistently.

## Components

`speech_jscc.evaluation.r4_stratified_allocation` will provide small,
independently testable helpers for profile definitions, source-symbol layer
indices, count-weighted normalization, source-order power expansion, power
measurements, deterministic utterance-level bootstrap summaries, paired tests,
and artifact schema validation.

`evaluate_r4_stratified_allocation.py` will reuse the existing R4 physical
forward path.  It will prepare the channel realization once per
utterance/realization and run the three profiles against that immutable state.
It will stop a full comparison when uniform fails the corrected legacy
reference check.  It will write the requested JSON, JSONL, CSV, YAML, and
Markdown artifacts under the supplied split-specific output root.

`configs/eval_r4_stratified_allocation.yaml` will declare the physical R4
configuration, selected checkpoint, metric policy, data roles, profiles, raw
weights, reference tolerance, and output root.  The profile interface keeps
`mapping_policy`, `repetition_policy`, and `power_policy` separate, but rejects
anything other than balanced-triplet/fixed-three in this experiment.

## Validation

New tests cover uniform equivalence at the power-vector boundary, eight-layer
shape validation, 240-symbol contiguous layer mapping, count-weighted power
normalization, square-root amplitude scaling, actual power equality, profile
power direction, deterministic pairing/bootstrap, static repetition/RE/source
counts, finite values, and profile artifact names.  The evaluator will support
dry run and a two-utterance/one-realization smoke run.  Full 64x2 and 48x2 runs
are intentionally command-only deliverables.

## Interpretation

The report labels this an exploratory fixed heuristic.  Oracle ablation gain
is not allocation gain; wireless ordering except Layer 1 is split-sensitive;
the result is not learned, optimal, production-ready, jammer-aware, or
SI-SDR-loss-fine-tuned.  A positive result is only a reason to reassess the
same profiles after separate SI-SDR-aware fine-tuning and reduced ablation.
