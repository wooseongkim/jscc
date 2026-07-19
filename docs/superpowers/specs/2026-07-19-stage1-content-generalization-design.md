# Stage-1 Content-Generalization Ladder Design

## Objective

Identify the first path at which Stage-1 fails to generalize from multiple training utterances to unseen utterances and unseen speakers. Preserve the existing O6 failure classification and treat J1 as exploratory evidence produced from a failed parent, never as a curriculum pass or J2 parent.

## Current Evidence Classification

- O6 remains `FAIL` because unseen-utterance V2 does not improve by 5% over the zero predictor.
- J1 is `exploratory_failed_parent` because its parent O6 checkpoint failed its gate.
- J1 is excluded from curriculum readiness and cannot be used as J2's parent.
- V1 is only a weak pass; it demonstrates partial learning on seen content under unseen channels, not that random-channel learning is solved.
- O6 and J1–J5 remain blocked until G3 passes at a sufficient training-subset size.

## Architecture

Extend the versioned random-distribution diagnostic infrastructure with a focused content-generalization module. The shared infrastructure continues to own provenance, seed derivation, manifests, cache hashes, fixed validation suites, checkpoints, resume validation, metrics, and gates. The content module owns speaker-aware subset construction and the four G0–G3 data paths.

The implementation exposes one CLI with `--stage {g0_direct,g1_pilot_reserved_identity,g2_fixed_clean,g3_random_clean}` and `--subset-size {16,64,256,full}`. All long runs require explicit external acknowledgement and are executed only through guarded shell scripts.

## Speaker-Aware Data Split

Speaker IDs are extracted deterministically from LibriSpeech paths of the form `<speaker>/<chapter>/<speaker>-<chapter>-<utterance>.flac`. If extraction fails, the speaker ID is recorded as `unknown`; unknown samples are not silently assigned to a speaker-specific group.

For each subset size:

1. Select optimization utterances deterministically from the train manifest while preserving multiple speakers where possible.
2. `seen_utterance_unseen_channel` uses optimization utterances with held-out channel/noise seeds.
3. `same_speaker_unseen_utterance_unseen_channel` uses train-manifest utterances from the selected train speakers that were excluded from optimization.
4. `unseen_speaker_unseen_utterance_unseen_channel` uses valid-manifest utterances. The current mini-LibriSpeech train and valid speaker sets have no overlap.

The validation IDs and seeds are fixed across G0–G3 and across subset sizes where the group definition permits. No test manifest or test cache is accepted. If a requested group lacks candidates, the group is recorded as unavailable with the exact reason and `unknown` speaker count.

## Training-Subset Sizes

Supported sizes are `16`, `64`, `256`, and `full`. Selection is nested: the 16 utterances are a prefix/subset of 64, which is a subset of 256, which is a subset of full train. This isolates the effect of content diversity.

Within a G stage, execute subset sizes in ascending order. Failure at 16 does not stop 64/256/full because measuring the content-size threshold is the purpose of the ladder. Passing a subset records the smallest passing size and advances to the next G stage. Failure at `full` identifies the current G stage as the first failing path and prevents later G stages.

Each G stage starts from the same deterministic fresh initialization. Results from different subset sizes do not resume each other and are not curriculum results.

## G0–G3 Paths

### G0: Multi-Utterance Direct Bypass

`latent -> JSCC encoder -> JSCC decoder`

No resource allocation, pilot mapping, channel, noise, CSI estimation, or equalization is used. Fixed neutral transmitter and receiver states and all-one gates are used. This isolates content compression and encoder/decoder capacity.

### G1: Pilot-Reserved Identity Path

`latent -> encoder -> uniform allocation -> pilot-reserved pack -> pilot removal -> deallocation -> decoder`

No fading, AWGN, jammer, CSI estimation, or equalization is used. Every encoder symbol must survive the identity round trip. This isolates allocation and mapping without physical-channel effects.

### G2: Fixed Clean Multipath/AWGN

G1 plus one fixed six-tap clean multipath channel, a fixed AWGN realization, fixed pilot mask, `dft_tap_ls`, estimated CSI, equalization, and `observable_v1`. The same realization is used for every utterance and optimization step. SNR is 10 dB. This tests multi-content learning under one repeatable channel.

### G3: Random Clean Multipath/AWGN

G1 plus a new six-tap multipath channel and AWGN realization each step, with SNR sampled uniformly from `[5,15]` dB. It uses `dft_tap_ls`, estimated CSI, and `observable_v1`. Channel and noise hashes must vary. This is the content-controlled successor to O6.

## Validation and Metrics

Every validation group records aggregate and per-layer:

- normalized MSE and zero-predictor improvement;
- target and reconstruction power;
- reconstruction/target power ratio;
- cosine similarity;
- Pearson correlation;
- finite status.

Layer 0 is copied into an explicit `layer0_summary` and never hidden by aggregate metrics.

Dataset diagnostics compare train and validation groups using:

- latent power, mean, and standard deviation;
- utterance duration distribution;
- utterance and speaker counts;
- unknown-speaker count;
- manifest hash;
- latent-cache hash;
- preprocessing configuration and hash.

The fixed validation-suite artifact contains utterance IDs, speaker IDs, group labels, channel/noise seeds, and a stable suite hash.

## Gates

A validation group passes when all of the following hold:

- relative improvement over zero predictor is at least 5%;
- power ratio is at least `0.01`;
- cosine similarity is positive;
- Pearson correlation is positive;
- all metrics and parameters are finite.

The stage/subset gate requires all available validation groups to pass. Layer 0 is reported separately but does not override an aggregate failure or conceal failures in other layers.

G3 must pass at a sufficient subset size before O6 can be reconsidered. The report records the smallest passing subset and results for all subset sizes that were actually executed. No unexecuted result is marked complete.

## Checkpoints and Provenance

Every checkpoint and result records:

- diagnostic engine and stage-definition versions;
- G stage and subset size;
- exact path mode;
- deterministic seed-derivation version;
- train and validation IDs with speakers;
- manifest, cache, preprocessing, and validation-suite hashes;
- model initialization hash;
- fixed channel/noise hashes for G2;
- channel/noise diversity for G3;
- local and cumulative optimizer steps;
- git commit and dirty-worktree state.

Resume is allowed only within the exact same G stage/subset trajectory. Cross-subset or cross-stage resume is rejected.

## External Execution

Create a guarded external script that runs G0 first and processes subset sizes `16 -> 64 -> 256 -> full`. It proceeds to G1 only after G0 passes at a measured subset size, and likewise for G2 and G3. It stops at the first stage that fails even with `full`.

The script uses `set -euo pipefail`, refuses existing outputs without `--overwrite` or valid `--resume`, captures stdout/stderr with `tee`, preserves failed-run logs outside not-yet-created result directories, and supports `--dry-run`. Codex does not execute long G runs.

## Reporting and Readiness

Write a consolidated report under `runs/stage1_content_generalization/` containing:

- O6 `FAIL` and J1 `exploratory_failed_parent` status;
- results by G stage and subset size;
- seen/same-speaker/unseen-speaker comparisons;
- per-layer and explicit layer-0 results;
- first failing G stage;
- smallest passing subset where applicable;
- exact next external command.

Uniform Stage-1 readiness remains false until G3 passes at a sufficient subset size and O6 is rerun or explicitly re-evaluated under the accepted expanded content protocol. J1–J5 cannot resume from the current exploratory J1 artifact.

## Testing

Focused tests cover deterministic nested subsets, speaker parsing and group separation, test-data rejection, fixed validation hashes, all four G paths, G2 fixed and G3 changing realizations, aggregate/per-layer/layer-0 metrics, dataset statistics, gate progression, checkpoint provenance/resume rejection, failed-parent J1 classification, J2 parent rejection, external script safety, and readiness blocking.

Codex may run unit tests, report regeneration, dry runs, deterministic step-0 checks, and at most a three-to-five-step smoke run. It must not run long G0–G3, O6, J1–J5, Uniform, or Weighted optimization.
