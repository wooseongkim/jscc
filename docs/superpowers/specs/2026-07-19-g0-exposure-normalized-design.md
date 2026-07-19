# G0 Exposure-Normalized Diagnostic Design

## Objective

Determine whether the existing G0 direct encoder/decoder path fails because of insufficient per-utterance exposure or because the current optimization/capacity cannot learn a multi-utterance latent distribution. Preserve the production JSCC architecture and all existing G0 artifacts.

## Fixed G0 Path

The diagnostic uses only:

`frozen SpeechTokenizer latent -> current JSCC encoder -> direct complex-symbol bypass -> current JSCC decoder -> reconstructed latent`

It excludes allocation, pilot mapping, channel effects, noise, CSI, equalization, jamming, learned gates, and latent refiners. SpeechTokenizer parameters are frozen and must receive no gradients.

## Subsets and Fixed Validation

Train subsets are the existing nested deterministic `16`, `64`, `256`, and `full` sets. The full eligible train subset contains 1491 utterances after fixed same-speaker holdouts. Every subset uses the same validation suite:

- `seen_utterance`;
- `same_speaker_unseen_utterance`;
- `unseen_speaker_unseen_utterance`.

Validation IDs, speaker IDs, preprocessing, latent cache, seeds, and suite hash are identical across subset sizes and epoch budgets. Test manifests and test caches are rejected.

## Exposure-Normalized Sampler

Supported checkpoint epochs are `[1,2,4,8,16,32,64]`. For subset size `N` and batch size `B`, cumulative optimizer steps at epoch `E` are `ceil(E*N/B)` when every epoch is represented by `ceil(N/B)` batches. An epoch is one deterministic shuffle of all `N` utterances followed by sampling without replacement; its final batch may contain fewer than `B` utterances.

The sampler derives an independent permutation from the root seed, subset identity, and epoch number. It records per-utterance presentation counts, cumulative sample presentations, completed epochs, batch sizes, and optimizer steps. At every completed epoch each utterance must have exactly the same presentation count.

Every subset starts from fresh initialization with the same initialization seed and identical model-parameter hash. Optimizer, learning rate, loss, clipping, preprocessing, and validation protocol are identical. Subsets do not resume from each other.

## Constant-Predictor Baselines

The train subset is traversed once to compute:

- zero predictor;
- global elementwise train mean latent with shape `[8,50,1024]`;
- per-layer scalar train mean expanded over time and latent dimensions;
- speaker-conditional elementwise train mean.

A speaker-conditional predictor is available only when that speaker contributes at least two optimization utterances. It is unavailable for unseen speakers and speakers below the threshold. Unavailable results are explicit and are never replaced by another baseline silently.

Each validation group reports model loss and baseline losses under the same uniform per-layer target-power-normalized Stage-1 equation. It reports absolute and relative improvement over zero, global mean, and layerwise mean. Speaker-conditional loss is also reported where available.

## Metrics

At epoch checkpoints, record aggregate, per-layer, explicit layer 0, and layers 1–7 aggregate metrics:

- normalized and raw MSE;
- target and reconstruction power;
- power ratio;
- cosine similarity;
- Pearson correlation;
- reconstruction mean and standard deviation;
- optimal-scalar-rescaled normalized loss;
- finite status;
- encoder gradient norm for every layer branch;
- decoder gradient norm;
- train and validation sample counts;
- optimizer steps and actual presentation counts.

The optimal scalar is diagnostic only and is never applied to training or checkpoint weights.

## Epoch Evaluation and Checkpoints

Evaluate at epochs `1,2,4,8,16,32,64` up to `--max-epochs`. Save a checkpoint at every evaluation epoch plus:

- `diagnostic_best.pt`, selected using the mean of same-speaker and unseen-speaker normalized loss;
- `diagnostic_final.pt`, containing the final model and optimizer state.

Checkpoints include sampler state, completed epoch, next epoch, per-utterance counts, optimizer step, model/optimizer, RNG state, baselines or their stable hashes, validation-suite hash, subset/protocol provenance, git state, and metric history. Resume requires exact equality of subset, batch size, seed, initialization hash, manifest/cache/preprocessing hashes, validation hash, optimizer/loss configuration, and stage version.

## Early Stopping

No finite run stops before epoch 16 because of a failed 5% gate. Nonfinite values terminate immediately.

If `--max-epochs` permits 32 or 64:

- continue while train loss or unseen-speaker validation loss has a meaningfully negative recent slope;
- classify plateau only when both recent loss slope and correlation slope are within configurable tolerances;
- otherwise continue to the next configured checkpoint.

Test data is never used for stopping or model selection.

## Passing Gate

G0 passes only on subset 256 or full after sufficient exposure when both same-speaker and unseen-speaker groups satisfy:

- zero-baseline improvement at least 5%;
- positive correlation meaningfully above zero;
- power ratio at least `0.01`;
- finite values and parameters.

G1–G3, O6, and J1–J5 remain blocked during this task.

## Result Classification

The final report selects one or more evidence-backed classifications:

- `insufficient optimization budget` when late loss remains clearly improving;
- `small-subset memorization` when train/seen improve but unseen utterances do not;
- `utterance generalization failure` when same-speaker unseen fails;
- `speaker generalization failure` when same-speaker passes and unseen-speaker fails;
- `layer-1-to-7 collapse` when layer 0 retains alignment while layers 1–7 collapse;
- `current encoder-decoder optimization/capacity limitation` only when train loss also fails after sufficient exposure and slopes plateau;
- `mixed cause` when multiple supported conditions coexist.

Architecture capacity is not declared solely from validation failure.

## External Execution

Create `scripts/run_g0_exposure_normalized_external.sh`. It runs subset sizes sequentially, uses `set -euo pipefail`, accepts `--batch-size`, `--max-epochs`, `--device`, `--resume`, `--overwrite`, and `--dry-run`, refuses existing outputs by default, writes sibling pending logs until the result directory exists, and stops on command failure.

Default output is `runs/stage1_content_generalization/g0_exposure_normalized_v1/`. Each subset has a separate directory, resolved config, command, environment, metrics JSONL, checkpoints, summary, baseline statistics, presentation counts, and run log.

Aggregate reporting writes `exposure_normalized_report.md`, `aggregate_by_epoch.csv`, `aggregate_by_subset.csv`, `per_layer_by_epoch.csv`, and `exposure_manifest.json`.

## Codex Execution Restriction

Codex may run tests, dry-runs, report generation, and one subset-16 smoke run of at most two epochs. It must not run long exposure-normalized experiments, G1–G3, O6, J stages, Uniform, or Weighted training.

## Follow-Up Boundary

If sufficiently exposed G0 still fails, the report prepares but does not implement a separate proposal comparing the current flatten MLP, corpus-level latent normalization, temporal structured encoder/decoder, and linear/PCA reconstruction references.
