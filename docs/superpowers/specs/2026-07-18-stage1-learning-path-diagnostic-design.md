# Stage-1 Learning-Path Diagnostic Design

## Objective

Build and run a focused, evidence-producing diagnostic for the existing fixed-transmitter Stage-1 JSCC training path. The diagnostic must distinguish meaningful but incomplete learning, zero-output collapse, broken gradients or updates, insufficient one-example capacity, resource/channel integration defects, excessive random-distribution difficulty, and misleading loss or logging.

The work uses the current local Stage-1 and channel-estimator implementation as authoritative. It preserves the user's uncommitted files and existing run artifacts. It does not start 20,000-step training, weighted training, Stage 2, or architecture redesign.

## Confirmed Local Context

- The active configuration is `configs/train_stage1_fixed_tx_uniform.yaml`.
- The model target shape is `[B,8,50,1024]`; the channel grid is `[B,64,32]` complex values.
- The eight encoder branches receive 256 complex channel uses each.
- Stage-1 uses all-one gates, uniform layer power, uniform allocation, multipath block fading, DFT tap LS CSI estimation, estimated equalization, and observable receiver state.
- The frozen SpeechTokenizer provides latent targets and waveform decoding but is not optimized.
- The existing best and last checkpoints and the 1,000-row JSONL log are present under `runs/stage1_uniform_1000/`.
- Existing validation uses seven deterministic scenarios and deterministic representation batches constructed once before training.
- Pilot insertion replaces selected allocated data symbols. Pilot removal currently zeros those positions while retaining grid shape; O2 versus O3 will determine whether this expected overhead is the first failing path stage.

## Exact Loss Contract

For reconstruction `R` and target `E` with shape `[B,L,T,D]`, current Stage-1 computes for each layer `l`:

```text
raw_mse_l = mean((R[:,l] - E[:,l])^2, dimensions=(batch,time,latent))
target_power_l = detach(mean(E[:,l]^2, dimensions=(batch,time,latent)))
normalized_mse_l = raw_mse_l / max(target_power_l, 1e-6)
total_loss = sum_l(weight_l * normalized_mse_l) / sum_l(weight_l)
```

All eight weights are `1.0`. No waveform, power, gate, refiner, jammer-estimation, or other auxiliary loss is included. Tests will establish that zero reconstruction gives normalized loss 1, perfect reconstruction gives 0, and a scaled target gives the analytic squared scale error.

## Architecture

### Shared diagnostic library

Create a focused module under `src/speech_jscc/diagnostics/` that owns reusable scientific calculations and trace structures:

- latent loss decomposition and predictor metrics;
- cosine, Pearson correlation, power ratio, bias, near-zero fraction, and finite checks;
- checkpoint parameter, optimizer, forbidden-key, and capacity audits;
- gradient/update auditing by encoder branch and decoder;
- intermediate-tensor gradient retention and reporting;
- fixed validation reconstruction collection;
- waveform comparison helpers using metrics already supported by the repository;
- overfit experiment definitions and common result schemas;
- deterministic serialization helpers for JSON and CSV.

The library will call the existing model, channel, allocator, pilot, estimator, and codec implementations. It will not duplicate the scientific path where the production functions can be invoked or instrumented safely.

### Checkpoint diagnostic CLI

`diagnose_stage1_learning.py` will:

1. resolve and validate the supplied Stage-1 config;
2. reproduce the same fixed validation batches and scenarios as training;
3. load the requested checkpoint and also inspect sibling best/last checkpoints when present;
4. evaluate trained, zero, per-batch-mean, optional cached global-mean, and fresh random-model predictors;
5. save per-sample, per-scenario, per-layer, and aggregate latent metrics;
6. run checkpoint, capacity, receiver-state, log, gradient, and dataflow audits;
7. decode and compare a bounded number of waveform examples;
8. write JSON, CSV, Markdown, plots, and audio under the supplied output directory.

The CLI will use deterministic seeds, validate checkpoint/config compatibility, and fail explicitly on shape mismatches or nonfinite required results.

### Overfit ladder CLI

`overfit_stage1_path.py` will run independently initialized experiments O0 through O7. Each experiment has an explicit data-path adapter and stochasticity policy:

- O0: decoder only, fixed synthetic complex input and target;
- O1: encoder directly to decoder, fixed states;
- O2: encoder, identity uniform allocation/deallocation, decoder;
- O3: fixed clean multipath, fixed pilots, DFT LS, estimated equalization;
- O4: O3 plus fixed 10 dB AWGN;
- O5: O4 plus fixed jammer channel, waveform, mask, and realization;
- O6: small fixed latent set with changing clean multipath/AWGN draws;
- O7: current full random Stage-1 distribution.

Each experiment starts from a fresh seeded model. O0 runs at most 500 steps; O1–O7 run at most 1,000 steps. The CLI records initial, best, final, and zero losses; relative improvement; step to best; power ratio; cosine; correlation; and encoder/decoder gradient norms. It stops after the first scientifically meaningful failure unless an explicit diagnostic option requests later stages. O7 is never entered before O0–O6 pass.

## Dataflow and Gradient Instrumentation

The diagnostic will trace:

```text
latent -> layer encoders -> concatenated symbols -> allocation
-> pilot replacement -> multipath -> CSI estimate -> equalization
-> pilot removal -> deallocation -> decoder -> reconstruction -> loss
```

Hooks or retained gradients will be applied without detaching the training path. Required intermediates are encoder symbols, allocated symbols, received grid, equalized grid, decoder input, and reconstruction. Receiver-state features remain intentionally detached, while loss gradients must still reach every encoder branch through transmitted symbols and differentiable channel arithmetic.

The update audit snapshots parameters immediately before and after one deterministic Adam step. It reports norms and nonfinite/zero fractions per encoder branch, decoder, and total model. It verifies that both model sides update and the codec neither receives gradients nor changes.

## Baselines and Metrics

For every validation scenario, sample, and layer, the trained reconstruction is compared with:

- all zeros;
- the target batch mean broadcast to each sample;
- an optional cached global training-latent mean when it can be computed without changing the scientific dataset;
- a fresh untrained model evaluated on the identical batch.

Each row contains target and reconstruction power, power ratio, raw and normalized MSE, zero normalized MSE, trained-minus-zero, cosine, Pearson correlation, means, standard deviations, bias, near-zero fraction, and finite status. Aggregate tables preserve layer and scenario boundaries so aggregate averages cannot hide collapsed layers.

Meaningful reconstruction requires joint evidence: loss below the zero-predictor loss, non-negligible output power, and positive alignment/correlation. A loss near 1 with near-zero output power and alignment is classified as collapse.

## Waveform Evaluation

For a small deterministic subset, the frozen SpeechTokenizer decoder will decode target, trained, zero, and mean latents. Existing waveform metric utilities will compute waveform SNR, SI-SDR, STFT L1, STOI when installed, and PESQ only when already supported and valid. Audio files will be written as source, clean codec, Stage-1 reconstruction, zero-latent decode, and mean-latent decode.

Waveform findings remain secondary to latent evidence. The report will not claim success unless trained decoding improves over zero-latent decoding.

## Capacity and Receiver-State Audits

The capacity audit reports latent real dimensions, complex and equivalent-real channel dimensions, nominal and post-pilot compression ratios, hidden sizes, branch allocations, pilot overhead, and trainable parameter counts split by encoder and decoder.

The receiver-state audit reports mean, standard deviation, extrema, clamp fractions, and finite fraction per observable-v1 feature for clean 5 dB, clean 15 dB, barrage, narrowband, burst, and pilot scenarios. Nearly constant or clamp-dominated features are marked uninformative but are not treated alone as a root cause.

## Existing Log Audit

The JSONL parser will summarize training and validation loss, per-layer raw MSE, receiver-state means and standard deviations, transmit power, clean/jammed splits, and jammer-type splits. It will explicitly list requested fields absent from the historical log, including historical reconstruction power, latent correlation, and gradient norms if absent. No missing values will be inferred.

## Tests and TDD Boundaries

Create the four requested test files:

- `tests/test_stage1_zero_baseline.py`
- `tests/test_stage1_gradient_path.py`
- `tests/test_stage1_overfit_diagnostics.py`
- `tests/test_stage1_dataflow_integrity.py`

Tests use reduced mock shapes and fixed seeds. Each new behavior is introduced through a failing test, then the smallest implementation is added. Long real-codec training is excluded from unit tests. Required integration tests verify both checkpoint files load, all diagnostic metrics remain finite, fixed/random channel policies behave as claimed, and existing Stage-1/channel tests remain green.

If investigation confirms a production bug, a minimal failing regression test will precede the smallest targeted fix. The affected overfit stage and all downstream required tests will then be rerun. No unrelated refactor or model enlargement is permitted.

## Artifacts and Reporting

Write under `runs/stage1_uniform_1000/diagnostics/`:

```text
checkpoint_diagnostic.json
checkpoint_diagnostic.md
validation_baselines.csv
validation_per_layer.csv
gradient_audit.json
receiver_state_audit.csv
overfit_results.csv
overfit_report.md
plots/
audio_examples/
```

The Markdown report includes a concise decision tree, the first failing overfit stage, verified implementation bugs before and after any fix, exact commands, test results, and one final classification with a scoped next-task recommendation.

## Error Handling and Resource Limits

- Existing run files are read-only inputs; diagnostic outputs are isolated beneath `diagnostics/`.
- Checkpoint, config, tensor-shape, and model-state incompatibilities fail with actionable messages.
- Unsupported optional waveform metrics are recorded as unavailable rather than causing false failure.
- O0–O7 use bounded steps and deterministic checkpoints; no 20,000-step or weighted run is invoked.
- CPU/GPU selection follows the resolved config. The report records the actual device and runtime.
- A stage is declared failed only from its recorded metrics and thresholds, not solely because its loss decreased.

## Acceptance Decision Logic

The report applies the requested first-failure interpretation:

- O0 fails: decoder/loss/optimizer defect or inadequate decoder memorization capacity.
- O0 passes and O1 fails: encoder/decoder interface, bottleneck, or gradient-path defect.
- O1 passes and O2 fails: allocation/deallocation defect.
- O2 passes and O3 fails: pilot/CSI/equalization/channel integration defect.
- O3–O5 pass and O6 fails: random clean-channel distribution or schedule difficulty.
- O6 passes and O7 fails: jammer distribution difficulty.
- All pass while the checkpoint remains collapsed: Stage-1 optimization schedule/distribution exposure is insufficient rather than a broken deterministic path.

Compression severity is supporting evidence only. Any recommended curriculum, capacity increase, or architecture change is deferred to a separate task.
