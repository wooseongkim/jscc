# Stage-1 Learning-Path Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run a bounded, evidence-producing Stage-1 diagnostic that identifies the first failing learning-path stage and distinguishes collapse, broken optimization, resource/channel defects, and distribution difficulty.

**Architecture:** A small `speech_jscc.diagnostics` package supplies pure metric/audit functions and experiment runners. Two repository-root CLIs compose those functions around the existing Stage-1 implementation: one diagnoses existing checkpoints and one runs the sequential O0–O7 ladder. Production channel/model code changes only after a reproducible failing regression test identifies a defect.

**Tech Stack:** Python 3.10+, PyTorch, pytest, CSV/JSON, matplotlib, soundfile, existing SpeechTokenizer and audio-metric wrappers.

## Global Constraints

- Treat the current local Stage-1 and channel-estimator files as authoritative and preserve unrelated uncommitted changes.
- Never start 20,000-step Stage-1 training or weighted Stage-1 training.
- Do not add learned gates, jammer estimation, latent refinement, dynamic allocation, Stage 2, model enlargement, or SpeechTokenizer fine-tuning.
- Default ladder execution is sequential, starts at O0, and stops at the first failure; only `--continue_after_failure` permits later stages.
- O0–O2 pass only with the specified 80%/0.2 loss condition plus power ratio `>=0.01`, cosine `>0`, and correlation `>0`.
- O3–O5 pass only with the specified 50%/0.5 loss condition plus power ratio `>=0.01`, cosine `>0`, and correlation `>0`.
- O6–O7 pass only with at least 5% improvement plus power ratio `>=0.01`, cosine `>0`, and correlation `>0`.
- Pilot overwrite is a suspected resource-mapping defect, not assumed overhead.
- Claim exact validation identity only if sample order/IDs, latent-cache hash, channel/noise/jammer seeds, and pilot-mask hash are all verified.
- Follow RED → verify failure → GREEN → verify pass for every new behavior.

---

### Task 1: Pure latent metrics and zero-baseline contract

**Files:**
- Create: `src/speech_jscc/diagnostics/__init__.py`
- Create: `src/speech_jscc/diagnostics/metrics.py`
- Create: `tests/test_stage1_zero_baseline.py`

**Interfaces:**
- Produces: `latent_metric_rows(reconstruction, target, epsilon, predictor, scenario, sample_ids=None) -> list[dict[str, object]]`
- Produces: `aggregate_latent_rows(rows, group_keys) -> list[dict[str, object]]`
- Produces: `normalized_layer_loss(reconstruction, target, epsilon) -> Tensor[L]`
- Produces: `zero_predictor_loss(target, weights, epsilon) -> tuple[Tensor, Tensor]`

- [ ] **Step 1: Write failing analytic-loss tests**

```python
def test_zero_reconstruction_has_unit_per_layer_nmse():
    target = torch.tensor([[[[1., 2.]], [[3., 4.]]]])
    loss, layers = zero_predictor_loss(target, torch.ones(2), 1e-6)
    torch.testing.assert_close(layers, torch.ones(2))
    torch.testing.assert_close(loss, torch.tensor(1.0))

def test_perfect_and_scaled_reconstruction_match_analytic_nmse():
    target = torch.randn(2, 3, 4, 5)
    torch.testing.assert_close(normalized_layer_loss(target, target, 1e-6), torch.zeros(3))
    torch.testing.assert_close(
        normalized_layer_loss(0.25 * target, target, 1e-6),
        torch.full((3,), 0.75**2),
    )
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_stage1_zero_baseline.py -q`
Expected: collection failure because `speech_jscc.diagnostics.metrics` does not exist.

- [ ] **Step 3: Implement metric calculations**

Implement per sample/layer flattening with target power, reconstruction power, power ratio, raw/normalized/zero NMSE, improvement, cosine, Pearson correlation, means, standard deviations, bias, near-zero fraction (`abs(x) < 1e-4`), and finite status. Use `epsilon` only for denominators; return correlation `0.0` for degenerate vectors and mark the degeneracy in the row.

- [ ] **Step 4: Add metric-field and finite tests, then run GREEN**

Run: `pytest tests/test_stage1_zero_baseline.py -q`
Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/speech_jscc/diagnostics tests/test_stage1_zero_baseline.py
git commit -m "test: define stage1 latent diagnostic metrics"
```

### Task 2: Resource, pilot, and dataflow integrity audit

**Files:**
- Create: `src/speech_jscc/diagnostics/dataflow.py`
- Create: `tests/test_stage1_dataflow_integrity.py`
- Modify only if a defect is proven: `src/models/resource_allocator.py`, `src/evaluation/paired.py`, or `src/channels/pilot.py`

**Interfaces:**
- Produces: `audit_resource_mapping(symbols, pilot_mask, layer_channel_uses) -> dict[str, object]`
- Produces: `trace_stage1_forward(codec, model, batch, config, retain_grad=False) -> dict[str, Tensor | object]`
- Produces: `assert_stage1_trace_integrity(trace, expected_shape) -> dict[str, object]`

- [ ] **Step 1: Write failing pilot-overwrite accounting and round-trip tests**

```python
def test_pilot_audit_counts_overwritten_encoder_symbols():
    symbols = torch.arange(32).reshape(1, 8, 4).to(torch.complex64)
    mask = make_pilot_mask(symbols.shape, spacing=2, time_spacing=2)
    audit = audit_resource_mapping(symbols, mask, (8, 8, 8, 8))
    assert audit["grid_resources_per_sample"] == 32
    assert audit["pilot_resources_per_sample"] == 8
    assert audit["nonpilot_data_resources_per_sample"] == 24
    assert audit["encoder_symbols_per_sample"] == 32
    assert audit["overwritten_encoder_symbols_per_sample"] == 8
    assert audit["resource_mapping_defect"] is True

def test_uniform_allocate_deallocate_is_exact_without_channel():
    result = allocate_resources(symbols, torch.ones_like(symbols.real), (8, 8, 8, 8), mode="uniform")
    torch.testing.assert_close(deallocate_resources(result.symbols, result.resource_to_source), symbols)
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_stage1_dataflow_integrity.py -q`
Expected: missing diagnostic module/function failure.

- [ ] **Step 3: Implement read-only resource accounting and forward tracing**

Trace target, each encoder branch input, data symbols, allocated symbols, pilot mask/transmitted symbols, received/equalized grids, pilot-removed resources, decoder input, reconstruction, states, gates, allocation map, and ordering. Record shapes, zero fractions, finite status, and hashes. Do not change production mapping.

- [ ] **Step 4: Add shape/order/pilot-count tests and run GREEN**

Cover all-one gates, equal power fractions, no dropped layer partition before pilots, pilot removal count, decoder input nonzero/finite, `[B,L,T,D]` ordering, and no detach between data symbols and loss.

Run: `pytest tests/test_stage1_dataflow_integrity.py -q`
Expected: all tests pass and current pilot accounting reports the suspected mismatch.

- [ ] **Step 5: If and only if the accounting plus O2/O3 proves a defect, add a separate failing regression test before the smallest fix**

The regression must express the required invariant: encoder data-symbol count equals usable non-pilot destinations, or mapping explicitly reserves pilots without silently discarding emitted data. Do not choose or implement a fix during this task step; defer until the ladder locates the first failing stage.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/speech_jscc/diagnostics/dataflow.py tests/test_stage1_dataflow_integrity.py
git commit -m "test: audit stage1 resource and pilot dataflow"
```

### Task 3: Gradient and optimizer-update audit

**Files:**
- Create: `src/speech_jscc/diagnostics/gradients.py`
- Create: `tests/test_stage1_gradient_path.py`

**Interfaces:**
- Produces: `gradient_update_audit(codec, model, batch, config, weights, normalization) -> dict[str, object]`
- Consumes: `trace_stage1_forward(..., retain_grad=True)` from Task 2.

- [ ] **Step 1: Write failing tests for branch/decoder gradients and updates**

Build a small `MockContinuousCodec`/`SpeechJSCC` case. Assert every `encoder.layer_encoders.N` group and `decoder` has finite nonzero gradient norm, Adam changes both sides, codec gradients remain absent and parameters unchanged, and intermediate gradients exist for data symbols, allocated symbols, received, equalized, decoder input, and reconstruction.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_stage1_gradient_path.py -q`
Expected: missing audit function failure.

- [ ] **Step 3: Implement one-step audit**

Snapshot named parameters, retain intermediate gradients, run the exact normalized Stage-1 loss, call backward, collect gradient statistics before clipping, perform one configured Adam step, and collect update/relative-update norms. Include optimizer membership and exact trainable counts. Never call the normal helper that detaches returned intermediates.

- [ ] **Step 4: Run GREEN**

Run: `pytest tests/test_stage1_gradient_path.py -q`
Expected: all assertions pass or expose a specific broken boundary without changing production code.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/speech_jscc/diagnostics/gradients.py tests/test_stage1_gradient_path.py
git commit -m "test: audit stage1 gradients and updates"
```

### Task 4: Bounded overfit experiment engine and thresholds

**Files:**
- Create: `src/speech_jscc/diagnostics/overfit.py`
- Create: `tests/test_stage1_overfit_diagnostics.py`

**Interfaces:**
- Produces: `OverfitThreshold(loss_improvement, absolute_loss, min_power_ratio=0.01, min_cosine=0.0, min_correlation=0.0)`
- Produces: `classify_overfit_result(stage, result) -> tuple[bool, list[str]]`
- Produces: `run_overfit_stage(stage, config, device, steps, seed) -> dict[str, object]`
- Produces: `run_overfit_ladder(..., continue_after_failure=False) -> list[dict[str, object]]`

- [ ] **Step 1: Write failing threshold and sequencing tests**

Test OR semantics for O0–O5 loss criteria, AND semantics for alignment guards, O6/O7 5% plus positive correlation, default stop at first failure, continuation flag behavior, and refusal to run a later stage when prerequisites are unevaluated.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_stage1_overfit_diagnostics.py -q`
Expected: missing overfit module failure.

- [ ] **Step 3: Implement O0 and O1 with small-shape test hooks**

O0 optimizes decoder parameters only against a fixed synthetic complex input/latent. O1 optimizes encoder and decoder with direct complex-symbol bypass and fixed zero states. Both collect initial/best/final/zero losses, relative improvement, step to best, final latent metrics, and first/last encoder/decoder gradient norms.

- [ ] **Step 4: Verify O0/O1 memorization tests**

Use reduced shapes and sufficient deterministic iterations to assert substantial loss reduction without a slow marker.

- [ ] **Step 5: Implement O2–O5 fixed-path adapters**

O2 uses exact uniform allocation/deallocation without pilots/channel. O3 uses a single fixed clean multipath realization, zero noise, fixed pilots, DFT LS, and observable state. O4 adds fixed 10 dB noise. O5 adds a fixed barrage jammer at 0 dB JSR with independent fixed channel. Assert fixed stochastic tensors hash identically across steps.

- [ ] **Step 6: Implement O6/O7 distribution adapters**

O6 changes channel/noise hashes each step over a small fixed latent set, no jammer, SNR `[5,15]`. O7 uses the configured full distribution. Test only stochastic-policy/hash behavior at small shapes; real bounded execution happens later.

- [ ] **Step 7: Run GREEN**

Run: `pytest tests/test_stage1_overfit_diagnostics.py -q`
Expected: all tests pass.

- [ ] **Step 8: Commit Task 4**

```bash
git add src/speech_jscc/diagnostics/overfit.py tests/test_stage1_overfit_diagnostics.py
git commit -m "feat: add bounded stage1 overfit ladder"
```

### Task 5: Checkpoint, provenance, receiver-state, capacity, and log audits

**Files:**
- Create: `src/speech_jscc/diagnostics/audits.py`
- Extend: all four focused test files as appropriate

**Interfaces:**
- Produces: `audit_checkpoint(path, model_factory, initialization_seed) -> dict[str, object]`
- Produces: `audit_validation_provenance(config, checkpoint, batches) -> dict[str, object]`
- Produces: `audit_receiver_states(traces, feature_names) -> list[dict[str, object]]`
- Produces: `audit_capacity(model, representation_shape, pilot_mask) -> dict[str, object]`
- Produces: `summarize_metrics_log(path) -> dict[str, object]`

- [ ] **Step 1: Add failing tests for best/last loading, forbidden keys, counts, provenance wording, and missing historical fields**

Use temporary small checkpoints. Require `validation_identity="exact"` only with all hashes/IDs/seeds present; otherwise require `"deterministically_regenerated_same_configuration"` and a list of missing evidence.

- [ ] **Step 2: Verify RED**

Run the focused tests and confirm missing functions cause failure.

- [ ] **Step 3: Implement checkpoint and capacity audits**

Report encoder/decoder norms, reproducible initialization deltas when possible, optimizer step values and LR, nonfinite counts, forbidden keys, parameter counts, symbol/output powers, state distributions, latent dimensions, pilot counts, usable data resources, and effective compression ratio.

- [ ] **Step 4: Implement provenance, receiver-state, and historical-log audits**

Hash tensors with stable CPU byte serialization. Derive clamp bounds from `observable_channel_state.py`. Parse present JSONL fields and explicitly enumerate absent requested metrics.

- [ ] **Step 5: Run focused tests GREEN and commit**

```bash
git add src/speech_jscc/diagnostics/audits.py tests/test_stage1_*.py
git commit -m "feat: audit stage1 checkpoints and provenance"
```

### Task 6: Checkpoint diagnostic CLI, waveform metrics, plots, and reports

**Files:**
- Create: `diagnose_stage1_learning.py`
- Create: `src/speech_jscc/diagnostics/reporting.py`
- Modify: `src/speech_jscc/diagnostics/__init__.py`

**Interfaces:**
- CLI: `python diagnose_stage1_learning.py --config PATH --checkpoint PATH --output_dir PATH`
- Produces all specified JSON/CSV/Markdown/plot/audio artifacts.

- [ ] **Step 1: Add failing CLI orchestration tests with mock codec/checkpoint fixtures**

Test parser defaults, output schema, finite serialization, sibling best/last inspection, and conservative provenance wording.

- [ ] **Step 2: Verify RED**

Run relevant focused tests; expect import/CLI failure.

- [ ] **Step 3: Implement deterministic validation reconstruction and baselines**

Reuse `_fixed_validation_batches` semantics from training while recording that exact identity is unverified unless all provenance fields exist. Evaluate trained, zero, batch mean, optional cached global mean, and fresh random model on the same regenerated batches.

- [ ] **Step 4: Implement waveform comparison**

Decode target/trained/zero/mean latents with the frozen codec. Add waveform SNR and STFT L1 locally; reuse SI-SDR/STOI. Save bounded WAV examples with soundfile. Record unavailable optional metrics without failing.

- [ ] **Step 5: Implement CSV/JSON/Markdown/plot output**

Use atomic write-to-temporary then replace within the output directory. Include the decision tree, loss equation, resource defect status, provenance statement, absent log fields, and no-success-versus-zero guard.

- [ ] **Step 6: Run focused tests GREEN and commit**

```bash
git add diagnose_stage1_learning.py src/speech_jscc/diagnostics tests/test_stage1_*.py
git commit -m "feat: diagnose existing stage1 checkpoints"
```

### Task 7: Overfit CLI and real diagnostic execution

**Files:**
- Create: `overfit_stage1_path.py`
- Output only: `runs/stage1_uniform_1000/diagnostics/`

**Interfaces:**
- CLI: `python overfit_stage1_path.py --config PATH --output_dir PATH [--continue_after_failure] [--max_stage O7]`

- [ ] **Step 1: Add failing CLI sequencing test**

Require default `continue_after_failure=False`, first-failure stop, report creation on failure, and no weighted/full-training entrypoint calls.

- [ ] **Step 2: Verify RED, implement CLI, and verify GREEN**

The CLI selects configured device, prints stage progress, writes `overfit_results.csv` after every stage, and rewrites `overfit_report.md` on completion or failure.

- [ ] **Step 3: Run required focused verification before real artifacts**

```bash
pytest tests/test_stage1_zero_baseline.py \
       tests/test_stage1_gradient_path.py \
       tests/test_stage1_overfit_diagnostics.py \
       tests/test_stage1_dataflow_integrity.py \
       tests/test_stage1_fixed_tx_training.py \
       tests/test_observable_receiver_state.py \
       tests/test_dft_tap_ls.py \
       tests/test_multipath_channel.py -q
```

- [ ] **Step 4: Run checkpoint diagnosis**

```bash
python diagnose_stage1_learning.py \
  --config configs/train_stage1_fixed_tx_uniform.yaml \
  --checkpoint runs/stage1_uniform_1000/stage1_best.pt \
  --output_dir runs/stage1_uniform_1000/diagnostics
```

- [ ] **Step 5: Run bounded ladder without continuation**

```bash
python overfit_stage1_path.py \
  --config configs/train_stage1_fixed_tx_uniform.yaml \
  --output_dir runs/stage1_uniform_1000/diagnostics
```

Expected: O0 begins; each next stage runs only after the prior passes; execution stops and reports at the first failure.

- [ ] **Step 6: Inspect evidence and form one root-cause hypothesis**

Compare checkpoint baselines, gradient audit, resource counts, and first failing stage. If a production defect is confirmed, return to the relevant task, add the failing regression first, apply one minimal fix, and rerun from the affected stage. Otherwise do not change architecture.

- [ ] **Step 7: Commit Task 7 code (not generated run artifacts unless repository convention requires them)**

```bash
git add overfit_stage1_path.py
git commit -m "feat: add stage1 overfit diagnostic CLI"
```

### Task 8: Full regression verification and completion evidence

**Files:**
- Potentially update generated reports under `runs/stage1_uniform_1000/diagnostics/`

- [ ] **Step 1: Run full repository tests**

Run: `pytest tests/ -q`
Expected: all tests pass; optional dependency skips are reported, not hidden.

- [ ] **Step 2: Validate artifacts and commands**

Check every required JSON parses, every CSV has headers/data, every required numeric field is finite or explicitly unavailable, audio files open, plots exist, and reports state the actual validation provenance.

- [ ] **Step 3: Reconcile acceptance criteria**

Confirm exact zero validation loss, trained/zero/mean/random comparisons, layer power/correlation, gradients/updates, resource/pilot/channel path, first failing stage, capacity ratio, receiver-state distributions, test results, final classification, and next-task recommendation are present.

- [ ] **Step 4: Use verification-before-completion before claiming success**

Re-run the focused suite and any affected regression test after the final code change. Capture exact pass/skip counts and diagnostic commands in the completion report.

- [ ] **Step 5: Final handoff**

Report added/modified files, generated artifacts, exact equations and measurements, any verified/fixed bug, first failing stage, final classification, and one scoped next task. Explicitly state that no 20,000-step or weighted training was started.
