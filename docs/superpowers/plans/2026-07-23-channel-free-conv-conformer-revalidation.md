# Channel-Free Conv-Conformer Revalidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a controlled channel-free Conv-Conformer training and waveform evaluation matrix with a frame-preserving ragged CF-2 bottleneck.

**Architecture:** Extend the existing Conv-Conformer with an explicit temporal symbol layout abstraction, then reuse one channel-free training/evaluation engine across CF-1–CF-5. Keep codec decoding differentiable but codec parameters frozen.

**Tech Stack:** Python, PyTorch, PyYAML, SciPy WAV I/O, pytest.

## Global Constraints

- Do not execute long GPU experiments in Codex.
- Do not read latent caches or execute channel, jammer, OFDM, pilot, allocation, CSI, equalizer, gate, or refiner code.
- Preserve frozen SpeechTokenizer parameters and continuous `[B,8,50,1024]` representations.
- Use fixed 64-utterance unseen-speaker evaluation and waveform feasibility criteria.

---

### Task 1: Current implementation audit

**Files:**
- Create: `docs/channel_free_conv_conformer_audit.md`

- [ ] Record current encoder/decoder flows, interpolation, sharing, gradient path,
      parameter counts, channel-use counts, and A/B/C provenance from executable code.
- [ ] Verify all claims with model construction and artifact inspection commands.

### Task 2: Temporal symbol layout and ragged CF-2 path

**Files:**
- Modify: `src/speech_jscc/models/conv_conformer.py`
- Test: `tests/test_channel_free_revalidation.py`

- [ ] Add failing tests for the deterministic ten-frame mask, 240/1,920 valid
      counts, zero invalid slots, masked power normalization, no interpolation,
      pack/unpack identity, and 30/50-frame output shapes.
- [ ] Run the focused tests and confirm RED failures.
- [ ] Implement fixed-width masked temporal symbol encoding and decoding.
- [ ] Run focused tests and confirm GREEN.

### Task 3: Exact summed-latent metrics and differentiable waveform path

**Files:**
- Create: `src/speech_jscc/training/channel_free_revalidation.py`
- Test: `tests/test_channel_free_revalidation.py`

- [ ] Add failing tests for exact decoder-input summation, frame metrics, waveform
      gradient reaching reconstructed latent and JSCC decoder, and absent codec grads.
- [ ] Implement metric and gradient utilities without detach/no-grad around reconstructed decoding.
- [ ] Verify focused tests pass.

### Task 4: Curriculum, checkpoint selection, and training CLI

**Files:**
- Create: `train_channel_free_conv_conformer.py`
- Create: `configs/channel_free_revalidation.yaml`
- Test: `tests/test_channel_free_revalidation.py`

- [ ] Add failing tests for CF-1–CF-5 definitions, curriculum stages, isolated loss
      gradient diagnostics, unambiguous checkpoint names, resume validation, and
      output no-overwrite behavior.
- [ ] Implement the shared training engine and exact artifact schema.
- [ ] Verify dry runs and a maximum five-step smoke run.

### Task 5: Baselines and fixed waveform evaluation

**Files:**
- Create: `eval_channel_free_conv_conformer.py`
- Test: `tests/test_channel_free_revalidation.py`

- [ ] Add failing tests for deterministic unseen-speaker IDs, official and
      continuous-sum baselines, matched-Gaussian error levels, and feasibility rules.
- [ ] Implement fixed 64-utterance evaluation, WAV examples, per-layer/frame CSVs,
      final comparison, and dominant-limitation classification.
- [ ] Verify dry-run behavior and synthetic metric tests.

### Task 6: External orchestration

**Files:**
- Create: `scripts/run_channel_free_revalidation_external.sh`
- Test: `tests/test_channel_free_revalidation_script.py`

- [ ] Add failing tests for strict shell mode, selected experiment execution,
      dry-run, resume, overwrite refusal, and sequential gating.
- [ ] Implement commands for baselines, CF-1–CF-5, and final evaluation.
- [ ] Verify all script dry runs.

### Task 7: Regression verification

**Files:**
- Test: `tests/test_channel_free_revalidation.py`
- Test: `tests/test_channel_free_revalidation_script.py`

- [ ] Run focused channel-free and Conv-Conformer tests.
- [ ] Run existing channel and jammer tests.
- [ ] Run the full test suite.
- [ ] Record exact external CUDA commands and make no feasibility claim before
      fixed 64-utterance waveform results exist.
