# R4 Waveform-Aware Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add resumable 20,000-step waveform-connected fine-tuning through the exact R4 time-domain OFDM, LS-CSI, repetition-3 coherent-MRC path without changing the codec, architecture, allocator, combiner, checkpoint shapes, or production G/J path.

**Architecture:** Extract the differentiable per-TTI physical forward from `evaluate_repetition3_mrc.py` into a focused `R4WaveformForward` module. Both evaluator and trainer call this shared implementation. A separate trainer owns data ordering, curriculum, fixed validation, checkpoint selection, resume state, logging, and external execution policy.

**Tech Stack:** Python 3.11, PyTorch complex autograd, SpeechTokenizer, pytest, YAML, JSONL/CSV.

## Global Constraints

- Initialize strictly from `runs/waveform_aware_wireless/clean_channel_training/best_waveform_si_sdr.pt`.
- Total budget is exactly 20,000 optimizer steps: 4,000/8,000/8,000.
- Full 20,000-step training and 64×2×3 evaluation are external only.
- SpeechTokenizer parameters remain frozen while decoder-input gradients remain enabled.
- R4 profile, allocator, power allocation, coherent MRC, source count, and packet energy remain unchanged.
- No jammer, architecture change, refiner, learned allocation, unequal repetition, or production G/J edit.

---

### Task 1: Shared curriculum and selection contracts

**Files:**
- Create: `tests/test_r4_waveform_finetune.py`
- Create: `src/speech_jscc/training/r4_waveform_finetune.py`

**Interfaces:**
- Produces `R4Curriculum.stage(step)`, `sample_snr(step, seed)`,
  `clean_gate(metrics, thresholds)`, and `CheckpointSelector`.

- [ ] Write failing tests for exact stage boundaries, categorical probabilities,
  deterministic sampling, explicit delta names, clean eligibility, and the rule
  that `best_clean_gate.pt` cannot be selected when any constraint fails.
- [ ] Run `pytest tests/test_r4_waveform_finetune.py -q` and verify import/behavior failures.
- [ ] Implement immutable curriculum dataclasses, deterministic categorical
  sampling, clean margins, and checkpoint selection.
- [ ] Re-run the focused tests and verify they pass.

### Task 2: Autograd-safe allocation and shared physical forward

**Files:**
- Modify: `src/channels/global_triplet_allocator.py`
- Create: `src/speech_jscc/training/r4_waveform_finetune.py`
- Modify: `evaluate_repetition3_mrc.py`
- Test: `tests/test_r4_waveform_finetune.py`
- Test: `tests/test_r4_global_triplets.py`

**Interfaces:**
- Produces `R4ForwardCondition`, `R4WaveformOutput`, and
  `R4WaveformForward.forward(representation, waveform, condition, delayed_csi,
  generator, training)`.
- Evaluator consumes the same shared physical forward.

- [ ] Write a failing mapping/deallocation gradient test using a leaf source tensor.
- [ ] Write failing noiseless/LS finite forward consistency tests.
- [ ] Replace differentiable destination in-place writes with out-of-place
  scatter/index operations where necessary; mapping metadata remains detached.
- [ ] Implement the shared forward using `insert_physical_pilots`,
  `modulate_tti`, `apply_tti_multipath`, `demodulate_tti`,
  `estimate_comb_dft_ls`, `allocate_global_balanced_triplets`, and
  `coherent_mrc` directly.
- [ ] Return all requested physical, mapping, decoder-state, CSI, EVM, SINR,
  power, and auxiliary tensors in `R4WaveformOutput`.
- [ ] Route the R4 evaluator through the shared physical helper without changing
  its metric definitions or R3 legacy route.
- [ ] Verify focused R4/evaluator tests.

### Task 3: Frozen codec waveform gradient and combined objective

**Files:**
- Modify: `src/speech_jscc/training/r4_waveform_finetune.py`
- Test: `tests/test_r4_waveform_finetune.py`

**Interfaces:**
- Produces `freeze_codec_for_input_gradient`, `r4_training_objective`, and
  `component_gradient_norms_by_module`.

- [ ] Write a failing real/mock-codec test proving reconstructed-latent,
  JSCC-encoder, and JSCC-decoder gradients are nonzero while codec gradients are
  `None`.
- [ ] Implement frozen codec decoding using
  `decode_frozen_representation_with_gradient` with no reconstructed-latent
  detach and no decoder `no_grad`.
- [ ] Reuse per-layer NMSE and differentiable multi-resolution STFT; add waveform
  L1 and pure-neural channel-free regularization.
- [ ] Log each component, effective weight, encoder norm, and decoder norm.
- [ ] Verify the explicit gradient test and finite one-pass backward.

### Task 4: Strict checkpoint compatibility and resumable trainer

**Files:**
- Create: `configs/train_r4_waveform_finetune.yaml`
- Create: `train_r4_waveform_finetune.py`
- Test: `tests/test_r4_waveform_finetune.py`

**Interfaces:**
- Consumes the shared forward/objective and accepted checkpoint.
- Produces four named checkpoints, exact resume metadata, resolved configuration,
  environment, command, metrics JSONL, and validation manifest.

- [ ] Write failing compatibility tests for codec type/path, latent
  `[8,50,1024]`, `conv_conformer_v1`, 1,920 symbols, channel state 8, R4,
  repetition 3, and strict model loading.
- [ ] Write a failing resume round-trip test for global step, optimizer,
  curriculum, best scores, validation IDs/seeds, delayed report, and RNG state.
- [ ] Implement strict validation and optimizer scope limited to encoder/decoder.
- [ ] Implement deterministic epoch-aware waveform ordering and shared-channel
  per-batch TTI progression.
- [ ] Implement Stage A/B/C learning rates, losses, SNR distributions, and exact
  checkpoint state.
- [ ] Implement overwrite refusal, `--resume`, `--dry-run`, `--smoke-steps`,
  `--allow-long-run`, finite diagnostics, and nonzero failure status.
- [ ] Verify dry-run and resume tests.

### Task 5: Fixed paired validation and checkpoint selection

**Files:**
- Modify: `src/speech_jscc/training/r4_waveform_finetune.py`
- Modify: `train_r4_waveform_finetune.py`
- Test: `tests/test_r4_waveform_finetune.py`

**Interfaces:**
- Produces deterministic light/full validation rows, paired initial-baseline
  deltas, pure-neural regression, noiseless-R4 regression, clean margins, and
  checkpoint scores.

- [ ] Write failing tests for fixed held-out IDs/seeds and exact paired baseline
  reuse.
- [ ] Implement light validation every 250 steps with 4–8 utterances and one seed.
- [ ] Implement full validation every 1,000 steps with configured utterances and
  three fixed seeds.
- [ ] Record `si_sdr_absolute`, `delta_si_sdr_vs_clean_codec`, and
  `delta_si_sdr_vs_initial_r4` separately.
- [ ] Implement pure-neural and noiseless-R4 regressions and clean constraint
  normalized margins.
- [ ] Save each named checkpoint only under its specified policy.
- [ ] Verify selection tests.

### Task 6: External scripts and final evaluator

**Files:**
- Create: `scripts/run_r4_waveform_finetune_external.sh`
- Create: `scripts/run_r4_waveform_finetune_eval_external.sh`
- Modify: `evaluate_r4_repetition3_mrc.py`
- Create: `docs/r4_waveform_finetune.md`
- Test: `tests/test_r4_waveform_finetune_scripts.py`

**Interfaces:**
- Produces dry-run, smoke, full training, resume, light validation, and final
  64×2×3 commands.

- [ ] Write failing script contract tests for `set -euo pipefail`, repository-root
  execution, `tee`, overwrite refusal, resume, dry-run, and nonzero failures.
- [ ] Add external scripts and checkpoint-selection metadata to final evaluator.
- [ ] Add concise documentation with all exact commands and artifact paths.
- [ ] Verify script dry-runs without starting long work.

### Task 7: Bounded verification

**Files:**
- No new production files.

- [ ] Run focused tests:
  `pytest tests/test_r4_waveform_finetune.py tests/test_r4_waveform_finetune_scripts.py tests/test_r4_global_triplets.py tests/test_repetition_mrc.py tests/test_physical_ofdm_profiles.py -q`.
- [ ] Run one actual checkpoint forward/backward and verify encoder/decoder
  gradient norms are nonzero and codec gradients are absent.
- [ ] Run at most 3 optimizer steps with smoke output.
- [ ] Run all repository tests with `pytest tests/ -q`.
- [ ] Inspect git status and confirm no production G/J file changed.
- [ ] Report that 20,000-step training and 64×2×3 final evaluation were not run.
