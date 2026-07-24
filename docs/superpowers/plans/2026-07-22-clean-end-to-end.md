# Clean End-to-End JSCC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish codec identity, existing-J5 channel-free behavior, ideal resource identity, clean-channel degradation, and a fresh channel-free waveform-calibrated upper bound.

**Architecture:** Add one shared evaluation module, three focused CLIs, and three safe external launchers. Reuse existing codec, G0 direct-bypass, paired channel, manifests, cache, checkpoint, and metric utilities.

**Tech Stack:** Python 3.11, PyTorch, SpeechTokenizer, pytest, CSV/JSON, SciPy WAV.

## Global Constraints

- Preserve J1–J5 artifacts unchanged.
- No long optimization inside Codex.
- No waveform loss or architecture/receiver modification.
- Use real train/validation data; reject test data.

---

### Task 1: Pure path and metric contracts

**Files:** Create `src/evaluation/clean_end_to_end.py`; test `tests/test_clean_end_to_end_pipeline.py`.

**Interfaces:** Produce normalization identity, neutral state, direct bypass,
ideal OFDM identity, summed-latent, oracle replacement, waveform-relative metric,
and checkpoint-selection helpers.

- [ ] Write tests that import the proposed helpers and assert identity/order/gates.
- [ ] Run tests and observe missing-import RED.
- [ ] Implement minimal pure helpers.
- [ ] Run focused tests GREEN.

### Task 2: Status and existing-checkpoint diagnostic

**Files:** Create `diagnose_clean_end_to_end.py`, config, status manifest, and script.

- [ ] Add CLI/no-overwrite tests.
- [ ] Implement codec identity, cached/direct comparison, direct J5 and ideal OFDM phases.
- [ ] Run a 1–2 utterance CPU smoke.

### Task 3: Clean channel ladder

**Files:** Create `eval_clean_channel_ladder.py`, launcher, and `tests/test_clean_channel_ladder.py`.

- [ ] Test C0–C4 condition generation and paired seeds.
- [ ] Implement latent/waveform evaluation and relative metrics.
- [ ] Dry-run and smoke-test only.

### Task 4: Fresh channel-free upper bound

**Files:** Create `train_channel_free_conv_conformer.py`, launcher, and `tests/test_channel_free_training.py`.

- [ ] Test separate latent/waveform checkpoint selection and fresh/warm-start provenance.
- [ ] Implement direct-bypass latent training plus waveform validation selection.
- [ ] Run a one-step CPU smoke only.

### Task 5: Verification and handoff

- [ ] Run focused tests.
- [ ] Run `pytest tests/ -q`.
- [ ] Dry-run all long launchers.
- [ ] Report available local evidence and exact external CUDA commands.
