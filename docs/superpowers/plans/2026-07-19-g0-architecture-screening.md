# G0 Architecture Screening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add strict, reproducible G0 screening for flat, normalized-flat, PCA, and Conv-Conformer alternatives without changing historical production behavior.

**Architecture:** Keep the existing flat classes unchanged; add focused temporal-model, normalization/PCA, checkpoint, and screening modules. Extend the existing component factory only at the explicit architecture dispatch point and reuse the exposure/content utilities.

**Tech Stack:** Python 3.11, PyTorch, PyYAML, pytest, shell.

## Global Constraints

- Frozen SpeechTokenizer representation is `[B,8,50,1024]`.
- Total budget is 1,920 complex symbols, 240 per layer.
- Historical/default architecture remains `flat_mlp`.
- No channel, mapper, pilot, CSI, equalizer, or jammer in G0.
- No long optimization or full-corpus PCA is run inside Codex.

---

### Task 1: Temporal model and factory

**Files:** Create `src/speech_jscc/models/conv_conformer.py`; modify `src/speech_jscc/models/__init__.py`, `src/speech_jscc/models/system.py`, `src/speech_jscc/experiment.py`; test `tests/test_conformer_block.py`, `tests/test_conv_conformer_jscc.py`.

- [ ] Write shape, gradient, symbol-budget, conditioning, power, head-gradient, and parameter-limit tests and verify RED.
- [ ] Implement Macaron Conformer, layer mixer, FiLM, temporal resampling, encoder/decoder, and explicit factory dispatch.
- [ ] Run focused tests and retain unchanged flat construction.

### Task 2: Strict architecture checkpoints

**Files:** Create `src/speech_jscc/models/architecture_checkpoint.py`; test `tests/test_conv_conformer_checkpoint.py`.

- [ ] Write rejection tests for missing, flat, legacy, normalization, symbol-frame, latent-shape, and partial-load cases and verify RED.
- [ ] Implement metadata creation/validation and strict-only load.
- [ ] Run checkpoint tests.

### Task 3: Train-only normalization and PCA reference

**Files:** Create `src/speech_jscc/diagnostics/latent_normalization.py`, `src/speech_jscc/diagnostics/pca_reference.py`; test `tests/test_latent_normalization.py`, `tests/test_per_layer_pca_reference.py`.

- [ ] Write analytic round-trip, provenance, split rejection, PCA projection, component-budget, and no-covariance tests and verify RED.
- [ ] Implement streaming per-layer statistics and randomized per-layer low-rank PCA.
- [ ] Run focused tests.

### Task 4: Revised gates and screening engine

**Files:** Create `src/speech_jscc/diagnostics/architecture_screening.py`, `diagnose_g0_architecture_screening.py`; test `tests/test_g0_architecture_screening.py`, `tests/test_revised_g0_gate.py`.

- [ ] Write RED tests for direct-only routing, baseline reporting, gate separation, provenance, and long-run acknowledgement.
- [ ] Implement architecture/reference selection, exposure loop, metrics, summaries, and report aggregation.
- [ ] Run focused tests and a bounded smoke check.

### Task 5: External runner and configuration

**Files:** Create `configs/g0_conv_conformer_v1.yaml`, `scripts/run_g0_architecture_screening_external.sh`; test `tests/test_g0_architecture_screening_script.py`.

- [ ] Write RED tests for safe output handling, options, dry-run, and selected-only execution.
- [ ] Implement root-safe shell runner and exact external commands.
- [ ] Run dry-run and shell tests.

### Task 6: Verification

**Files:** Update only defects revealed in related files/tests.

- [ ] Run all requested focused tests.
- [ ] Run `pytest tests/ -q`.
- [ ] Run construction/parameter report, dry-run, and one forward/backward smoke test.
- [ ] Review git diff/status to confirm unrelated tracked and untracked files remain intact.
