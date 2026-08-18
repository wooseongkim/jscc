# Physical OFDM Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add evaluation-only NR-inspired R2/R3 time-domain OFDM profiles with causal delayed-CSI selection and bounded power allocation.

**Architecture:** A physical waveform module defines OFDM numerology and modulation. A separate FDD mapping module selects 1,920 candidate RE and performs causal allocation. A standalone evaluator loads the accepted checkpoint without changing production paths.

**Tech Stack:** Python, PyTorch complex tensors, pytest, YAML.

## Global Constraints

- Preserve the accepted checkpoint and 1,920-symbol CF-2 interface.
- No repetition, MRC, coding, jammer, training, or production G/J changes.
- R3 is the default physical profile; legacy 64×32 is regression-only.
- Use LMMSE with known transmit amplitude and retain regularized shrinkage.

---

### Task 1: Physical profile and waveform primitives

**Files:**
- Create: `src/channels/physical_ofdm.py`
- Test: `tests/test_physical_ofdm_profiles.py`

- [ ] Write failing tests for R2/R3 timing, active bins, masks, CP, identity, and channel equivalence.
- [ ] Run the focused tests and verify missing-interface failures.
- [ ] Implement immutable profiles, masks, unitary modulation/demodulation, and convolution.
- [ ] Run focused tests until green.

### Task 2: Candidate selection, allocation, and power

**Files:**
- Create: `src/speech_jscc/diagnostics/physical_fdd.py`
- Modify: `tests/test_physical_ofdm_profiles.py`

- [ ] Write failing tests for 1,920-RE selection, bijection, causality, energy, and bounds.
- [ ] Verify failures.
- [ ] Implement deterministic bootstrap, delayed selection, stratified mapping, bounded weights, and LMMSE recovery.
- [ ] Run focused tests until green.

### Task 3: Evaluation and reporting

**Files:**
- Create: `configs/ofdm_nr_like_r2.yaml`
- Create: `configs/ofdm_nr_like_r3.yaml`
- Create: `evaluate_physical_fdd.py`
- Create: `scripts/run_physical_ofdm_profiles_external.sh`
- Create: `runs/physical_ofdm_profiles/implementation_audit.md`
- Create: `runs/physical_ofdm_profiles/physical_environment.md`

- [ ] Add script/config contract tests and verify failures.
- [ ] Implement R2/R3 smoke and R3 full modes with immutable outputs.
- [ ] Record physical, stochastic, channel, latent, waveform, and energy metadata.
- [ ] Run dry-run and short CPU smoke.

### Task 4: Regression verification

- [ ] Run physical-profile focused tests.
- [ ] Run existing CF-2 and causal-FDD tests.
- [ ] Run the full repository suite.
- [ ] Report the exact external R3 command without running the long evaluation locally.
