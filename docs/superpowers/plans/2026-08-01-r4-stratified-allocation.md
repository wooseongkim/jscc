# R4 Stratified Allocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a paired R4 evaluator for uniform, core-protection, and layer-1-focused transmit-power profiles.

**Architecture:** A focused evaluation module expands normalized layer weights into source-order multipliers and composes them with immutable R4 triplet allocation power.  A CLI reuses the existing physical forward primitives, sharing each allocation and channel condition across profiles, and emits paired statistics/artifacts.

**Tech Stack:** Python 3, PyTorch, PyYAML, pytest, existing R4 OFDM/MRC modules.

## Global Constraints

- Preserve 1,920 source symbols, 5,760 data REs, three repetitions, OFDM profile, decoder, and official SI-SDR metric.
- Use no NMSE-derived priority and no detailed 8-layer ordering in profiles.
- Do not edit existing user changes; create only new files for this feature.
- Refuse full comparison when uniform baseline reference reproduction fails.

---

### Task 1: Allocation primitives

**Files:**
- Create: `src/speech_jscc/evaluation/r4_stratified_allocation.py`
- Test: `tests/test_r4_stratified_allocation.py`

- [ ] Write failing tests for the three profile names, 8x240 source mapping, count-weighted normalization, sqrt amplitude scaling, source-power composition, and invariant counts.
- [ ] Run `pytest -q tests/test_r4_stratified_allocation.py` and confirm import failure.
- [ ] Implement immutable profile definitions and pure tensor helpers.
- [ ] Re-run the focused tests.

### Task 2: Paired statistics and artifact schema

**Files:**
- Modify: `src/speech_jscc/evaluation/r4_stratified_allocation.py`
- Modify: `tests/test_r4_stratified_allocation.py`

- [ ] Write failing tests for deterministic utterance bootstrap, paired grouping, profile-row completeness, and reference check.
- [ ] Run the selected tests and confirm the intended failures.
- [ ] Implement summaries, paired deltas, deterministic CI, and schema validation.
- [ ] Re-run focused tests.

### Task 3: R4 CLI and config

**Files:**
- Create: `evaluate_r4_stratified_allocation.py`
- Create: `configs/eval_r4_stratified_allocation.yaml`
- Modify: `tests/test_r4_stratified_allocation.py`

- [ ] Write failing CLI dry-run/validation tests.
- [ ] Implement checkpoint loading, shared mapping/channel conditions, per-profile source scaling, MRC-consistent composed powers, requested artifacts, and dry-run guard.
- [ ] Run dry-run and focused tests.

### Task 4: Verification

**Files:**
- Test: `tests/test_r4_stratified_allocation.py`

- [ ] Run the specified focused test and R4 regression suite.
- [ ] Run a two-utterance/one-realization smoke evaluation when local data/checkpoint dependencies are available.
- [ ] Record only evidence from executed commands; do not launch full 64x2 or 48x2 evaluations.
