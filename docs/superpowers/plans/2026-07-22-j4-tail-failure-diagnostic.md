# J4 Tail-Failure Paired Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose the Layer-7 negative tail at the observed J4 worst condition without changing training or accepted artifacts.

**Architecture:** Add pure selection/statistics/classification helpers, a diagnostic-only equalizer cap in the existing paired evaluator, and a no-training CLI that creates one stochastic realization then derives every comparison mode from it. The CLI writes supplements and reports to a new output directory and never mutates J4 summaries or checkpoints.

**Tech Stack:** Python, PyTorch, pytest, YAML, CSV/JSON, matplotlib.

## Global Constraints

- Do not start J5 or retrain J4.
- Preserve original J4 evidence and all production defaults.
- Every paired mode must reuse latent, masks, channels, jammer waveform, AWGN, and seed.
- Full diagnostic requires explicit external execution acknowledgement.

---

### Task 1: Pure diagnostic policy and statistics

**Files:**
- Create: `src/speech_jscc/diagnostics/j4_tail.py`
- Test: `tests/test_j4_tail_diagnostic.py`

- [ ] Write failing tests for worst-condition selection, Wilson intervals, grouped failure rates, and root-cause classification.
- [ ] Run `pytest tests/test_j4_tail_diagnostic.py -q` and confirm missing APIs fail.
- [ ] Implement the minimal pure helpers and rerun the tests.

### Task 2: Paired channel interventions

**Files:**
- Modify: `src/evaluation/paired.py`
- Create: `src/speech_jscc/diagnostics/j4_tail.py`
- Test: `tests/test_j4_tail_diagnostic.py`

- [ ] Add failing tests for data-only masks, tensor identity, oracle CSI, and gain-capped ZF.
- [ ] Add an optional diagnostic-only gain cap whose default is `None` and leaves existing behavior bit-for-bit unchanged.
- [ ] Rerun focused paired and legacy tests.

### Task 3: Diagnostic CLI and artifact writer

**Files:**
- Create: `diagnose_j4_tail_failure.py`
- Create: `configs/j4_failure_diagnostic.yaml`
- Create: `scripts/run_j4_tail_diagnostic_external.sh`
- Test: `tests/test_j4_tail_diagnostic.py`

- [ ] Add failing CLI dry-run/no-overwrite tests.
- [ ] Implement checkpoint verification, expanded unseen-speaker selection, paired modes, checkpoint comparisons, reports, plots, and immutable summary supplement.
- [ ] Run a dry run and a one-content/one-realization CPU smoke test.

### Task 4: Verification

**Files:**
- Modify only files required by failing regression tests.

- [ ] Run focused J4/J3 tests.
- [ ] Run `pytest tests/ -q`.
- [ ] Inspect original summary hash before and after verification.
- [ ] Report the external CUDA command; do not execute the long diagnostic locally.
