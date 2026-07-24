# J3 Provenance Correction and J4 Burst Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the verified J3 PASS with corrected J2-transfer provenance, then add an externally executed random full-band single-interval burst-jammer boundary and training stage.

**Architecture:** Preserve immutable J3 metrics/checkpoint bytes and add sidecar accepted-artifact manifests. Reuse the common content-generalization engine and existing `kind="burst"` channel primitive, adding J4-only mask diagnostics, tail aggregation, gates, validation suites, strict J3 transfer, and launchers.

**Tech Stack:** Python 3.11, PyTorch, pytest, YAML, shell.

## Global Constraints

- Do not rerun or rewrite J3 weights or scientific metrics.
- J4 starts strictly from the corrected accepted J3 checkpoint.
- J4 uses one contiguous non-wrapping full-band time interval and no adaptive modules.
- Long boundary and 4096-step jobs run only through external scripts.

---

### Task 1: Correct and freeze J3 provenance

**Files:**
- Create: `scripts/correct_j3_provenance.py`
- Create: `tests/test_j3_provenance.py`
- Modify: `diagnose_stage1_content_generalization.py`

- [ ] Write tests proving J2 strict-load hash equality, immutable original metrics, corrected parent hashes, accepted manifest creation, and future `j2_transfer` metadata.
- [ ] Run the focused test and verify RED.
- [ ] Implement read-only verification plus sidecar generation and fix future J3 provenance generation.
- [ ] Run the focused test and verify GREEN.
- [ ] Generate `summary.original.json`, `summary.corrected.json`, `provenance_correction.json`, and `accepted_manifest.json` without changing `summary.json` or `diagnostic_last.pt`.

### Task 2: Add J4 burst primitives and scientific gates

**Files:**
- Create: `src/speech_jscc/diagnostics/j4_burst.py`
- Create: `tests/test_burst_jammer.py`
- Create: `tests/test_j4_classification.py`

- [ ] Write failing tests for contiguous full-band masks, rounding, no wrap/leakage, overlap, global/active JSR, paired policies, tail statistics, and PASS/MARGINAL/FAIL precedence.
- [ ] Run the focused tests and verify RED.
- [ ] Implement burst diagnostics, seed policy, tail aggregation, artifact verification, gates, and resume metadata validation.
- [ ] Run the focused tests and verify GREEN.

### Task 3: Add J4 boundary sweep

**Files:**
- Create: `diagnose_j4_burst_boundary.py`
- Create: `configs/conv_conformer_j4_random_burst.yaml`
- Create: `scripts/run_j4_burst_boundary.sh`

- [ ] Add failing tests for the 27-point grid, evidence-based selection, accepted-J3 transfer, no-overwrite, and dry-run.
- [ ] Implement the paired 16-realization sweep, distribution/tail CSVs, worst hashes, plots, and selected distribution.
- [ ] Run dry-run and one small CPU boundary smoke.

### Task 4: Add J4 training integration

**Files:**
- Create: `train_j4_conv_conformer.py`
- Create: `scripts/run_j4_conv_conformer_external.sh`
- Modify: `src/speech_jscc/diagnostics/conv_conformer_integration.py`
- Modify: `diagnose_stage1_content_generalization.py`

- [ ] Add failing tests for strict J3 transfer, random burst diversity/coverage, validation suite stability, resume lineage, summary schema, and no J5 progression.
- [ ] Implement J4 stage policy, validation, metrics, gates, checkpoint provenance, resume, and plots.
- [ ] Run one 3-step CPU forward/backward smoke and both launcher dry-runs.

### Task 5: Verify repository

- [ ] Run focused J3/J4 tests.
- [ ] Run `pytest tests/ -q`.
- [ ] Verify J1/J2/J3 accepted hashes remain unchanged.
- [ ] Report external CUDA commands and explicitly leave J4 scientific results unexecuted.
