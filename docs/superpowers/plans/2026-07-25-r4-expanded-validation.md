# R4 Expanded Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic speaker-disjoint expanded validation and checkpoint-selection system for existing R4 checkpoints.

**Architecture:** A focused evaluation module owns manifests, paired statistics,
candidate discovery, and selection. The CLI owns artifact I/O and delegates every
physical forward to the existing `R4WaveformForward`.

**Tech Stack:** Python 3.11, PyTorch, NumPy, PyYAML, pytest.

## Global Constraints

- Do not train or modify model weights except copying the selected checkpoint.
- Preserve the legacy `dev-clean-2` final suite and seeds.
- Use `test-clean` only as `selection_validation`, never as reported test data.
- Do not duplicate the R4 physical path.
- Do not run the full 48x2x3 selection or 64x2x3 final evaluation in Codex.

---

### Task 1: Manifest and overlap protocol

**Files:**
- Create: `src/speech_jscc/evaluation/expanded_validation.py`
- Test: `tests/test_r4_expanded_validation.py`

**Interfaces:**
- Produces: `ManifestEntry`, `build_protocol_manifests`, and
  `audit_protocol_overlap`.

- [ ] Write failing tests for deterministic speaker-balanced selection and all
  pairwise overlap failures.
- [ ] Run `pytest tests/test_r4_expanded_validation.py -q` and verify failure.
- [ ] Implement manifest construction with source/role metadata, crop metadata,
  hashes, and zero-overlap assertions.
- [ ] Re-run focused tests and verify pass.

### Task 2: Deterministic paired statistics

**Files:**
- Modify: `src/speech_jscc/evaluation/expanded_validation.py`
- Test: `tests/test_r4_expanded_validation.py`

**Interfaces:**
- Produces: `explicit_metric_row`, `paired_statistics`, and
  `summarize_checkpoint`.

- [ ] Write failing tests for explicit delta names, deterministic paired
  bootstrap, and utterance-level aggregation.
- [ ] Verify the tests fail for missing functions.
- [ ] Implement the minimal deterministic statistics.
- [ ] Verify focused tests pass.

### Task 3: Candidate discovery and ranking

**Files:**
- Modify: `src/speech_jscc/evaluation/expanded_validation.py`
- Test: `tests/test_r4_expanded_validation.py`

**Interfaces:**
- Produces: `discover_candidates`, `rank_candidates`, and
  `write_selected_checkpoint`.

- [ ] Write failing tests proving light validation cannot directly select a
  checkpoint and nonpassing selections cannot carry passing metadata.
- [ ] Verify expected failures.
- [ ] Implement inventory, gates, ranking, and selected-checkpoint metadata.
- [ ] Verify focused tests pass.

### Task 4: Paired R4 evaluator and CLI

**Files:**
- Create: `evaluate_r4_expanded_validation.py`
- Create: `configs/eval_r4_expanded_validation.yaml`
- Create: `scripts/run_r4_expanded_validation_external.sh`
- Modify: `src/speech_jscc/evaluation/expanded_validation.py`
- Test: `tests/test_r4_expanded_validation.py`

**Interfaces:**
- Consumes: existing `R4WaveformForward` and checkpoint/config utilities.
- Produces: dry-run, constrained smoke, full selection, and separately guarded
  final-test commands.

- [ ] Write failing tests for shared delayed-CSI trajectories, candidate-order
  invariance, output no-overwrite, and final-manifest exclusion.
- [ ] Verify expected failures.
- [ ] Implement pre-generated initial-model CSI trajectories and shared
  candidate evaluation using `torch.no_grad()`.
- [ ] Implement CLI artifact writers and long-run guards.
- [ ] Verify focused tests pass.

### Task 5: Documentation and verification

**Files:**
- Create: `docs/r4_expanded_validation.md`
- Modify: `tests/test_r4_expanded_validation.py`

**Interfaces:**
- Produces exact external commands and audit/report documentation.

- [ ] Add audit, protocol, and command documentation.
- [ ] Run the focused test command.
- [ ] Run the requested related regression test command.
- [ ] Run `pytest tests/ -q` if feasible.
- [ ] Run dry-run and at most 2-candidate x 2-utterance x 1-realization smoke.
- [ ] Inspect every generated artifact and report any remaining external work.

