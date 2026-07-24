# Stage-1 Content-Generalization Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement an externally executed G0–G3 ladder that locates the first unseen-content generalization failure while preserving O6 failure and quarantining exploratory J1 evidence.

**Architecture:** Extend the existing diagnostics with a focused speaker-aware content module and a thin CLI. Reuse existing model/channel/mapping operations and provenance utilities, but keep G-path construction, content splits, metrics, and reports in bounded functions. A guarded shell driver handles subset escalation and stage stopping without running long jobs in Codex.

**Tech Stack:** Python 3.11, PyTorch, pytest, YAML/JSON/CSV/JSONL, Bash.

## Global Constraints

- O6 remains `FAIL`; J1 is `exploratory_failed_parent`, excluded from curriculum/readiness and forbidden as J2 parent.
- Do not alter production architecture, channel/equalizer mathematics, mapping, loss, layer weights, or neural inputs.
- Reject test manifests/caches; SpeechTokenizer remains frozen.
- Codex runs tests, dry-runs, report updates, and at most five smoke steps only.
- No long G/O6/J/Uniform/Weighted optimization runs inside Codex.

---

### Task 1: Evidence Status and Readiness Quarantine

**Files:**
- Modify: `src/speech_jscc/diagnostics/stage1_readiness.py`
- Modify: `evaluate_stage1_readiness.py`
- Create: `tests/test_content_generalization_readiness.py`

**Interfaces:**
- Produces: `classify_distribution_evidence(o6_summary, j1_summary)` and G3 readiness checks.

- [ ] Write failing tests asserting O6 remains failed, J1 becomes `exploratory_failed_parent`, and J1 cannot parent J2.
- [ ] Run the focused tests and verify RED.
- [ ] Implement classification and readiness quarantine without changing result tensors.
- [ ] Run focused tests and verify GREEN.

### Task 2: Speaker-Aware Nested Content Splits

**Files:**
- Create: `src/speech_jscc/diagnostics/content_generalization.py`
- Create: `tests/test_content_generalization_data.py`

**Interfaces:**
- Produces: `parse_speaker_id(path)`, `build_content_subsets(...)`, `build_content_validation_suite(...)`, and stable hashes.

- [ ] Write failing tests for LibriSpeech speaker parsing, unknown fallback, nested 16/64/256/full subsets, group disjointness, fixed seeds, and test-source rejection.
- [ ] Run focused tests and verify RED.
- [ ] Implement deterministic speaker-balanced nested selection and seen/same-speaker/unseen-speaker validation groups.
- [ ] Run focused tests and verify GREEN.

### Task 3: Dataset and Latent Statistics

**Files:**
- Modify: `src/speech_jscc/diagnostics/content_generalization.py`
- Modify: `tests/test_content_generalization_data.py`

**Interfaces:**
- Produces: `aggregate_dataset_statistics(examples, metadata)` including latent power/mean/std, duration, speaker counts, and preprocessing hash.

- [ ] Write analytic failing tests for statistics and preprocessing hash sensitivity.
- [ ] Run tests and verify RED.
- [ ] Implement streaming statistics and metadata aggregation without loading full corpora simultaneously.
- [ ] Run tests and verify GREEN.

### Task 4: G0–G3 Path Execution and Metrics

**Files:**
- Modify: `src/speech_jscc/diagnostics/content_generalization.py`
- Create: `diagnose_stage1_content_generalization.py`
- Create: `tests/test_content_generalization_paths.py`

**Interfaces:**
- Produces: `run_content_path(stage, ...)`, fixed/random realization policies, aggregate/per-layer metrics, and explicit `layer0_summary`.

- [ ] Write failing tests for G0 bypass, exact G1 pilot-reserved identity, fixed G2 hashes, changing G3 hashes, no oracle inputs, and layer-0 output.
- [ ] Run tests and verify RED.
- [ ] Implement the four paths by composing existing encoder/decoder, mapping, and paired-channel operations.
- [ ] Implement CLI long-run guard, dry-run, training/validation loop, and finite assertions.
- [ ] Run focused tests plus at most a three-step smoke run.

### Task 5: Gates, Checkpoints, and Reports

**Files:**
- Modify: `src/speech_jscc/diagnostics/content_generalization.py`
- Modify: `diagnose_stage1_content_generalization.py`
- Create: `tests/test_content_generalization_reporting.py`

**Interfaces:**
- Produces: strict same-stage/subset resume validation, group gates, subset/stage progression decisions, and consolidated reports.

- [ ] Write failing tests for per-group 5% gates, subset escalation, full-subset stage stop, cross-stage/subset resume rejection, and unexecuted-state reporting.
- [ ] Run tests and verify RED.
- [ ] Implement checkpoint provenance and report aggregation under `runs/stage1_content_generalization/`.
- [ ] Run focused tests and verify GREEN.

### Task 6: Safe External Ladder Script

**Files:**
- Create: `scripts/run_stage1_content_generalization_external.sh`
- Create: `scripts/evaluate_stage1_content_generalization.sh`
- Create: `tests/test_content_generalization_external.py`

**Interfaces:**
- Produces guarded G0→G3 and 16→64→256→full execution, stopping only after full-subset stage failure.

- [ ] Write failing script tests for strict shell mode, dry-run, overwrite/resume, sibling pending logs, subset escalation, and stage stop.
- [ ] Run tests and verify RED.
- [ ] Implement scripts and run all with `--dry-run` only.
- [ ] Run focused tests and verify GREEN.

### Task 7: Integration Verification

**Files:**
- Modify only files implicated by failures.

- [ ] Run all new focused tests.
- [ ] Run existing O5/O6/J/Stage-1/channel/mapping tests.
- [ ] Run `pytest tests/ -q` and record exact result.
- [ ] Regenerate distribution/readiness reports and confirm O6/J1 classifications.
- [ ] Run content scripts with `--dry-run`; do not execute long optimization.
- [ ] Inspect status/diff to confirm unrelated untracked files remain untouched.
