# O5 Root-Cause Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement guarded offline O5 condition tooling and safe external long-run scripts without changing production Stage-1 behavior.

**Architecture:** Pure diagnostic functions construct/hash paired fixed realizations and evaluate explicitly labeled offline branches. A guarded CLI handles optimization, logging, checkpoint/resume, and reporting; shell wrappers own long execution.

**Tech Stack:** Python, PyTorch, pytest, YAML/JSON/CSV, Bash.

## Global Constraints

- No Codex-side run above five optimization steps.
- No production model/channel/estimator/resource/loss/state/checkpoint changes.
- No oracle neural inputs or production-selectable oracle subtraction.
- Existing uncommitted local work is preserved.

### Task 1: Fixed conditions, hashes, JSR, scale, and slope primitives

**Files:** create `src/speech_jscc/diagnostics/o5_root_cause.py`; create `tests/test_o5_root_cause_diagnostics.py`.

- [ ] Write failing tests for C1/C3/C5 masks, paired hashes, JSR conventions, oracle isolation/subtraction, data-only metrics, analytic scale, and slopes.
- [ ] Run `pytest tests/test_o5_root_cause_diagnostics.py -q` and confirm RED.
- [ ] Implement minimal pure functions and fixed-realization dataclasses.
- [ ] Re-run and confirm GREEN.

### Task 2: Guarded CLI, learning metrics, and diagnostic checkpoints

**Files:** create `diagnose_o5_root_cause.py`; extend both focused tests.

- [ ] Write failing tests for dry-run, long-run guard, output collision, per-layer JSONL, checkpoint labels, and exact resume.
- [ ] Implement CLI orchestration, fixed optimization, metrics, checkpoint/resume, summaries, and aggregate reporting.
- [ ] Verify with tests and one optional 3-step smoke only.

### Task 3: External scripts and extension policy

**Files:** create `scripts/run_o5_root_cause_external.sh`, `scripts/run_o5_extension_external.sh`, `tests/test_o5_external_execution.py`.

- [ ] Write failing static/dry-run tests for strict Bash mode, C0–C6 commands, separate outputs/logs, overwrite guard, sensitivity commands, and resume extensions.
- [ ] Implement scripts and generated `external_commands.md` behavior.
- [ ] Verify scripts with `bash -n` and dry-run tests; never execute long commands.

### Task 4: Verification and handoff

- [ ] Run the mandated focused pytest command.
- [ ] Run `pytest tests/ -q`.
- [ ] Run the specified CLI dry-run and optionally one <=5-step smoke.
- [ ] Inspect generated command text and confirm no production diff.
- [ ] Report exact external commands, expected paths, and confirmation that no long run ran inside Codex.
