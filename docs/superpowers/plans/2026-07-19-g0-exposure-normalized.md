# G0 Exposure-Normalized Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare G0 subset sizes using true epoch/sample-presentation budgets, constant-predictor baselines, and fixed validation groups without changing the architecture.

**Architecture:** Add a versioned epoch sampler and baseline/metric helpers beside the content diagnostic module, then expose a dedicated G0 CLI. A report aggregator and guarded external script run subsets independently and preserve existing G0 evidence.

**Tech Stack:** Python, PyTorch, pytest, JSONL/CSV/Markdown, Bash.

## Global Constraints

- Keep G0 direct path and frozen SpeechTokenizer unchanged.
- No G1–G3, O6, J, Uniform, or Weighted runs.
- No long optimization inside Codex; only subset-16 up to two epochs smoke.
- Reject test data and preserve unrelated local changes.

### Task 1: Epoch Sampler and Exposure Accounting

**Files:** Create `src/speech_jscc/diagnostics/g0_exposure.py`; create `tests/test_g0_exposure_sampler.py`.

- [ ] Test deterministic without-replacement epochs, batch tails, step formula, and presentation counts (RED).
- [ ] Implement sampler and checkpoint schedule (GREEN).

### Task 2: Baselines and Metrics

**Files:** Modify `g0_exposure.py`; create `tests/test_g0_exposure_baselines.py`.

- [ ] Test zero/global/layerwise/speaker means analytically and unavailable speaker behavior (RED).
- [ ] Implement baseline evaluation, optimal scale, layer 0, layers 1–7, and gradient summaries (GREEN).

### Task 3: Dedicated CLI and Checkpoints

**Files:** Create `diagnose_g0_exposure_normalized.py`; create `tests/test_g0_exposure_cli.py`.

- [ ] Test long-run guard, provenance, resume rejection, best/final checkpoint naming, and no test input (RED).
- [ ] Implement epoch loop, fixed validation, early-stop/plateau logic, artifacts, and strict resume (GREEN).
- [ ] Run dry-run and subset-16 one-to-two epoch smoke only.

### Task 4: Reports and External Script

**Files:** Create `scripts/summarize_g0_exposure.py`, `scripts/run_g0_exposure_normalized_external.sh`, `tests/test_g0_exposure_reporting.py`.

- [ ] Test report schemas/classification and shell safety (RED).
- [ ] Implement aggregate CSV/manifest/Markdown and sequential subset runner (GREEN).
- [ ] Run script with `--dry-run` only.

### Task 5: Verification

- [ ] Run new focused tests and existing content/O6/readiness tests.
- [ ] Run full `pytest tests/ -q`.
- [ ] Verify no long optimization or downstream stage ran.
