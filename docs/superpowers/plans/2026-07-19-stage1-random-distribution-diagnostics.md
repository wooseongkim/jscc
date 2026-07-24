# Stage-1 Random-Distribution Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce internally consistent O5 evidence, audit the historical O5/C1 protocol difference, and prepare reproducible externally executed O6 and J1–J5 diagnostics plus a strict Uniform Stage-1 readiness gate.

**Architecture:** O5 reporting and protocol auditing remain read-only consumers of existing artifacts. O6 and J1–J5 share one versioned random-distribution engine that owns subsets, seed derivation, sampling, validation, metrics, checkpoints, lineage, gates, and forgetting checks; thin CLIs and shell scripts provide guarded external execution. A separate readiness evaluator validates artifacts and lineage without mutating them.

**Tech Stack:** Python 3.11, PyTorch, pytest, YAML/JSON/CSV/JSONL, Bash.

## Global Constraints

- Preserve `pilot_reserved_v1`, the JSCC architecture, loss, channel/equalizer mathematics, observable receiver state, and checkpoint compatibility rules.
- Never use test manifests/caches for training or model selection; reject them in code.
- Codex runs only tests, report regeneration, step-0/dry-run checks, and at most five optimization steps per new path.
- O6/J1–J5/full Uniform/Weighted long jobs must not run in Codex; Weighted training is out of scope.
- Preserve unrelated and untracked local files.
- An extension is valid only when condition, seed, latent/model/realization hashes, and checkpoint lineage agree.

---

### Task 1: O5 rescaled metrics and budget-separated reports

**Files:**
- Modify: `src/speech_jscc/diagnostics/o5_root_cause.py`
- Modify: `diagnose_o5_root_cause.py`
- Modify: `scripts/summarize_o5_root_cause.py`
- Create: `tests/test_o5_report_regeneration.py`
- Create: `scripts/regenerate_o5_reports.sh`

**Interfaces:**
- Produces: `optimal_scale_metrics(reconstruction, target, weights, epsilon)` with `global_power_weighted_rescaled_nmse` and `stage1_layerwise_rescaled_loss`.
- Produces: exact-step extraction and lineage-validated extension comparison functions used by report generation.

- [ ] Write analytic failing tests for distinct global/layerwise scale metrics and exact step-500 extraction.
- [ ] Run `pytest tests/test_o5_report_regeneration.py -q` and confirm failures are caused by missing APIs/fields.
- [ ] Implement the metric helper and migrate diagnostic records without treating the global value as Stage-1 loss.
- [ ] Implement report regeneration from each `metrics.jsonl` step-500 record and same-trajectory extension validation.
- [ ] Generate `aggregate_comparison_500.{csv,json}`, `extension_comparison.{csv,json}`, and an evidence-specific `root_cause_report.md`.
- [ ] Run the focused test and report regeneration; verify C1 500/1000 values exactly match the approved values.

### Task 2: Original O5 versus C1 protocol audit

**Files:**
- Create: `src/speech_jscc/diagnostics/o5_protocol_audit.py`
- Create: `audit_o5_protocol_difference.py`
- Create: `tests/test_o5_protocol_audit.py`

**Interfaces:**
- Produces: `Comparison(status, old_value, new_value, evidence)` where status is `same`, `numerically_equivalent`, `different`, or `unknown`.
- Produces: step-0 hash comparison and static resolved-protocol rows.

- [ ] Write failing tests for unknown historical evidence, intentional hash differences, and seed 23003 versus 23023 classification.
- [ ] Run the focused tests and confirm RED.
- [ ] Implement static protocol extraction and step-0 construction without optimization.
- [ ] Classify the historical result as `different realization, not directly comparable`; do not infer unavailable values.
- [ ] Emit `protocol_difference_report.md`, CSV, hash JSON, and reproduction commands.
- [ ] Run the audit step-0 mode and focused tests.

### Task 3: Shared random-distribution provenance, subsets, and seed derivation

**Files:**
- Create: `src/speech_jscc/diagnostics/random_distribution.py`
- Create: `tests/test_o6_random_clean.py`

**Interfaces:**
- Produces: versioned `StageDefinition`, `DiagnosticSubset`, `SeedDeriver`, `Provenance`, and validation-suite manifest/hash builders.
- Consumes: configured train/valid manifests and SpeechTokenizer cache only.

- [ ] Write failing tests for deterministic 16/8 disjoint IDs, manifest/cache hashes, fixed validation seeds, varying train channel/noise hashes, and test-source rejection.
- [ ] Run focused tests and confirm RED.
- [ ] Implement deterministic subset selection and provenance fields required by the approved specification.
- [ ] Implement domain-separated per-step seed derivation and fixed held-out validation-suite hashing.
- [ ] Run focused tests and confirm GREEN.

### Task 4: Shared engine training, validation, checkpoints, and O6 CLI

**Files:**
- Modify: `src/speech_jscc/diagnostics/random_distribution.py`
- Create: `diagnose_stage1_random_distribution.py`
- Modify: `tests/test_o6_random_clean.py`

**Interfaces:**
- Produces: `run_stage(...)`, `evaluate_suites(...)`, strict checkpoint save/resume validation, O6 gate and readiness classifications.

- [ ] Write failing tests for long-run acknowledgement, changing stochastic hashes, fixed validation, required metrics, checkpoint provenance, and resume lineage.
- [ ] Run focused tests and confirm RED.
- [ ] Implement O6 sampling, V1/V2/V3 evaluation, metrics, finite assertions, checkpointing, and gate classification.
- [ ] Enforce external acknowledgement for more than five steps and ensure dry-run performs no optimization.
- [ ] Run focused tests, dry-run, and at most a five-step smoke test.

### Task 5: J1–J5 definitions, curriculum lineage, gates, and forgetting

**Files:**
- Modify: `src/speech_jscc/diagnostics/random_distribution.py`
- Create: `tests/test_stage1_jammer_curriculum.py`

**Interfaces:**
- Produces exact J1–J5 versioned distributions, curriculum/fresh initialization validation, stage gates, and catastrophic-forgetting records.

- [ ] Write failing tests for every distribution, hidden oracle/jammer-label inputs, parent lineage, mode mismatch rejection, stop-on-failure, and forgetting metrics.
- [ ] Run focused tests and confirm RED.
- [ ] Implement J stage sampling and held-out scenario suites through the common engine.
- [ ] Implement parent checkpoint validation, cumulative/local step accounting, gate progression, and previous-stage evaluation.
- [ ] Run focused tests and optionally one J1 run of no more than five steps.

### Task 6: Safe external execution scripts

**Files:**
- Create: `scripts/run_o6_random_clean_external.sh`
- Create: `scripts/run_o6_extension_external.sh`
- Create: `scripts/run_stage1_jammer_curriculum_external.sh`
- Create: `scripts/evaluate_stage1_curriculum_stage.sh`
- Create: `tests/test_stage1_external_scripts.py`

**Interfaces:**
- Consumes: the common diagnostic CLI and stage summaries.
- Produces: guarded external commands with logs, provenance, resume, device, overwrite, and dry-run controls.

- [ ] Write failing structural and behavior tests for `set -euo pipefail`, overwrite refusal, `tee`, stage stopping, resume, and dry-run.
- [ ] Run focused tests and confirm RED.
- [ ] Implement scripts with O6 1000-step and conditional 3000-step commands plus J1–J5 gated curriculum execution.
- [ ] Run every script with `--dry-run`; do not execute long commands.

### Task 7: Readiness evaluator, final command preparation, and progress report

**Files:**
- Create: `src/speech_jscc/diagnostics/stage1_readiness.py`
- Create: `evaluate_stage1_readiness.py`
- Create: `scripts/prepare_uniform_stage1_full_external.sh`
- Create: `tests/test_stage1_readiness_gate.py`

**Interfaces:**
- Produces read-only readiness JSON/Markdown and the consolidated distribution diagnostic report.
- Prints a full Uniform command only when every artifact, hash, lineage, metadata, and gate check passes.

- [ ] Write failing tests for false readiness, synthetic true readiness, stale hash/lineage rejection, legacy checkpoint rejection, and command hiding.
- [ ] Run focused tests and confirm RED.
- [ ] Implement strict read-only artifact validation and progress reporting that never marks unexecuted stages complete.
- [ ] Implement the guarded final command preparer with explicit `--execute-ready-training` requirement.
- [ ] Run focused tests and readiness evaluation against current artifacts; expect readiness false until external O6/J runs exist.

### Task 8: Integration verification

**Files:**
- Modify only files implicated by test failures.

- [ ] Run the six requested focused test files together.
- [ ] Run existing O5, Stage-1, channel, pilot, gradient, and dataflow tests.
- [ ] Run `pytest tests/ -q` and record exact results.
- [ ] Regenerate O5 reports and run protocol step-0 audit.
- [ ] Run all external scripts with `--dry-run` and record commands/results.
- [ ] Inspect `git diff`/status to confirm unrelated untracked files are untouched and no long optimization occurred.
