# R4 Broadband UEP Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, black-box optimizer that searches valid layer-wise repetition and total-power allocations on `expanded_selection`, then evaluates only frozen selected profiles on `legacy_final`.

**Architecture:** A pure optimizer module owns feasibility, sampling, objective, Pareto filtering, and profile serialization.  The command-line layer is deliberately thin: it delegates waveform evaluation to the existing broadband-UEP evaluator/forward path and never uses legacy data during selection.  Generic allocation support carries an arbitrary `(r, p)` profile through the existing variable-copy placement and MRC path.

**Tech Stack:** Python 3.11, PyTorch, pytest, existing R4 OFDM/`R4WaveformForward` evaluator stack.

## Global Constraints

- Search uses `expanded_selection` only; `legacy_final` is accepted only with `--run-final-eval` and a frozen `selected_profiles.json`.
- Candidate repetition is integer `r_i in {1,2,3,4,5}` with `sum(r)=24`; each layer has 240 symbols and total data RE remains 5,760.
- Power uses logits `z`, `p=softmax(z)`, and `a_i=24*p_i/r_i`; total packet energy remains 5,760.
- Do not use layer-ablation scores, P2, RP2, or a fixed layer-priority vector in candidate generation/refinement.
- Reuse the existing waveform evaluator/forward, canonical broadband-jammer replay, official full-crop unaligned SI-SDR, and MRC implementation; do not introduce a surrogate metric.
- The same cached channel/AWGN/jammer tensors are reused for U0 and every candidate in a selection condition.
- CUDA is user-executed only. CPU checks prove structure; a CUDA smoke result is required before claiming runtime validation.
- Preserve all existing source/checkpoint/model/metric/mapping settings and do not modify legacy selection artifacts.

---

### Task 1: Test-driven optimizer core

**Files:**
- Create: `src/speech_jscc/evaluation/r4_broadband_uep_optimizer.py`
- Create: `tests/test_r4_broadband_uep_optimizer.py`

**Interfaces:**
- Produces `enumerate_feasible_repetitions()`, `validate_repetition()`, `softmax_power()`, `per_re_power()`, `make_candidate()`, `sample_random_candidates()`, `propose_repetition_moves()`, `propose_power_transfers()`, `objective_from_summary()`, `pareto_front()`, and `select_profiles()`.
- Consumes numeric summary dictionaries keyed by paired U0 metrics; it has no waveform-forward or legacy-manifest import.

- [ ] **Step 1: Write failing feasibility and power tests**

```python
def test_uniform_candidate_has_uniform_per_re_power():
    candidate = make_candidate((3,) * 8, [0.0] * 8)
    assert candidate.power_share == pytest.approx((0.125,) * 8)
    assert candidate.per_re_power == pytest.approx((1.0,) * 8)
    assert total_re(candidate.repetition) == 5760
    assert total_energy(candidate.repetition, candidate.per_re_power) == pytest.approx(5760)
```

- [ ] **Step 2: Run the core test to verify it fails**

Run: `pytest -q tests/test_r4_broadband_uep_optimizer.py`

Expected: import failure because the optimizer module is absent.

- [ ] **Step 3: Implement minimal deterministic optimizer primitives**

```python
def per_re_power(r: Sequence[int], p: Sequence[float]) -> tuple[float, ...]:
    return tuple(24.0 * share / copies for copies, share in zip(r, p))
```

Implement feasible-vector generation by recursion, Dirichlet sampling without a priority vector, canonical candidate hashing, objective penalties based only on paired U0 results, Pareto dominance, and the three profile selectors.

- [ ] **Step 4: Run core tests and refactor only after green**

Run: `pytest -q tests/test_r4_broadband_uep_optimizer.py`

Expected: all optimizer-core cases pass.

### Task 2: Generic allocation profile support

**Files:**
- Modify: `src/channels/r4_uep_allocator.py`
- Modify: `src/speech_jscc/training/r4_waveform_finetune.py`
- Modify: `evaluate_r4_expanded_validation.py`
- Test: `tests/test_r4_broadband_uep_optimizer.py`

**Interfaces:**
- Consumes serialized candidate profiles with `profile_id`, `repetition`, `power_share`, and `per_re_power`.
- Produces a `VariableCopyAllocation` with exact copy counts, exactly 5,760 selected REs, and effective-channel power scaling that agrees with transmitted `sqrt(a_i)`.

- [ ] **Step 1: Add failing generic-profile allocation tests**

```python
def test_generic_profile_keeps_copy_count_re_and_energy():
    profile = generic_profile("candidate", (5, 4, 3, 3, 3, 2, 2, 2), (0.2,) * 5)
    allocation = allocate_r4_uep(..., profile=profile)
    assert allocation.copy_count_by_layer() == profile.repetition
    assert allocation.selected_re_count == 5760
    assert allocation.total_transmit_energy == pytest.approx(5760)
```

- [ ] **Step 2: Run the allocation test to verify it fails**

Run: `pytest -q tests/test_r4_broadband_uep_optimizer.py -k generic_profile`

- [ ] **Step 3: Extend allocation and forward plumbing minimally**

Generalize the existing profile representation to accept `r=1..5` and explicit total-power shares.  Preserve exact U0 delegation to the historical global balanced-triplet allocator; map all non-U0 candidate source symbols with `sqrt(a_i)` at transmit placement and the same factor in MRC effective channel.

- [ ] **Step 4: Run optimizer and UEP profile tests**

Run: `pytest -q tests/test_r4_broadband_uep_optimizer.py tests/test_r4_broadband_uep_profiles.py`

### Task 3: Selection-search orchestration and artifacts

**Files:**
- Create: `optimize_r4_broadband_uep.py`
- Create: `configs/optimize_r4_broadband_uep.yaml`
- Modify: `evaluate_r4_broadband_uep_profiles.py`
- Test: `tests/test_r4_broadband_uep_optimizer.py`

**Interfaces:**
- Selection command accepts `--selection-split expanded_selection`; it writes `selected_profiles.json` including `x_best`, `x_stable`, `x_aggressive`, source checkpoint metadata, hashes, objective summaries, and no legacy path.
- Evaluation backend accepts arbitrary serialized profiles and a prebuilt condition cache; it returns realization rows and aggregate candidate summaries.

- [ ] **Step 1: Add failing no-legacy-leakage and selection-schema tests**

```python
def test_selection_rejects_legacy_split_and_serialized_selection_has_no_legacy_paths(tmp_path):
    with pytest.raises(ValueError, match="expanded_selection"):
        validate_search_split("legacy_final")
    artifact = build_selected_profiles_artifact(...)
    assert artifact["selection_split"] == "expanded_selection"
    assert "legacy" not in json.dumps(artifact).lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest -q tests/test_r4_broadband_uep_optimizer.py -k 'legacy or selected_profiles'`

- [ ] **Step 3: Implement random stage, coordinate refinement, cached paired evaluation, and artifact writing**

Build U0 once on `expanded_selection`; cache every condition’s codec input, channel/AWGN report, and broadband jammer tensor.  Pass those exact cached conditions to candidates.  Use Dirichlet alphas `.5, 1, 2, 5`, top-K local refinement, deterministic duplicate elimination, scalar ranking, and Pareto filtering.  Serialize all candidate and objective components.

- [ ] **Step 4: Run selection/orchestration tests and CPU dry run**

Run: `pytest -q tests/test_r4_broadband_uep_optimizer.py`

Run: `python optimize_r4_broadband_uep.py --config configs/optimize_r4_broadband_uep.yaml --dry-run`

### Task 4: Frozen final-evaluation mode

**Files:**
- Modify: `optimize_r4_broadband_uep.py`
- Test: `tests/test_r4_broadband_uep_optimizer.py`

**Interfaces:**
- `--run-final-eval --selected-profiles PATH --final-split legacy_final` loads only frozen serialized candidates and does not call search/ranking code.

- [ ] **Step 1: Add a failing frozen-input test**

```python
def test_final_eval_requires_frozen_selection_and_cannot_modify_it(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_selected_profiles(tmp_path / "missing.json")
    before = selected_path.read_bytes()
    run_final_eval(...)
    assert selected_path.read_bytes() == before
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest -q tests/test_r4_broadband_uep_optimizer.py -k final_eval`

- [ ] **Step 3: Implement final-eval input validation and backend invocation**

Require an optimizer selection artifact, verify schema/selection split/checkpoint hash, write final artifacts to a separate output directory, and never alter `selected_profiles.json`.

- [ ] **Step 4: Run focused regressions**

Run: `pytest -q tests/test_r4_broadband_uep_optimizer.py tests/test_r4_broadband_uep_profiles.py tests/test_r4_jammer_baseline.py tests/test_repetition_mrc.py tests/test_si_sdr_loss.py`

### Task 5: CUDA handoff and verification

**Files:**
- Modify: `docs/superpowers/plans/2026-08-07-r4-broadband-uep-optimizer.md`

- [ ] **Step 1: Run CPU dry-run and focused tests**

Record actual outputs only; do not make CUDA calls from Codex.

- [ ] **Step 2: Provide user-executed CUDA smoke command**

```bash
python optimize_r4_broadband_uep.py \
  --config configs/optimize_r4_broadband_uep.yaml \
  --checkpoint runs/waveform_aware_wireless/r4_si_sdr_finetune/si_sdr_medium/local_step_003000.pt \
  --selection-split expanded_selection --snr-db 5 --jsr-db no_jammer 0 5 \
  --max-utterances 4 --max-realizations 1 --max-candidates-stage1 16 \
  --top-k-for-refine 4 --local-refine-steps 8 --device cuda \
  --output-dir runs/waveform_aware_wireless/r4_broadband_uep_optimization/smoke --overwrite
```

- [ ] **Step 3: Inspect supplied CUDA smoke artifacts before completion claims**

Require finite rows, an internally paired U0 cache, candidates evaluated through the real forward, and a populated Pareto/selection artifact.

## Self-review

- Search-only and final-only boundaries are separate tasks and use explicit split validation.
- All candidate mathematics, feasibility, hashing, objective, and Pareto rules have test-first coverage.
- No task feeds hand-crafted layer importance or P2/RP2 into proposals.
- The backend is reused rather than a surrogate score; CUDA validation remains user-operated.
