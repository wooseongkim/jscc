# R4 Jammer Estimator and MoE Refiner Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a standalone supervised trainer that learns the receiver-only jammer estimator and post-decoder MoE latent refiner on the unmodified R4 physical path.

**Architecture:** The trainer restores the fixed medium JSCC checkpoint, freezes codec/JSCC by default, and repeatedly evaluates R4 physical conditions with five deterministic jammer types. It owns estimator/refiner modules, phase-specific optimizer selection, losses, validation, and independent checkpoints; the existing SI-SDR trainer is never imported or modified.

**Tech Stack:** Python 3.11, PyTorch, YAML, pytest, existing R4 physical forward.

## Global Constraints

- Physical jammer vocabulary: `no_jammer`, `broadband_awgn`, `subband`, `burst`, `tone`; config alias `narrowband` resolves to `subband`.
- True jammer labels/masks are loss targets only; they never enter `JammerEstimator.forward`.
- Do not modify global-triplet allocation, UEP, repetition/MRC, or existing SI-SDR trainer behavior.
- Codec is always frozen/eval; JSCC encoder always frozen; decoder unfreezing is explicit config opt-in.
- Default refiner behavior remains `no_refiner` outside the new trainer.

---

### Task 1: Trainer utility contracts and taxonomy tests

**Files:**
- Create: `src/speech_jscc/training/train_r4_jammer_refiner.py`
- Create: `tests/test_train_r4_jammer_refiner_smoke.py`

**Interfaces:**
- `canonical_jammer_type(name: str) -> str`
- `sample_jammer_condition(config, step, seed) -> dict`
- `build_phase_schedule(config) -> tuple[PhaseSpec, ...]`

- [ ] **Step 1: Write failing taxonomy and phase tests**

```python
def test_narrowband_alias_is_subband():
    assert canonical_jammer_type("narrowband") == "subband"

def test_three_phase_schedule_is_contiguous():
    phases = build_phase_schedule({"phases": ...})
    assert [phase.name for phase in phases] == ["estimator_pretrain", "oracle_mask_refiner", "learned_mask_moe"]
```

- [ ] **Step 2: Run RED tests**

Run: `pytest -q tests/test_train_r4_jammer_refiner_smoke.py`

- [ ] **Step 3: Implement pure config validation/sampling utilities**

```python
JAMMER_TYPE_TO_INDEX = {name: index for index, name in enumerate(JAMMER_TYPE_CLASSES)}
def canonical_jammer_type(name):
    return "subband" if name == "narrowband" else name
```

- [ ] **Step 4: Run GREEN tests**

Run: `pytest -q tests/test_train_r4_jammer_refiner_smoke.py`

### Task 2: Phase losses and optimizer isolation

**Files:**
- Modify: `src/speech_jscc/training/train_r4_jammer_refiner.py`
- Test: `tests/test_train_r4_jammer_refiner_smoke.py`

**Interfaces:**
- `build_refiner_optimizer(estimator, refiner, model, config, phase)`
- `compute_refiner_losses(...) -> (total, components)`

- [ ] **Step 1: Write failing optimizer and label-leakage tests**

```python
def test_default_optimizer_excludes_jscc_parameters(...):
    assert optimizer_ids.isdisjoint(jscc_ids)

def test_oracle_phase_uses_label_only_after_estimator_forward(...):
    assert physical_mask_used_by_refiner
```

- [ ] **Step 2: Run RED tests**

Run: `pytest -q tests/test_train_r4_jammer_refiner_smoke.py`

- [ ] **Step 3: Implement CE/BCE/Dice/latent/STFT/SI-SDR/identity loss composition**

```python
total = sum(weights[name] * component for name, component in components.items())
```

- [ ] **Step 4: Run GREEN tests**

Run: `pytest -q tests/test_train_r4_jammer_refiner_smoke.py`

### Task 3: CLI training loop, checkpointing, and fixed validation

**Files:**
- Create: `configs/train_r4_jammer_refiner.yaml`
- Modify: `src/speech_jscc/training/train_r4_jammer_refiner.py`
- Create: `train_r4_jammer_refiner.py`
- Test: `tests/test_train_r4_jammer_refiner_smoke.py`

**Interfaces:**
- CLI supports config/checkpoint/device/output/max steps/seed/long-run guard.
- `last.pt`, `best_validation_si_sdr.pt`, `best_validation_latent.pt` store estimator/refiner, optimizer, phase, label vocabulary, config, and source checkpoint metadata.

- [ ] **Step 1: Write failing two-step CPU subprocess smoke test**

```python
def test_cli_two_step_cpu_smoke_writes_reloadable_last_checkpoint(tmp_path):
    result = subprocess.run([... "--max-steps", "2", "--device", "cpu"], check=True)
    payload = torch.load(tmp_path / "last.pt", weights_only=False)
    assert {"estimator", "adaptive_refiner", "optimizer", "jammer_label_vocabulary"} <= payload.keys()
```

- [ ] **Step 2: Run RED smoke test**

Run: `pytest -q tests/test_train_r4_jammer_refiner_smoke.py`

- [ ] **Step 3: Implement output isolation, long-run guard, phase loop, validation rows, and checkpoint selection**

```python
torch.save(payload, output / "last.pt")
write_jsonl(output / "validation_metrics.jsonl", validation_rows)
```

- [ ] **Step 4: Run CPU GREEN smoke and CLI help**

Run: `pytest -q tests/test_train_r4_jammer_refiner_smoke.py && python train_r4_jammer_refiner.py --help`

### Task 4: Compatibility verification

**Files:**
- Test: `tests/test_jammer_estimator.py`, `tests/test_adaptive_latent_refiner.py`, `tests/test_r4_jammer_refiner.py`, `tests/test_r4_waveform_finetune.py`

- [ ] **Step 1: Run existing no-refiner and R4 regression tests**

Run: `pytest -q tests/test_jammer_estimator.py tests/test_adaptive_latent_refiner.py tests/test_r4_jammer_refiner.py tests/test_r4_waveform_finetune.py`

- [ ] **Step 2: Run trainer smoke and syntax verification**

Run: `python -m compileall -q src/speech_jscc/training/train_r4_jammer_refiner.py train_r4_jammer_refiner.py && pytest -q tests/test_train_r4_jammer_refiner_smoke.py`

## Self-review

- The only allowed physical forward is `R4WaveformForward`; no allocator/MRC code is changed.
- Validation uses fixed seeds and does not affect optimizer or selection state.
- Oracle labels are never estimator forward inputs.
- Existing trainer config and CLI are untouched.
