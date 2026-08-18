# R4 Jammer Estimator and MoE Latent Refiner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add receiver-only jammer posterior/mask estimation and a posterior-gated MoE residual latent refiner to R4 while preserving the existing physical transmission path by default.

**Architecture:** A `JammerEstimator` consumes only physical receiver observations and returns a posterior plus a soft physical-grid mask. A zero-initialized MoE refiner consumes the raw JSCC decoder latent, observable decoder state, and either oracle or estimated mask; `R4WaveformForward` selects this post-decoder path only for explicit refiner modes.

**Tech Stack:** Python 3.11, PyTorch, pytest, existing R4WaveformForward / physical OFDM path.

## Global Constraints

- Posterior classes: `no_jammer`, `broadband_awgn`, `subband`, `burst`, `block`, `tone`.
- True jammer type/mask are loss labels only and are never estimator forward inputs.
- No modification to `allocate_global_balanced_triplets`, `allocate_r4_uep`, `UEPProfile`, resource allocation, or MRC.
- The default mode is `no_refiner` and must preserve the existing reconstruction result exactly.
- Refiner processing occurs only after the JSCC decoder returns raw latent reconstruction.

---

### Task 1: Jammer estimator contract and leakage-safe loss

**Files:**
- Create: `src/models/jammer_estimator.py`
- Create: `tests/test_jammer_estimator.py`

**Interfaces:**
- Produces `JAMMER_TYPE_CLASSES`, `JammerEstimate`, `JammerEstimator`, and `jammer_estimation_loss`.
- `JammerEstimator.forward(received_grid, pilots, pilot_mask, estimated_channel, noise_variance)` has no true-label arguments.

- [ ] **Step 1: Write failing shape and leakage tests**

```python
def test_estimator_returns_grid_mask_and_posterior():
    estimate = estimator(received_grid, pilots, pilot_mask, channel, noise)
    assert estimate.posterior.shape == (2, 6)
    assert estimate.mask_logits.shape == received_grid.shape
    assert torch.allclose(estimate.posterior.sum(-1), torch.ones(2))

def test_estimator_forward_does_not_accept_true_mask_or_type():
    with pytest.raises(TypeError):
        estimator(..., true_jammer_mask=mask)
```

- [ ] **Step 2: Run the test to verify RED**

Run: `pytest -q tests/test_jammer_estimator.py`

- [ ] **Step 3: Implement the minimal observable estimator and loss**

```python
@dataclass
class JammerEstimate:
    posterior: Tensor
    mask_logits: Tensor
    mask_prob: Tensor
    mask_ratio: Tensor

def jammer_estimation_loss(estimate, *, jammer_type, jammer_mask):
    return cross_entropy + bce + dice
```

- [ ] **Step 4: Run the estimator tests to verify GREEN**

Run: `pytest -q tests/test_jammer_estimator.py`

### Task 2: Posterior-gated MoE residual refiner

**Files:**
- Create: `src/models/adaptive_latent_refiner.py`
- Create: `tests/test_adaptive_latent_refiner.py`

**Interfaces:**
- Produces `MoEAdaptiveLatentRefiner` with `forward(raw_latent, decoder_state, mask_prob, jammer_posterior)`.
- Output has shape `[B,L,T,D]` and residual heads are zero-initialized.

- [ ] **Step 1: Write failing MoE shape and identity tests**

```python
def test_moe_refiner_returns_raw_latent_at_initialization():
    output = refiner(raw, state, mask, posterior)
    assert output.shape == raw.shape
    assert torch.allclose(output, raw, atol=1e-7)
```

- [ ] **Step 2: Run the test to verify RED**

Run: `pytest -q tests/test_adaptive_latent_refiner.py`

- [ ] **Step 3: Implement expert residual networks and posterior mixture**

```python
delta_by_expert = torch.stack([expert(features) for expert in experts], dim=1)
delta = (posterior[..., None, None, None] * delta_by_expert).sum(dim=1)
return raw_latent + delta
```

- [ ] **Step 4: Run the MoE tests to verify GREEN**

Run: `pytest -q tests/test_adaptive_latent_refiner.py`

### Task 3: R4 post-decoder integration and ablation modes

**Files:**
- Modify: `src/speech_jscc/training/r4_waveform_finetune.py`
- Create: `tests/test_r4_jammer_refiner.py`

**Interfaces:**
- `R4WaveformForward(..., jammer_estimator=None, adaptive_refiner=None, refiner_mode="no_refiner")`.
- `R4WaveformOutput` gains `raw_reconstruction`, `jammer_posterior`, `jammer_mask_prob`, `refiner_mode`.

- [ ] **Step 1: Write failing R4 no-refiner regression and mode smoke tests**

```python
def test_no_refiner_preserves_raw_decoder_output(r4_forward, condition):
    output = r4_forward.forward(representation, channel_condition=condition)
    assert torch.equal(output.reconstruction, output.raw_reconstruction)

def test_oracle_and_learned_mask_refiner_modes_produce_finite_output(...):
    assert torch.isfinite(output.reconstruction).all()
```

- [ ] **Step 2: Run the test to verify RED**

Run: `pytest -q tests/test_r4_jammer_refiner.py`

- [ ] **Step 3: Insert mode dispatch immediately after `model.decoder`**

```python
raw_reconstruction = self.model.decoder(physical.combined.estimate, physical.decoder_state)
reconstruction, estimate = self._apply_refiner(raw_reconstruction, physical)
```

- [ ] **Step 4: Run R4 integration tests to verify GREEN**

Run: `pytest -q tests/test_r4_jammer_refiner.py tests/test_r4_waveform_finetune.py`

### Task 4: Checkpoint/config plumbing and focused verification

**Files:**
- Modify: relevant evaluator/training CLI configuration loaders only where an existing checkpoint option is parsed.
- Test: `tests/test_jammer_estimator.py`, `tests/test_adaptive_latent_refiner.py`, `tests/test_r4_jammer_refiner.py`

**Interfaces:**
- Explicit refiner mode and optional estimator/refiner checkpoint paths.
- Learned modes fail clearly when dependencies are missing; `no_refiner` requires no new state.

- [ ] **Step 1: Write failing checkpoint dependency test**

```python
def test_learned_mode_requires_estimator_and_refiner():
    with pytest.raises(ValueError, match="jammer_estimator"):
        R4WaveformForward(codec, model, refiner_mode="learned_mask_refiner")
```

- [ ] **Step 2: Run it to verify RED**

Run: `pytest -q tests/test_r4_jammer_refiner.py::test_learned_mode_requires_estimator_and_refiner`

- [ ] **Step 3: Add validation and checkpoint loading helpers**

```python
if refiner_mode in LEARNED_REFINER_MODES and jammer_estimator is None:
    raise ValueError("learned refiner mode requires jammer_estimator")
```

- [ ] **Step 4: Run focused tests and no-refiner regression**

Run: `pytest -q tests/test_jammer_estimator.py tests/test_adaptive_latent_refiner.py tests/test_r4_jammer_refiner.py tests/test_r4_waveform_finetune.py`

## Self-review

- Estimator labels are absent from the estimator input contract.
- MoE is post-decoder only and uses posterior mixture weights.
- `no_refiner` is default and preserves old reconstruction.
- Oracle-mask use is explicitly confined to the named oracle ablation.
- UEP allocator, mapping, and MRC files are not modified.
