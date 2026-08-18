# R4 Jammer-Aware SINR RE Allocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the new jammer-aware path's alpha-discounted triplet placement with a fixed-UEP, SINR-utility RE allocator that gives each repeated source symbol one RE per subband.

**Architecture:** Keep `UEPProfile` and the physical/MRC path unchanged. Add a variable-copy allocation builder that consumes a delayed LS-CSI report and a delayed receiver-estimated interference-power report, calculates `a_l |H|^2/(N+I)`, and greedily maximizes the MRC log-utility marginal gain. R4WaveformForward selects this allocator only for the new explicit allocation mode; legacy and alpha-risk paths remain regression-compatible.

**Tech Stack:** Python, PyTorch, existing R4 OFDM/LS/MRC evaluator, pytest.

## Global Constraints

- Keep `UEPProfile`, repetition vector `r`, total power share `p`, layer importance order `pi`, OFDM physical path, and MRC unchanged.
- Do not use a true jammer mask/tensor in deployable allocation mode; oracle interference is an explicit upper-bound mode only.
- Use only receiver reports generated before the transmitted TTI.
- Every source symbol gets exactly `r_l` copies; each selected RE is used once; total selected RE is exactly 5,760.
- Partition candidate REs into `max(r)` contiguous frequency subbands and place at most one copy of a given source symbol in each subband; do not use copy anchors or fixed offsets.
- Preserve no-risk legacy behavior when the new allocation mode is not selected.

---

### Task 1: Define interference report and SINR allocation API

**Files:**
- Modify: `src/channels/re_risk.py`
- Modify: `src/channels/__init__.py`
- Test: `tests/test_r4_jammer_aware_allocation.py`

**Interfaces:**
- Produces `REInterferenceReport(generated_tti, available_tti, interference_power)`.
- Produces `estimate_rx_residual_interference_power(...) -> Tensor[candidate_data_re]`.
- Produces `oracle_jamming_interference_report(...) -> REInterferenceReport` for upper-bound-only evaluation.

- [ ] **Step 1: Write failing tests**

```python
def test_residual_interference_estimator_has_no_true_jammer_input():
    params = inspect.signature(estimate_rx_residual_interference_power).parameters
    assert {"jammer_mask", "jammer_tensor", "jammer_type"}.isdisjoint(params)

def test_delayed_interference_report_rejects_current_tti():
    with pytest.raises(ValueError, match="causally available"):
        require_available_interference_report(tx_tti=2, report=REInterferenceReport.from_power(2, torch.ones(9720)))
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/test_r4_jammer_aware_allocation.py -k interference`

Expected: import failure because the report and estimator do not yet exist.

- [ ] **Step 3: Implement the minimal report/estimator API**

```python
@dataclass(frozen=True)
class REInterferenceReport:
    generated_tti: int
    available_tti: int
    interference_power: Tensor

def estimate_rx_residual_interference_power(...):
    # pilot residual interpolation + positive local-energy residual;
    # no inverse-channel-gain term and no jammer labels.
```

- [ ] **Step 4: Run the tests and verify GREEN**

Run: `pytest -q tests/test_r4_jammer_aware_allocation.py -k interference`

Expected: PASS.

### Task 2: Build fixed-UEP SINR marginal-utility allocator

**Files:**
- Create: `src/channels/r4_jammer_aware_allocator.py`
- Test: `tests/test_r4_jammer_aware_allocation.py`

**Interfaces:**
- Consumes `PhysicalOFDMProfile`, `GlobalTripletCSIReport`, `REInterferenceReport`, `UEPProfile`, and `layer_importance_order`.
- Produces `JammerAwareVariableCopyAllocation`, compatible with `r4_physical_layer_forward` via `place`, `extract_source_order`, `source_to_candidate_indices`, and `power_source_order`.
- Main API: `allocate_r4_jammer_aware_sinr(...)`.

- [ ] **Step 1: Write failing tests**

```python
def test_sinr_allocator_enforces_copy_counts_unique_res_and_subband_diversity():
    allocation = allocate_r4_jammer_aware_sinr(..., uep_profile=XBEST)
    assert allocation.selected_re_count == 5760
    assert torch.equal(allocation.copy_count_per_source, expected_per_source_counts)
    assert torch.unique(allocation.selected_candidate_indices[allocation.selected_candidate_indices >= 0]).numel() == 5760
    assert allocation.distinct_subband_per_source

def test_higher_interference_lowers_assignment_sinr():
    assert allocation.assignment_sinr[low_interference_re] > allocation.assignment_sinr[high_interference_re]
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/test_r4_jammer_aware_allocation.py -k 'sinr_allocator or copy_counts or interference_lowers'`

Expected: import failure because the allocator does not yet exist.

- [ ] **Step 3: Implement the allocator**

```python
gamma[layer, re] = per_re_power[layer] * csi_gain[re] / (noise[re] + interference[re] + eps)
marginal = weight[layer] * log((1 + combined_sinr[source] + gamma[layer, re]) / (1 + combined_sinr[source]))
```

Select the maximum feasible marginal pair repeatedly, then run deterministic improving single-RE exchanges. Candidate feasibility enforces free RE, exact copy count, and distinct-subband-per-source.

- [ ] **Step 4: Run the tests and verify GREEN**

Run: `pytest -q tests/test_r4_jammer_aware_allocation.py -k 'sinr_allocator or copy_counts or interference_lowers'`

Expected: PASS.

### Task 3: Connect the R4 forward path and evaluator configuration

**Files:**
- Modify: `src/speech_jscc/training/r4_waveform_finetune.py`
- Modify: `configs/eval_r4_jammer_refiner_fixed.yaml`
- Modify: `src/speech_jscc/evaluation/evaluate_r4_jammer_refiner_checkpoints.py`
- Test: `tests/test_r4_jammer_aware_allocation.py`

**Interfaces:**
- Adds allocation mode `jammer_aware_sinr` with `delayed_rx_interference` and `oracle_jamming_interference` evidence modes.
- Adds optional 5/10 dB jammer comparison conditions without changing UEP r/p.

- [ ] **Step 1: Write failing forward tests**

```python
def test_jammer_aware_sinr_forward_uses_delayed_reports_and_preserves_uep_profile():
    first = engine.forward(..., tti=0)
    second = engine.forward(..., tti=1, delayed_csi=first.next_delayed_csi, delayed_interference=first.next_re_interference)
    assert second.allocation.uep_profile == engine.uep_profile

def test_current_tti_oracle_mask_is_rejected_in_deployable_mode():
    with pytest.raises(ValueError):
        engine.forward(..., delayed_interference=current_tti_report)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/test_r4_jammer_aware_allocation.py -k forward`

Expected: missing forward API/config mode.

- [ ] **Step 3: Implement mode selection and artifact fields**

Add mode-specific allocation metadata: delayed report TTI, SINR summary, subband count, exact copy counts, selected RE count, and oracle/deployable flag. Leave all existing allocation modes unchanged.

- [ ] **Step 4: Run the tests and verify GREEN**

Run: `pytest -q tests/test_r4_jammer_aware_allocation.py`

Expected: PASS.

### Task 4: Regression and CPU smoke

**Files:**
- Test: `tests/test_r4_waveform_finetune.py`
- Test: `tests/test_re_risk.py`

- [ ] **Step 1: Run focused regression tests**

Run: `pytest -q tests/test_r4_jammer_aware_allocation.py tests/test_re_risk.py tests/test_repetition_mrc.py tests/test_r4_waveform_finetune.py`

Expected: PASS.

- [ ] **Step 2: Run a CPU two-condition forward smoke**

Run: `python evaluate_r4_jammer_refiner_checkpoints.py ... --allocation-mode jammer_aware_sinr --max-utterances 2 --max-realizations 1 --device cpu`

Expected: finite rows, exactly 5,760 selected REs, delayed-report provenance, and no duplicate RE assignments.

