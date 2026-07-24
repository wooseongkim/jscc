# FDD Temporal CSI Evaluation Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an evaluation-only, leakage-safe FDD engine comparing iid, correlated fixed-interleaved, delayed-CSI, and oracle-current allocation under current-slot MMSE reception.

**Architecture:** Add a standalone correlated-tap channel module and a diagnostic module owning the feedback timeline, deterministic interleaver, stratified allocator, and metrics. A dedicated CLI loads the accepted waveform-aware checkpoint and runs paired 64-slot trajectories without touching production G/J paths.

**Tech Stack:** Python 3.11, PyTorch, SciPy `J0`, pytest, YAML.

## Global Constraints

- Defaults: 3.5 GHz, 3 m/s, 1 ms slot, one-slot unquantized feedback.
- Taps follow Jakes-derived Gauss–Markov evolution and preserve the configured PDP.
- Slot 0 always uses uniform fixed-interleaver bootstrap.
- T2 uses only the prior receiver report; T3 oracle CSI affects allocation only.
- RX uses current estimated CSI and MMSE in T1/T2/T3.
- Preserve 240 symbols per layer, 1,920 total, pilot_reserved_v1, and CF-2 balanced-ragged packing.
- Layer order `[1,0,2,5,3,4,6,7]` is smoke-derived, provisional, and not accepted scientific evidence.

---

### Task 1: Temporal multipath channel

**Files:** Create `src/channels/temporal_multipath.py`; test `tests/test_fdd_temporal_csi.py`.

- [ ] Write failing tests for Jakes rho, lag-one correlation, iid independence, PDP normalization, and FFT response identity.
- [ ] Run the tests and confirm missing interfaces fail.
- [ ] Implement deterministic tap trajectory generation.
- [ ] Re-run and confirm the temporal tests pass.

### Task 2: FDD timeline and mapping

**Files:** Create `src/speech_jscc/diagnostics/fdd_temporal_csi.py`; modify `tests/test_fdd_temporal_csi.py`.

- [ ] Write failing tests for one-slot delay, bootstrap, leakage prevention, future-channel invariance, report sensitivity, interleaver inversion, allocation bijection, pilot separation, and layer counts.
- [ ] Implement immutable reports, feedback buffer, fixed interleaver, and stratified delayed-CSI allocation.
- [ ] Re-run focused tests.

### Task 3: Paired T0–T3 evaluator

**Files:** Create `configs/fdd_temporal_csi.yaml`, `evaluate_fdd_temporal_csi.py`, `scripts/run_fdd_temporal_csi_external.sh`, and `runs/mmse_csi_interleaving/temporal_channel_audit.md`.

- [ ] Add CLI dry-run and paired-seed tests.
- [ ] Implement 64-slot deterministic permutations, shared T1–T3 channel/noise/symbol tensors, current-RX MMSE, T2 delayed allocation, T3 allocation-only oracle, and including/excluding-slot-0 reports.
- [ ] Run one short CPU trajectory smoke and all script dry runs.

### Task 4: Verification

- [ ] Run focused FDD, channel-free, ideal-OFDM, pilot, and dataflow tests.
- [ ] Run `pytest tests/ -q`.
- [ ] Report only measured smoke evidence locally and provide the exact external CUDA command for the full evaluation.
