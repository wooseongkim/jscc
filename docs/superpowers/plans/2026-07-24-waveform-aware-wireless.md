# Waveform-Aware Wireless Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the accepted CF-2 balanced-ragged Conv-Conformer checkpoint to the existing pilot-reserved wireless path, prove ideal-OFDM identity, evaluate clean channels, and expose conditional waveform-aware fine-tuning.

**Architecture:** Add one focused library module for CF-2 preflight, ideal mapping, clean-channel condition construction, waveform gates, and report aggregation. Add separate evaluation and training entry points so no optimizer is constructed during zero-shot evaluation; use existing pilot, allocation, multipath, estimator, equalizer, codec, metric, and checkpoint utilities.

**Tech Stack:** Python 3.11, PyTorch, pytest, YAML, existing Speech JSCC modules.

## Global Constraints

- The source checkpoint is `runs/channel_free_revalidation/cf2_50frames_1920/best_waveform_si_sdr.pt`.
- Preserve 50 symbol frames, balanced-ragged packing, 240 valid complex symbols per layer, and 1,920 total.
- Ideal OFDM uses grid `[64,32]`, 128 pilots, 1,920 data resources, identity channel, no noise, and exact CSI.
- Do not train unless random clean-channel zero-shot fails waveform feasibility.
- Do not modify or overwrite accepted CF-2 or jammer artifacts.

---

### Task 1: Preflight and condition primitives

**Files:**
- Create: `tests/test_waveform_aware_wireless.py`
- Create: `src/speech_jscc/diagnostics/waveform_aware_wireless.py`

**Interfaces:**
- Produces: `cf2_preflight(model, config, device) -> dict`
- Produces: `ideal_ofdm_round_trip(symbols, model, config) -> dict`
- Produces: `wireless_feasibility_gate(metrics, thresholds) -> dict`
- Produces: `clean_validation_conditions(seed, realizations_per_utterance) -> list[dict]`

- [ ] Write failing tests for exact CF-2 mask/counts, masked zeros/power invariance, mapping identity, pilot counts/disjointness, deterministic SNR slices, and waveform gates.
- [ ] Run `pytest tests/test_waveform_aware_wireless.py -q` and confirm missing-module failure.
- [ ] Implement the minimal primitives using existing balanced-ragged, pilot, and resource-allocation functions.
- [ ] Re-run the focused test and confirm all assertions pass.

### Task 2: Zero-shot waveform evaluation

**Files:**
- Create: `eval_waveform_aware_wireless.py`
- Create: `configs/waveform_aware_wireless.yaml`
- Modify: `tests/test_waveform_aware_wireless.py`

**Interfaces:**
- Consumes: Task 1 primitives and existing CF-2 validation manifest/checkpoint utilities.
- Produces: modes `preflight`, `ideal_ofdm`, and `clean_zero_shot`; JSON/CSV/WAV artifacts under `runs/waveform_aware_wireless`.

- [ ] Add failing CLI/dry-run and ideal-stop-policy tests.
- [ ] Run focused tests and confirm the new CLI contract fails before implementation.
- [ ] Implement strict checkpoint/config validation, paired direct/ideal evaluation, fixed 10 dB and random `[5,10,15]` dB evaluation with two deterministic realizations per utterance, waveform/channel metrics, and stop-on-failed-ideal behavior.
- [ ] Run focused tests and a dry run.

### Task 3: Conditional clean-channel fine-tuning

**Files:**
- Create: `train_waveform_aware_clean_channel.py`
- Create: `scripts/run_waveform_aware_wireless_external.sh`
- Modify: `tests/test_waveform_aware_wireless.py`

**Interfaces:**
- Consumes: zero-shot summary and CF-2 source checkpoint.
- Produces: conditional optimizer path and `best_summed_latent_nmse.pt`, `best_waveform_si_sdr.pt`, `last.pt`.

- [ ] Add failing tests that training refuses when zero-shot passes, requires the exact source checkpoint, keeps codec frozen, and names selection checkpoints unambiguously.
- [ ] Run focused tests and confirm expected failures.
- [ ] Implement the existing three-stage curriculum over random clean channels, component gradient logging, strict codec freezing, and external script flags.
- [ ] Run focused tests and script dry runs.

### Task 4: Verification and report handoff

**Files:**
- Create at runtime: `runs/waveform_aware_wireless/final_comparison/summary.json`
- Create at runtime: `runs/waveform_aware_wireless/final_comparison/report.md`

**Interfaces:**
- Consumes: available ideal, zero-shot, and optional training summaries.
- Produces: selected checkpoint and jammer-unblocked decision.

- [ ] Run `pytest tests/test_waveform_aware_wireless.py tests/test_channel_free_revalidation.py tests/test_g1_mapping_equivalence.py tests/test_stage1_dataflow_integrity.py -q`.
- [ ] Run `pytest tests/ -q`.
- [ ] Run tensor preflight and ideal-OFDM evaluation when the local device/runtime permits; otherwise provide the exact external CUDA command without claiming a result.
- [ ] Inspect generated artifacts and report only evidence actually produced.
