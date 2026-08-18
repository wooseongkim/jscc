# R4 Perceptual and ASR Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic ESTOI, WER, ViSQOL, and absolute SI-SDR-tail reporting to the fixed R4 jammer evaluation without changing physical-layer, UEP, allocation, or refiner behavior.

**Architecture:** A small `speech_metrics` adapter owns optional full-reference ESTOI/ViSQOL and frozen-ASR WER evaluation. `evaluate_r4_jammer_refiner_checkpoints.py` calls it only after raw waveform reconstruction, writes per-row values, and aggregates by jammer/JSR/SNR. The raw decoder remains the paper metric; the refiner remains diagnostic-only.

**Tech Stack:** Python, pystoi, Whisper `small.en`, ViSQOL speech mode, pytest.

## Global Constraints

- Use the fixed held-out condition plan; do not alter seeds, crops, UEP `r,p`, mapping, MRC, or OFDM.
- Use raw decoder waveform only for the main reported metrics.
- No GPU command is run by Codex; CUDA commands are supplied to the user.
- ESTOI/WER/ViSQOL must be finite or fail fast, and must not silently become surrogate scores.
- WER uses LibriSpeech transcript labels; ASR model is frozen Whisper `small.en`.

---

### Task 1: Metric adapters and dependency contract

**Files:**
- Create: `src/speech_jscc/evaluation/speech_quality_metrics.py`
- Modify: `pyproject.toml`
- Test: `tests/test_speech_quality_metrics.py`

**Interfaces:**
- Produces `compute_estoi(reference, estimate, sample_rate) -> float`.
- Produces `compute_wer(reference_text, hypothesis_text) -> float`.
- Produces `compute_visqol(reference_path, estimate_path, sample_rate) -> float`.
- Produces `FrozenWhisperTranscriber(model_name, device).transcribe(waveform, sample_rate) -> str`.

- [ ] Write failing tests for ESTOI range/finiteness, normalized WER, unavailable optional backend failure, and no surrogate fallback.
- [ ] Run `pytest -q tests/test_speech_quality_metrics.py` and verify the missing module failure.
- [ ] Implement the adapters with explicit optional-dependency errors and add the metrics optional dependency group.
- [ ] Re-run the test file and verify pass.

### Task 2: Fixed evaluator integration and summaries

**Files:**
- Modify: `src/speech_jscc/evaluation/evaluate_r4_jammer_refiner_checkpoints.py`
- Modify: `configs/eval_r4_jammer_refiner_fixed.yaml`
- Modify: `tests/test_evaluate_r4_jammer_refiner_checkpoints.py`

**Interfaces:**
- Per-row columns: `raw_estoi`, `raw_wer`, `raw_visqol_mos_lqo`, `raw_si_sdr_lt_minus10`.
- Summary rows aggregate each metric and absolute SI-SDR below −10 dB fraction by jammer, JSR, and SNR.

- [ ] Write failing tests asserting raw-only metric columns, strict summary aggregation, and disabled metric modes preserve evaluator behavior.
- [ ] Run the focused evaluator tests and verify the new assertions fail.
- [ ] Wire the adapters after `raw_waveform` is decoded, persist one WAV only in a temporary metric workspace, and add config/CLI backend settings.
- [ ] Re-run focused tests and verify pass.

### Task 3: Determinism and smoke validation

**Files:**
- Modify: `tests/test_evaluate_r4_jammer_refiner_checkpoints.py`
- Create: `docs/r4_perceptual_metric_protocol.md`

- [ ] Write a failing test requiring identical metric row keys and deterministic cached transcript reuse across checkpoint variants.
- [ ] Run the test and verify expected failure.
- [ ] Implement deterministic cache keys from condition keys plus checkpoint identity; document resampling/normalization and model versions.
- [ ] Run `pytest -q tests/test_speech_quality_metrics.py tests/test_evaluate_r4_jammer_refiner_checkpoints.py`.
- [ ] Run a CPU two-row smoke after dependencies are installed; supply the full CUDA command for the user.
