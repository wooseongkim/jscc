# R3 Three-Copy Coherent MRC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement three-copy R3 frequency diversity and raw-observation coherent MRC.

**Architecture:** A focused repetition/MRC module owns group selection, triplet mapping, power, and combining. The physical OFDM evaluator remains separate and loads the accepted checkpoint without training.

**Tech Stack:** Python, PyTorch complex tensors, pytest, YAML.

## Global Constraints

- Exactly 1,920 source symbols and 5,760 physical data RE.
- No coding, clamp, erasure, retraining, jammer, or production changes.
- Primary energy is fixed power per copy; default output is unbiased coherent MRC.

### Task 1: Mapping and MRC primitives

- [ ] Add failing cardinality, causality, energy, and analytic MRC tests.
- [ ] Verify RED.
- [ ] Implement `src/channels/repetition_mrc.py`.
- [ ] Verify focused tests GREEN.

### Task 2: Physical checkpoint evaluation

- [ ] Add configuration and script contract tests.
- [ ] Implement `evaluate_repetition3_mrc.py` and external launcher.
- [ ] Run dry-run and two-utterance checkpoint smoke.

### Task 3: Full evaluation and regression

- [ ] Run the single R3 64×2×3 evaluation.
- [ ] Validate artifacts and engineering gates.
- [ ] Run focused and full repository tests.
