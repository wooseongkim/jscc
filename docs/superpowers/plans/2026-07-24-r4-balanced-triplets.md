# R4 Balanced Triplets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add R4 physical OFDM, physical-delay preservation, global balanced triplets, and bounded weak-triplet power.

**Architecture:** Extend physical profiles and delay utilities; keep allocation/power in focused modules; reuse coherent MRC unchanged.

**Tech Stack:** Python, PyTorch, pytest, YAML.

1. Write failing R4 physical/delay tests; implement profile and sparse delays.
2. Write failing global mapping/power tests; implement allocator and projector.
3. Implement evaluator/config/script and allocator validation.
4. Run smoke, one full R4 evaluation, artifact checks, and full regressions.
