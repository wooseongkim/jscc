# J5 Pilot-Targeted Jammer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Freeze accepted J4 evidence and add reproducible pilot-only J5 boundary, training, paired evaluation, and waveform bridge infrastructure.

**Architecture:** Reuse paired evaluation and content-generalization engines. Add partial pilot-mask generation, pilot-local power normalization/diagnostics, strict parent verification, explicit J5 gates, and external-only launchers.

**Tech Stack:** Python, PyTorch, pytest, YAML, shell.

## Tasks

- [ ] Add RED tests for J4 acceptance, partial pilot masks, local/global JSR, gain logging, and J5 gates.
- [ ] Implement pure J5 diagnostics and accepted-manifest generation.
- [ ] Extend paired equalizer logging without changing unclipped behavior.
- [ ] Implement boundary, training, final paired, and waveform bridge CLIs and safe external scripts.
- [ ] Run focused tests, CPU smoke/dry runs, and full pytest; do not execute long CUDA jobs.
