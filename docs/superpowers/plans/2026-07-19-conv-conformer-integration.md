# Conv-Conformer G1-G3 Integration Plan

**Goal:** Sequentially validate accepted Conv-Conformer through identity mapping, fixed clean channel, and random clean channels.

**Architecture:** Reuse production allocation, pilot-reserved mapper, multipath, DFT-tap LS, ZF, observable receiver state, and G0 exposure utilities. Add a diagnostic-only integration module, strict provenance, sequential runner, and evidence reports.

## Tasks

1. Add RED tests and implement G1-E numerical equivalence.
2. Add fixed/random clean realization helpers and revised stage gates.
3. Add strict stage metadata/resume validation and progression decisions.
4. Add exposure-normalized integration CLI using fresh Conv-Conformer initialization.
5. Add safe stage-specific and sequence external scripts.
6. Run focused tests, full repository tests, dry-runs, G1-E, and bounded smoke checks only.
