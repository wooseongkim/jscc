# NR-Inspired Physical OFDM Profiles Design

## Scope

Add an evaluation-only physical OFDM engine for `nr_like_r2` and
`nr_like_r3`. Preserve `legacy_64x32` for regression and leave production G/J
paths unchanged. R3 is the default and the only full scientific evaluation.

## Architecture

- `channels.physical_ofdm` owns immutable profile definitions, centered
  active-bin mappings, pilot/data/null masks, unitary OFDM modulation,
  cyclic-prefix handling, and time-domain multipath convolution.
- `speech_jscc.diagnostics.physical_fdd` owns deterministic candidate-RE
  selection, causal delayed-CSI selection, stratified layer placement,
  bounded power allocation, exact inverse mapping, and LMMSE recovery.
- `evaluate_physical_fdd.py` reuses the accepted CF-2 checkpoint, frozen
  SpeechTokenizer, time-correlated tap trajectories, and waveform metrics.

The LMMSE estimator includes the known transmit amplitude in the effective
channel. Its shrinkage is retained; it is not fully debiased into ZF.

## Physical Contract

R2 uses 144 active subcarriers and R3 uses 216, both with 256-point FFT,
30 kHz spacing, 7.68 MHz sampling, 18-sample CP, and 28 OFDM symbols per TTI.
Active bins are centered around and exclude DC. Pilot symbols are 3 and 17
with opposite comb-2 parity. Exactly 1,920 source symbols are transmitted
once; all unused candidate data RE are zero.

Nominal SNR is referenced to active data-symbol energy, not time-domain
waveform power or unused/null resources. Bounded delayed-CSI power weights
have alpha 0.5 and relative bounds 0.5–2.0, normalized to mean one over the
1,920 selected RE.

## Causality

TTI 0 uses deterministic uniform spreading. Later TTIs use only the receiver
report generated in the previous TTI. The receiver reconstructs selection,
mapping, and powers from the same immutable report. Current/future CSI is
rejected by hard assertions.

## Verification

Tests cover physical timing, masks and counts, centered bins, energy,
selection bijection, causality, bounded power, CP safety, OFDM identity,
time-domain/frequency-domain equivalence, and existing regression suites.
Only short R2/R3 checkpoint smoke evaluations run locally; the full R3
64×2×3 evaluation is external.
