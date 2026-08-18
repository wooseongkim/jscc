# R4 Broadband UEP Optimizer Design

## Scope

Implement a simulation-based mixed-integer nonlinear black-box optimizer for
layer-wise repetition and total power shares under an oracle broadband-AWGN
attack class.  The optimizer creates and ranks candidates only; it reuses the
existing UEP waveform evaluator/forward path.

## Protocol boundary

Selection search uses `expanded_selection` exclusively.  Candidate generation,
objective calculation, Pareto ranking, and profile selection must not read a
legacy-final manifest, result, or metric.  The final-evaluation mode accepts a
frozen `selected_profiles.json` and is the sole code path allowed to load
`legacy_final`.

## Candidate

Each candidate is `(r, z)`:

- `r_i` is integer in `{1,2,3,4,5}`, with `sum(r)=24`.
- `p=softmax(z)`, so every power share is positive and shares sum to one.
- Per-RE multiplier is `a_i=24*p_i/r_i`.
- Every layer has 240 source symbols; total RE is exactly 5,760 and total
  energy is exactly 5,760 after placement.

`U0` is represented by `r=(3,...,3), p=(1/8,...,1/8)` and delegates to the
historical global balanced-triplet path.  Hand-crafted P2/RP2 do not appear in
the candidate generator or refiner.

## Evaluation backend

The backend reuses the UEP evaluator's R4 waveform forward, official
unaligned/full-crop SI-SDR, LS channel estimation, coherent MRC, and canonical
broadband jammer tensor replay.  A condition cache is built once for U0, then
the same waveform, crop, channel, AWGN, jammer grid/tensor hash, and target are
passed to every candidate.  Candidate-specific repetition mapping and
power-dependent MRC effective channel are the only differences.

Selection U0 is an internally cached paired baseline.  Legacy final-evaluation
replays the canonical historical broadband seeds and requires the U0 anchor.

## Search

1. Random stage samples feasible `r` without layer-priority bias and samples
   Dirichlet powers with alpha 0.5, 1, 2, and 5 using a fixed seed.
2. The scalar-objective top-K candidates seed local refinement.
3. Local refinement uses valid one-unit repetition transfers and simplex power
   transfers over a fixed delta schedule.
4. Duplicates use a deterministic candidate hash and are not re-evaluated.
5. All candidates are Pareto-ranked after evaluation.

## Objective and selection

The scalar objective is the weighted paired SI-SDR delta over broadband JSR
`-5,0,5,10`, less penalties for clean cost above 0.2 dB, p5 delta below -1 dB,
increased severe negative-tail fraction, and positive STFT-L1 regression.

Select:

- `x_best`: maximum scalar objective;
- `x_stable`: Pareto candidate satisfying clean cost <=0.2 dB, p5 >=-1 dB and
  severe-tail increase <=2%, with largest mean delta;
- `x_aggressive`: Pareto candidate with clean cost <=0.3 dB and largest mean
  delta.

If a category has no valid profile it is recorded as `NOT_FOUND`; no fallback
profile is substituted.

## Validation

Tests cover feasible repetition enumeration, simplex/energy math, deterministic
candidate hashes, unbiased random generation, paired objective computation,
Pareto rules, and the prohibition on legacy search input.  CPU validates the
structure; CUDA smoke is required before reporting implementation complete.
