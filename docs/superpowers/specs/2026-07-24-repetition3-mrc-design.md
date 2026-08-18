# R3 Three-Copy Coherent MRC Design

## Scope

Add an evaluation-only R3 transmission mode that sends each of the existing
1,920 CF-2 source symbols exactly once in each of three disjoint 72-subcarrier
frequency groups. Production G/J, checkpoint architecture, decoder interface,
and source-symbol count remain unchanged.

## Mapping

Each group has 1,944 candidate data RE and independently selects 1,920 using
only the previous-TTI CSI report. TTI 0 uses deterministic uniform bootstrap.
Branch-specific two-dimensional spread orders use time offsets 0, 9, and 18.
The selected branch RE form 1,920 triplets. A robust sum-log reliability score
drives the existing provisional layer-priority assignment while retaining 240
source symbols per layer.

## Energy

The primary `fixed_power_per_copy` contract independently normalizes each
branch power weights to mean one, giving an expected energy budget of 1,920
per branch and 5,760 per packet. Optional `fixed_total_packet_energy` scales
each branch to mean one third and is not used for the full evaluation.

## Combining

Raw received samples, current receiver CSI, known transmit amplitudes, and
noise variance are reordered into source order. The unbiased coherent MRC
estimate is computed directly. No branch equalization, MMSE gain undoing,
deep-fade threshold, clamp, erasure, or branch deletion is permitted.

## Verification

Tests cover frequency partitions, mapping cardinality and inversion,
causality, branch power/energy, scalar and tensor MRC, oracle theory, finite
behavior, and existing physical/production regressions. A short checkpoint
smoke precedes one R3 64×2×3 evaluation.
