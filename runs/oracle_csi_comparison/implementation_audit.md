# Clean wireless SNR and equalization implementation audit

This audit records the behavior in `src/channels/rayleigh.py`,
`src/channels/multipath.py`, `src/channels/pilot.py`, and
`src/evaluation/paired.py`. It does not change runtime behavior.

1. `rayleigh_channel` computes requested complex-noise power as
   `mean(|transmitted|^2) / 10^(SNR_dB/10)`. The paired-evaluation generator
   used by the clean wireless diagnostic constructs explicit complex Gaussian
   noise from the configured transmit target power using the same equation.
   Complex Gaussian samples are normalized by `sqrt(2)`, so their expected
   complex power is the requested variance.
2. Noise is referenced to transmit/grid symbol power. It is not referenced to
   received power after `H`, and the channel is not normalized per realization.
3. The exponential PDP is normalized to sum to one. Complex tap variances
   therefore sum to one and `E[|H[k]|^2]=1`. Individual channel realizations
   are not rescaled to make their realized mean frequency-response power one.
4. The legacy `post_equalization_sinr` equalizes separately identified desired,
   jammer, and AWGN tensors, then computes desired-output power divided by
   jammer-output plus noise-output power.
5. Consequently, the legacy value is not `sum|x|^2/sum|x_hat-x|^2`. With
   estimated CSI it does not count `(H/H_hat-1)x` as residual distortion; the
   equalized desired term is retained in the numerator. The oracle comparison
   adds a separately named empirical residual SINR.
6. ZF uses `H*.conj()/clamp(|H|^2, 1e-6)`. There is no gain clipping unless an
   explicit `gain_cap` is supplied, and the accepted clean path supplies none.
   Deep-fade resources are not masked. Thus this is exact complex division
   except where `|H|^2 < 1e-6`, where epsilon regularization limits the gain.

For the paired B/C comparison, the transmitted symbols, true multipath channel,
AWGN, pilot mask, resource map, and decoder observable state are shared. Only
the equalizer coefficient differs: true `H` for B and pilot-derived `H_hat` for
C.
