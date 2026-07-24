# Temporal channel and CSI-path audit

This audit reflects the implementation present before the separate FDD engine
was added.

- `multipath_block_fading()` sampled fresh complex Gaussian taps on every call.
  Calls made for different batches, utterances, or optimization steps therefore
  used independent channels.
- The sampled taps did not persist between slots. No Doppler, Jakes, or other
  temporal evolution was implemented.
- Within one generated OFDM grid, the frequency response was block-fading over
  all OFDM symbols.
- Frequency response values were correctly produced by an FFT of time-domain
  multipath taps; they were not independently sampled per subcarrier.
- The legacy `estimate_transmitter_feedback()` path estimated CSI from the
  current batch pilots and could feed that same estimate to same-slot
  reliability allocation. That is non-causal for delayed FDD feedback.
- The accepted waveform-aware clean-channel runs used uniform allocation, so
  this same-slot feedback path did not affect those results.

The legacy iid channel and production G/J paths remain unchanged. The FDD
evaluation uses a separate tap-trajectory engine and a hard one-slot feedback
buffer.
