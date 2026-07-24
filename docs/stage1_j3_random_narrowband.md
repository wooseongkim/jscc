# J3 random narrowband barrage diagnostic

J3 preserves the accepted J2 Conv-Conformer, frozen SpeechTokenizer, uniform
1920-symbol mapping, pilot-reserved grid, DFT-tap LS estimator, and estimated-ZF
equalizer. It changes only the jammer mask: a contiguous non-wrapping frequency
interval is active over all OFDM time symbols, with a new location, legitimate
channel, jammer channel, Gaussian jammer waveform, and AWGN realization per
training step.

The boundary sweep evaluates the accepted J2 checkpoint at SNR 5/10/15 dB,
requested global JSR -10/-5/0 dB, and jammed fractions 0.125/0.25/0.50. The
training distribution is not valid until that sweep writes
`selected_training_distribution.json`.

Global JSR averages jammer and signal power over the complete grid. The
approximate local in-band value is
`global_jsr_db - 10*log10(jammed_subcarrier_fraction)`. Both requested global
and realized received global/in-band values are logged separately.

Run the external stages in order:

```bash
bash scripts/run_j3_narrowband_boundary.sh --device cuda
bash scripts/run_j3_conv_conformer_external.sh --device cuda
```

The training launcher always transfers strictly from the accepted J2
checkpoint and never starts J4. The provisional gates are stored in
`configs/conv_conformer_j3_random_narrowband.yaml`; aggregate, Layers 1-7,
Layers 6-7, Layer 7, strongest-condition, mask, diversity, finite, CSI, and
equalizer checks all contribute to classification.
