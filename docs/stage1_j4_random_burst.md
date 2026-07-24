# J4 random full-band burst diagnostic

J4 transfers strictly from the corrected J3 `ACCEPTED_PASS` manifest. The jammer
occupies all 64 subcarriers during one contiguous, non-wrapping interval of the
32-symbol OFDM time axis. Global-grid JSR and concentrated active-window JSR are
logged separately. Boundary evaluation uses 3 SNR values, 3 global JSR values,
3 burst fractions, three content groups, and 16 paired realizations.

Run in order:

```bash
bash scripts/run_j4_burst_boundary.sh --device cuda
bash scripts/run_j4_conv_conformer_external.sh --device cuda
```

The second command is valid only when the boundary writes a defined distribution.
It never starts J5. PASS additionally requires non-negative Layer-7 10th-percentile
improvement and at most 10% negative Layer-7 realizations; otherwise mean-gate
success is classified `MARGINAL_TAIL`.
