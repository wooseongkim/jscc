# J2 strong-barrage boundary diagnostic

J1 is immutable accepted evidence. J2 does not redefine or rerun J1.

## Required sequence

1. Run the dense no-training boundary sweep.
2. Inspect `selected_training_range.json`. Stop if `defined` is false.
3. Run the 512-step fresh-versus-J1 initialization comparison.
4. Inspect `initialization_decision.json`.
5. Run the selected 4096-step J2 diagnostic.

```bash
bash scripts/run_j2_boundary_sweep.sh --device cuda
bash scripts/run_j2_initialization_compare.sh --device cuda --steps 512
bash scripts/run_j2_conv_conformer_external.sh --device cuda --steps 4096 --batch-size 4
```

The sweep covers SNR `[0, 2.5, 5, 7.5, 10, 12.5, 15]` dB and requested JSR
`[-15, -12.5, -10, -7.5, -5, -2.5, 0, 2.5, 5]` dB with 16 independent
realizations per content group and grid point. The generated range is mandatory;
the placeholder channel range in the YAML is not a J2 training decision.

The initialization comparison uses identical explicit batch/channel/jammer/noise
seeds. `j1_transfer` loads only the accepted J1 model parameters; both policies
start a fresh Adam optimizer so optimizer policy is controlled.

J2 never passes requested or realized JSR, jammer labels, masks, or true channel
values to the neural model. Oracle comparisons remain offline diagnostics.

## Conditional root-cause follow-up

If the final J2 classification is failed or marginal, use identical seeds and
latent IDs to compare estimated/oracle CSI, clean/barrage, and full/data-only
jamming through `diagnose_o5_root_cause.py`. Equalizer clipping and layer-subset
comparisons are diagnostic-only and must be added only after the failing sample
IDs in the J2 summary identify the conditions to pair; they are not production
options and are not automatically executed.
