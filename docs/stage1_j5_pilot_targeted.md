# Stage-1 J5 pilot-targeted jammer

J1–J4 are immutable accepted curriculum evidence. J4 is frozen by
`runs/stage1_conv_conformer_jammer/j4_random_burst/accepted_manifest.json`; J5
launchers verify the checkpoint and every linked evidence hash before loading
weights.

J5 attacks only a deterministic random subset of the existing 128 pilot
resource elements. Direct jammer leakage on the 1920 data elements is required
to be zero. The requested pilot-local JSR is measured on attacked pilots. Its
full-grid equivalent is

`pilot_local_jsr_db + 10 log10(attacked_pilots / 2048)`.

Execution order:

1. `bash scripts/run_j5_pilot_boundary_external.sh --device cuda`
2. Inspect `j5_pilot_targeted/selected_training_distribution.json`.
3. `bash scripts/run_j5_conv_conformer_external.sh --device cuda`
4. `bash scripts/run_j5_final_diagnostics_external.sh --device cuda`
5. `bash scripts/run_j5_waveform_bridge_external.sh --device cuda`

The boundary sweep is required before training. The primary J5 classification
uses estimated CSI and latent gates. Oracle CSI and waveform metrics are
diagnostic/observational and cannot turn a production latent failure into PASS.
