# Conv-Conformer J1 external commands

Primary J1 run:

```bash
bash scripts/run_j1_conv_conformer_external.sh --device cuda --subset-size 256 --batch-size 4 --max-epochs 64 --num-workers 0
```

If and only if J1 fails, the existing fixed-realization diagnostic can isolate likely causes without changing production behavior:

```bash
python diagnose_o5_root_cause.py --config configs/conv_conformer_integration_v1.yaml --condition data_only_barrage_estimated_csi --steps 500 --seed 23 --output_dir runs/stage1_conv_conformer_jammer/j1_failure_diagnostics/jammer_on_data_only --allow_long_run
python diagnose_o5_root_cause.py --config configs/conv_conformer_integration_v1.yaml --condition pilot_only_jammer_estimated_csi --steps 500 --seed 23 --output_dir runs/stage1_conv_conformer_jammer/j1_failure_diagnostics/jammer_on_pilots_only --allow_long_run
python diagnose_o5_root_cause.py --config configs/conv_conformer_integration_v1.yaml --condition full_barrage_oracle_csi --steps 500 --seed 23 --output_dir runs/stage1_conv_conformer_jammer/j1_failure_diagnostics/oracle_legitimate_csi --allow_long_run
python diagnose_o5_root_cause.py --config configs/conv_conformer_integration_v1.yaml --condition full_barrage_estimated_csi --steps 500 --seed 23 --output_dir runs/stage1_conv_conformer_jammer/j1_failure_diagnostics/fixed_jammer_realization --allow_long_run
python diagnose_o5_root_cause.py --config configs/conv_conformer_integration_v1.yaml --condition full_barrage_estimated_csi --steps 1000 --seed 23 --output_dir runs/stage1_conv_conformer_jammer/j1_failure_diagnostics/fixed_legitimate_and_jammer_channels --allow_long_run
```

These are conditional offline diagnostics. They are not J2, do not feed oracle information to the neural network, and are not run automatically.
