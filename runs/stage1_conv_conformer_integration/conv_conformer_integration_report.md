# Conv-Conformer integration

G0 is accepted. G1-E equivalence is stored separately. Unexecuted stages are not classified.

## Stages

- g0_direct: passed
- g1_mapping_train: passed
- g2_fixed_clean: passed
- g3_random_clean: passed
- j1_weak_random_barrage: ready, not yet externally completed
- j2-j5: blocked until J1 passes

## Next external command

`bash scripts/run_j1_conv_conformer_external.sh --device cuda --subset-size 256 --batch-size 4 --max-epochs 64 --num-workers 0`
