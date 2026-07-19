# Stage-1 status

- G0: PASS
- G1: PASS
- G2: PASS
- G3: PASS
- J1: ready; external experiment not yet completed
- J2-J5: blocked until J1 passes

Next external command:

```bash
bash scripts/run_j1_conv_conformer_external.sh --device cuda --subset-size 256 --batch-size 4 --max-epochs 64 --num-workers 0
```
