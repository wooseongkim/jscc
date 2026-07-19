# Stage-1 Content-Generalization Report

## Prior evidence

- O6: **FAIL**. V1 is only a weak pass; random-channel learning is not considered solved.
- J1: **exploratory_failed_parent**. It is excluded from curriculum readiness and cannot parent J2.

## G0–G3 results

- g0_direct subset=16: FAIL
- g0_direct subset=64: FAIL
- g0_direct subset=256: FAIL
- g0_direct subset=full: FAIL

- First failing content stage: g0_direct
- Smallest passing subsets: `{}`

## Next command

`bash scripts/run_stage1_content_generalization_external.sh --device cuda`
