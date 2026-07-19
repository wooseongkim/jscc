# Stage-1 Distribution Diagnostic Progress

- Corrected O5: fixed C1 is learnable; its 500→1000 continuation shows optimization-duration limitation.
- Historical O5 and C1 are different realizations and are not directly comparable.
- O6 evidence: FAIL.
- J1 evidence: exploratory_failed_parent (never a J2 parent).
- o6_random_clean: not completed or failed
- j1_weak_barrage: not completed or failed
- j2_moderate_barrage: not completed or failed
- j3_strong_barrage: not completed or failed
- j4_mixed_sparse: not completed or failed
- j5_full_mixture: not completed or failed
- Full Uniform readiness: False

## Exact next external command

`bash scripts/run_stage1_jammer_curriculum_external.sh --stage j1_weak_barrage --device cuda`
