# O5 Root-Cause Report

## Paired 500-step comparison

All rows below are extracted at exactly step 500. Different optimization budgets are not mixed.

- clean_awgn_reference: loss=0.111219123, power_ratio=0.858551, correlation=0.948405
- data_only_barrage_estimated_csi: loss=0.199038818, power_ratio=0.738905, correlation=0.904579
- data_only_barrage_oracle_csi: loss=0.198155746, power_ratio=0.737182, correlation=0.905224
- full_barrage_estimated_csi: loss=0.248782903, power_ratio=0.672947, correlation=0.878736
- full_barrage_oracle_csi: loss=0.197877690, power_ratio=0.737415, correlation=0.905378
- full_barrage_oracle_subtraction: loss=0.111218460, power_ratio=0.858617, correlation=0.948405
- pilot_only_jammer_estimated_csi: loss=0.061544217, power_ratio=0.926359, correlation=0.972166

## Same-trajectory extensions

- full_barrage_estimated_csi: 500 steps loss 0.248782903 -> 1000 steps loss 0.024842672 (optimization_duration_limited).

## Evidence-based conclusion

- Fixed full-barrage learning with estimated CSI is possible for the C1 realization.
- C1 improved on the same trajectory from step 500 to step 1000, so 500 steps were insufficient for that realization.
- This fixed-realization result does not establish generalization to random channels or random jammers.
- Pilot contamination changes convergence but is not a structural impossibility in this fixed experiment.
- Oracle jammer subtraction returning near the clean reference validates the jammer integration path.
- The historical O5 and C1 use different realization seeds and must not be interpreted as a direct performance improvement.
