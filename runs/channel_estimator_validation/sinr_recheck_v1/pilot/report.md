# Channel Estimator Comparison

- channel model: `multipath_block`
- estimators: `['block_frequency_ls', 'dft_tap_ls', 'oracle']`


## SINR Definition

Post-equalization SINR is computed over data resources only. Pilot positions are excluded before power averaging, so zero-filled or pilot resources do not dilute the metric.
For an equalizer `G`, the diagnostic computes `G H_s X`, `G H_j J`, and `G W`, averages their powers over data resources, then reports `10 log10(P_signal / (P_interference + P_noise))`.
Both `estimated_minus_oracle_sinr_db` and `oracle_minus_estimated_sinr_db` are stored without clamping. Positive estimated-minus-oracle means the estimated ZF equalizer has higher aggregate data-resource SINR than oracle ZF for that realization.
Mean per-seed SINR in dB and dB of mean linear SINR are different and are stored separately in JSON aggregates.
Estimated ZF can occasionally exceed oracle ZF in this aggregate diagnostic because zero-forcing equalizer gain changes the weighting of noise/interference across deep fades; this report does not claim oracle ZF maximizes every aggregate weighting.

| estimator | SNR | jammer | median NMSE | mean NMSE | mean est SINR dB | mean oracle SINR dB | mean est-oracle dB |
|---|---:|---|---:|---:|---:|---:|---:|
| block_frequency_ls | 10.0 | pilot | 3.3043744564056396 | 4.569692158699036 | 8.076886182129384 | 3.8803427344560624 | 4.196543447673321 |
| dft_tap_ls | 10.0 | pilot | 3.2299599647521973 | 4.530115356147289 | 8.174809215068818 | 3.8803427344560624 | 4.294466480612755 |
| oracle | 10.0 | pilot | 0.0 | 0.0 | 3.8803427344560624 | 3.8803427344560624 | 0.0 |
