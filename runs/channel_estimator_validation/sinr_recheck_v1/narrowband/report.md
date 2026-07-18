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
| block_frequency_ls | 10.0 | narrowband | 0.33604127168655396 | 0.4149926418066025 | -1.233633930720389 | -5.084889217466116 | 3.8512552867457273 |
| dft_tap_ls | 10.0 | narrowband | 0.1386672407388687 | 0.21363479599356652 | -2.686321058785543 | -5.084889217466116 | 2.398568158680573 |
| oracle | 10.0 | narrowband | 0.0 | 0.0 | -5.084889217466116 | -5.084889217466116 | 0.0 |
