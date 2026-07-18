# Channel Estimator Comparison

- channel model: `multipath_block`
- estimators: `['inverse_distance_2d', 'block_frequency_ls', 'dft_tap_ls', 'oracle']`


## SINR Definition

Post-equalization SINR is computed over data resources only. Pilot positions are excluded before power averaging, so zero-filled or pilot resources do not dilute the metric.
For an equalizer `G`, the diagnostic computes `G H_s X`, `G H_j J`, and `G W`, averages their powers over data resources, then reports `10 log10(P_signal / (P_interference + P_noise))`.
Both `estimated_minus_oracle_sinr_db` and `oracle_minus_estimated_sinr_db` are stored without clamping. Positive estimated-minus-oracle means the estimated ZF equalizer has higher aggregate data-resource SINR than oracle ZF for that realization.
Mean per-seed SINR in dB and dB of mean linear SINR are different and are stored separately in JSON aggregates.
Estimated ZF can occasionally exceed oracle ZF in this aggregate diagnostic because zero-forcing equalizer gain changes the weighting of noise/interference across deep fades; this report does not claim oracle ZF maximizes every aggregate weighting.

| estimator | SNR | jammer | median NMSE | mean NMSE | mean est SINR dB | mean oracle SINR dB | mean est-oracle dB |
|---|---:|---|---:|---:|---:|---:|---:|
| block_frequency_ls | 5.0 | none | 0.2663505971431732 | 0.29266673102974894 | 1.4535375234112144 | -1.7512585108727217 | 3.204796034283936 |
| block_frequency_ls | 10.0 | none | 0.21477502584457397 | 0.2378885778784752 | 6.215624589323998 | 3.2487416134774687 | 2.966882975846529 |
| block_frequency_ls | 20.0 | none | 0.19313450157642365 | 0.21378795605152845 | 16.230461716651917 | 13.248741419315339 | 2.981720297336578 |
| block_frequency_ls | 30.0 | none | 0.18985207378864288 | 0.2108435570076108 | 26.179559993743897 | 23.248740873336793 | 2.9308191204071044 |
| dft_tap_ls | 5.0 | none | 0.055457863956689835 | 0.0735677933320403 | -0.6537466124864295 | -1.7512585108727217 | 1.0975118983862922 |
| dft_tap_ls | 10.0 | none | 0.017537303268909454 | 0.023264179960824548 | 3.940902409926057 | 3.2487416134774687 | 0.6921607964485884 |
| dft_tap_ls | 20.0 | none | 0.0017537258099764585 | 0.0023264185123844073 | 13.450590258836746 | 13.248741419315339 | 0.20184883952140809 |
| dft_tap_ls | 30.0 | none | 0.00017537102394271642 | 0.0002326420443569077 | 23.313237313628196 | 23.248740873336793 | 0.06449644029140472 |
| inverse_distance_2d | 5.0 | none | 0.4703715741634369 | 0.48566368013620376 | 2.5667988356947897 | -1.7512585108727217 | 4.318057346567511 |
| inverse_distance_2d | 10.0 | none | 0.3779904246330261 | 0.38086498633027077 | 7.203010967075825 | 3.2487416134774687 | 3.9542693535983564 |
| inverse_distance_2d | 20.0 | none | 0.3296850919723511 | 0.3358685295283794 | 17.064502444267273 | 13.248741419315339 | 3.815761024951935 |
| inverse_distance_2d | 30.0 | none | 0.317146897315979 | 0.3308033633232117 | 27.1182900428772 | 23.248740873336793 | 3.8695491695404054 |
| oracle | 5.0 | none | 0.0 | 0.0 | -1.7512585108727217 | -1.7512585108727217 | 0.0 |
| oracle | 10.0 | none | 0.0 | 0.0 | 3.2487416134774687 | 3.2487416134774687 | 0.0 |
| oracle | 20.0 | none | 0.0 | 0.0 | 13.248741419315339 | 13.248741419315339 | 0.0 |
| oracle | 30.0 | none | 0.0 | 0.0 | 23.248740873336793 | 23.248740873336793 | 0.0 |
