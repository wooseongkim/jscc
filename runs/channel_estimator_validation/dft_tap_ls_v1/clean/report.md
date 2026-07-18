# Channel Estimator Comparison

- channel model: `multipath_block`
- estimators: `['inverse_distance_2d', 'block_frequency_ls', 'dft_tap_ls', 'oracle']`

| estimator | SNR | jammer | median NMSE | mean NMSE | median SINR loss |
|---|---:|---|---:|---:|---:|
| block_frequency_ls | 5.0 | none | 0.2663505971431732 | 0.29266673102974894 | 0.0 |
| block_frequency_ls | 10.0 | none | 0.21477502584457397 | 0.2378885778784752 | 0.0 |
| block_frequency_ls | 20.0 | none | 0.19313450157642365 | 0.21378795605152845 | 0.0 |
| block_frequency_ls | 30.0 | none | 0.18985207378864288 | 0.2108435570076108 | 0.0 |
| dft_tap_ls | 5.0 | none | 0.055457863956689835 | 0.0735677933320403 | 0.0 |
| dft_tap_ls | 10.0 | none | 0.017537303268909454 | 0.023264179960824548 | 0.0 |
| dft_tap_ls | 20.0 | none | 0.0017537258099764585 | 0.0023264185123844073 | 0.0 |
| dft_tap_ls | 30.0 | none | 0.00017537102394271642 | 0.0002326420443569077 | 0.0 |
| inverse_distance_2d | 5.0 | none | 0.4703715741634369 | 0.48566368013620376 | 0.0 |
| inverse_distance_2d | 10.0 | none | 0.3779904246330261 | 0.38086498633027077 | 0.0 |
| inverse_distance_2d | 20.0 | none | 0.3296850919723511 | 0.3358685295283794 | 0.0 |
| inverse_distance_2d | 30.0 | none | 0.317146897315979 | 0.3308033633232117 | 0.0 |
| oracle | 5.0 | none | 0.0 | 0.0 | 0.0 |
| oracle | 10.0 | none | 0.0 | 0.0 | 0.0 |
| oracle | 20.0 | none | 0.0 | 0.0 | 0.0 |
| oracle | 30.0 | none | 0.0 | 0.0 | 0.0 |
