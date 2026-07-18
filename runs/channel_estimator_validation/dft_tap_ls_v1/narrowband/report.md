# Channel Estimator Comparison

- channel model: `multipath_block`
- estimators: `['block_frequency_ls', 'dft_tap_ls', 'oracle']`

| estimator | SNR | jammer | median NMSE | mean NMSE | median SINR loss |
|---|---:|---|---:|---:|---:|
| block_frequency_ls | 10.0 | narrowband | 0.33604127168655396 | 0.4149926418066025 | 0.0 |
| dft_tap_ls | 10.0 | narrowband | 0.1386672407388687 | 0.21363479599356652 | 0.0 |
| oracle | 10.0 | narrowband | 0.0 | 0.0 | 0.0 |
