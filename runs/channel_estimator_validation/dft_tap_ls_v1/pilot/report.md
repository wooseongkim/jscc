# Channel Estimator Comparison

- channel model: `multipath_block`
- estimators: `['block_frequency_ls', 'dft_tap_ls', 'oracle']`

| estimator | SNR | jammer | median NMSE | mean NMSE | median SINR loss |
|---|---:|---|---:|---:|---:|
| block_frequency_ls | 10.0 | pilot | 3.3043744564056396 | 4.569692158699036 | 0.0 |
| dft_tap_ls | 10.0 | pilot | 3.2299599647521973 | 4.530115356147289 | 0.0 |
| oracle | 10.0 | pilot | 0.0 | 0.0 | 0.0 |
