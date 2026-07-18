# Baseline Metric Protocol Comparison

Recommended protocol for main reporting:

- `metric_align=peak_xcorr`
- `snr_scale_match=true`
- `metric_zero_mean=true`
- report mean and median together
- always report SI-SDR <= -10 dB outlier count

`metric_align=none` results are retained for appendix-style comparison.

| setting | metric | mean | median | std | p10 | p90 | min | max | n | SI-SDR<=-10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A: align=none, scale_match=false | waveform_snr_db | 4.555 | 4.40608 | 1.15544 | 3.36839 | 6.14388 | 1.16211 | 7.27802 | 100 | 0 |
| A: align=none, scale_match=false | si_sdr_db | 3.21408 | 3.21541 | 1.66826 | 1.44114 | 5.26084 | -2.43295 | 6.81092 | 100 | 0 |
| A: align=none, scale_match=false | stft_l1 | 0.0543092 | 0.0503808 | 0.0186862 | 0.034318 | 0.0762197 | 0.019274 | 0.144088 | 100 | 0 |
| A: align=none, scale_match=false | best_lag_samples | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100 | 0 |
| A: align=none, scale_match=false | input_rms | 0.0480088 | 0.0457852 | 0.0150101 | 0.0349675 | 0.0632239 | 0.0186744 | 0.130166 | 100 | 0 |
| A: align=none, scale_match=false | output_rms | 0.0477899 | 0.0455713 | 0.0144873 | 0.0351704 | 0.0635993 | 0.0172054 | 0.126818 | 100 | 0 |
| B: align=none, scale_match=true | waveform_snr_db | 4.9781 | 4.90903 | 1.10206 | 3.79044 | 6.39298 | 1.96202 | 7.63294 | 100 | 0 |
| B: align=none, scale_match=true | si_sdr_db | 3.21408 | 3.21541 | 1.66826 | 1.44114 | 5.26084 | -2.43295 | 6.81092 | 100 | 0 |
| B: align=none, scale_match=true | stft_l1 | 0.0543092 | 0.0503808 | 0.0186862 | 0.034318 | 0.0762197 | 0.019274 | 0.144088 | 100 | 0 |
| B: align=none, scale_match=true | best_lag_samples | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100 | 0 |
| B: align=none, scale_match=true | input_rms | 0.0480088 | 0.0457852 | 0.0150101 | 0.0349675 | 0.0632239 | 0.0186744 | 0.130166 | 100 | 0 |
| B: align=none, scale_match=true | output_rms | 0.0477899 | 0.0455713 | 0.0144873 | 0.0351704 | 0.0635993 | 0.0172054 | 0.126818 | 100 | 0 |
| C: align=peak_xcorr, scale_match=false | waveform_snr_db | 4.79199 | 4.84793 | 1.08745 | 3.41192 | 6.17699 | 2.04152 | 7.53873 | 100 | 0 |
| C: align=peak_xcorr, scale_match=false | si_sdr_db | 3.54288 | 3.63231 | 1.53856 | 1.50566 | 5.45166 | -0.601317 | 7.11135 | 100 | 0 |
| C: align=peak_xcorr, scale_match=false | stft_l1 | 0.0542678 | 0.0503808 | 0.0186914 | 0.0343298 | 0.0762197 | 0.0193234 | 0.144088 | 100 | 0 |
| C: align=peak_xcorr, scale_match=false | best_lag_samples | 0.3 | 0 | 0.888819 | -1 | 1 | -3 | 3 | 100 | 0 |
| C: align=peak_xcorr, scale_match=false | input_rms | 0.0480088 | 0.0457852 | 0.0150101 | 0.0349675 | 0.0632239 | 0.0186744 | 0.130166 | 100 | 0 |
| C: align=peak_xcorr, scale_match=false | output_rms | 0.0477899 | 0.0455713 | 0.0144873 | 0.0351704 | 0.0635993 | 0.0172054 | 0.126818 | 100 | 0 |
| D: align=peak_xcorr, scale_match=true | waveform_snr_db | 5.19098 | 5.19564 | 1.06116 | 3.82807 | 6.54061 | 2.72007 | 7.88301 | 100 | 0 |
| D: align=peak_xcorr, scale_match=true | si_sdr_db | 3.54288 | 3.63231 | 1.53856 | 1.50566 | 5.45166 | -0.601317 | 7.11135 | 100 | 0 |
| D: align=peak_xcorr, scale_match=true | stft_l1 | 0.0542678 | 0.0503808 | 0.0186914 | 0.0343298 | 0.0762197 | 0.0193234 | 0.144088 | 100 | 0 |
| D: align=peak_xcorr, scale_match=true | best_lag_samples | 0.3 | 0 | 0.888819 | -1 | 1 | -3 | 3 | 100 | 0 |
| D: align=peak_xcorr, scale_match=true | input_rms | 0.0480088 | 0.0457852 | 0.0150101 | 0.0349675 | 0.0632239 | 0.0186744 | 0.130166 | 100 | 0 |
| D: align=peak_xcorr, scale_match=true | output_rms | 0.0477899 | 0.0455713 | 0.0144873 | 0.0351704 | 0.0635993 | 0.0172054 | 0.126818 | 100 | 0 |
