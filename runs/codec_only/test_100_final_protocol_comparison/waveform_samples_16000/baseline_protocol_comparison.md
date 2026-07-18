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
| A: align=none, scale_match=false | waveform_snr_db | 4.51595 | 4.41833 | 1.65894 | 2.72635 | 6.64827 | -1.87314 | 8.07396 | 100 | 1 |
| A: align=none, scale_match=false | si_sdr_db | 2.8812 | 2.94298 | 2.87264 | 0.600764 | 5.83764 | -15.7038 | 7.747 | 100 | 1 |
| A: align=none, scale_match=false | stft_l1 | 0.0586061 | 0.0540586 | 0.0267748 | 0.0327295 | 0.087549 | 0.00390507 | 0.184763 | 100 | 1 |
| A: align=none, scale_match=false | best_lag_samples | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100 | 1 |
| A: align=none, scale_match=false | input_rms | 0.0484235 | 0.0474403 | 0.0197035 | 0.0268392 | 0.0717383 | 0.00233991 | 0.122482 | 100 | 1 |
| A: align=none, scale_match=false | output_rms | 0.046992 | 0.0453637 | 0.0188895 | 0.0250798 | 0.0701972 | 0.00220764 | 0.119234 | 100 | 1 |
| B: align=none, scale_match=true | waveform_snr_db | 4.88333 | 4.72646 | 1.50482 | 3.3216 | 6.84393 | 0.115174 | 8.42138 | 100 | 1 |
| B: align=none, scale_match=true | si_sdr_db | 2.8812 | 2.94298 | 2.87264 | 0.600764 | 5.83764 | -15.7038 | 7.747 | 100 | 1 |
| B: align=none, scale_match=true | stft_l1 | 0.0586061 | 0.0540586 | 0.0267748 | 0.0327295 | 0.087549 | 0.00390507 | 0.184763 | 100 | 1 |
| B: align=none, scale_match=true | best_lag_samples | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100 | 1 |
| B: align=none, scale_match=true | input_rms | 0.0484235 | 0.0474403 | 0.0197035 | 0.0268392 | 0.0717383 | 0.00233991 | 0.122482 | 100 | 1 |
| B: align=none, scale_match=true | output_rms | 0.046992 | 0.0453637 | 0.0188895 | 0.0250798 | 0.0701972 | 0.00220764 | 0.119234 | 100 | 1 |
| C: align=peak_xcorr, scale_match=false | waveform_snr_db | 4.98354 | 4.93972 | 1.62665 | 3.13415 | 6.90198 | -1.27775 | 8.97997 | 100 | 1 |
| C: align=peak_xcorr, scale_match=false | si_sdr_db | 3.5668 | 3.5491 | 2.53229 | 1.2526 | 6.26783 | -11.2098 | 8.51667 | 100 | 1 |
| C: align=peak_xcorr, scale_match=false | stft_l1 | 0.0583944 | 0.0542556 | 0.0266396 | 0.0327295 | 0.0874616 | 0.00387945 | 0.182775 | 100 | 1 |
| C: align=peak_xcorr, scale_match=false | best_lag_samples | 0.24 | 1 | 3.20037 | -1 | 2 | -28 | 10 | 100 | 1 |
| C: align=peak_xcorr, scale_match=false | input_rms | 0.0484235 | 0.0474403 | 0.0197035 | 0.0268392 | 0.0717383 | 0.00233991 | 0.122482 | 100 | 1 |
| C: align=peak_xcorr, scale_match=false | output_rms | 0.046992 | 0.0453637 | 0.0188895 | 0.0250798 | 0.0701972 | 0.00220764 | 0.119234 | 100 | 1 |
| D: align=peak_xcorr, scale_match=true | waveform_snr_db | 5.30472 | 5.1378 | 1.51271 | 3.6817 | 7.18713 | 0.316732 | 9.08836 | 100 | 1 |
| D: align=peak_xcorr, scale_match=true | si_sdr_db | 3.5668 | 3.5491 | 2.53229 | 1.2526 | 6.26783 | -11.2098 | 8.51667 | 100 | 1 |
| D: align=peak_xcorr, scale_match=true | stft_l1 | 0.0583944 | 0.0542556 | 0.0266396 | 0.0327295 | 0.0874616 | 0.00387945 | 0.182775 | 100 | 1 |
| D: align=peak_xcorr, scale_match=true | best_lag_samples | 0.24 | 1 | 3.20037 | -1 | 2 | -28 | 10 | 100 | 1 |
| D: align=peak_xcorr, scale_match=true | input_rms | 0.0484235 | 0.0474403 | 0.0197035 | 0.0268392 | 0.0717383 | 0.00233991 | 0.122482 | 100 | 1 |
| D: align=peak_xcorr, scale_match=true | output_rms | 0.046992 | 0.0453637 | 0.0188895 | 0.0250798 | 0.0701972 | 0.00220764 | 0.119234 | 100 | 1 |
