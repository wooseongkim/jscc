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
| A: align=none, scale_match=false | waveform_snr_db | 4.56938 | 4.6661 | 1.42994 | 2.68429 | 6.27752 | 0.339169 | 7.50812 | 100 | 0 |
| A: align=none, scale_match=false | si_sdr_db | 3.18468 | 3.28664 | 2.04333 | 0.638045 | 5.59719 | -3.56552 | 6.98798 | 100 | 0 |
| A: align=none, scale_match=false | stft_l1 | 0.0573736 | 0.0514967 | 0.0212108 | 0.0325579 | 0.0834304 | 0.0152169 | 0.123225 | 100 | 0 |
| A: align=none, scale_match=false | best_lag_samples | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100 | 0 |
| A: align=none, scale_match=false | input_rms | 0.0486254 | 0.0451123 | 0.0161014 | 0.0315783 | 0.0660127 | 0.019738 | 0.104193 | 100 | 0 |
| A: align=none, scale_match=false | output_rms | 0.0483392 | 0.0450176 | 0.0156276 | 0.0308182 | 0.0658417 | 0.0198263 | 0.10747 | 100 | 0 |
| B: align=none, scale_match=true | waveform_snr_db | 4.99446 | 4.95736 | 1.30575 | 3.34107 | 6.65427 | 1.58345 | 7.7799 | 100 | 0 |
| B: align=none, scale_match=true | si_sdr_db | 3.18468 | 3.28664 | 2.04333 | 0.638045 | 5.59719 | -3.56552 | 6.98798 | 100 | 0 |
| B: align=none, scale_match=true | stft_l1 | 0.0573736 | 0.0514967 | 0.0212108 | 0.0325579 | 0.0834304 | 0.0152169 | 0.123225 | 100 | 0 |
| B: align=none, scale_match=true | best_lag_samples | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100 | 0 |
| B: align=none, scale_match=true | input_rms | 0.0486254 | 0.0451123 | 0.0161014 | 0.0315783 | 0.0660127 | 0.019738 | 0.104193 | 100 | 0 |
| B: align=none, scale_match=true | output_rms | 0.0483392 | 0.0450176 | 0.0156276 | 0.0308182 | 0.0658417 | 0.0198263 | 0.10747 | 100 | 0 |
| C: align=peak_xcorr, scale_match=false | waveform_snr_db | 4.87983 | 4.96649 | 1.28506 | 3.22419 | 6.46718 | 1.97727 | 8.04531 | 100 | 0 |
| C: align=peak_xcorr, scale_match=false | si_sdr_db | 3.63291 | 3.72536 | 1.73814 | 1.18953 | 5.71703 | -0.827686 | 7.5992 | 100 | 0 |
| C: align=peak_xcorr, scale_match=false | stft_l1 | 0.0572535 | 0.0513879 | 0.0212192 | 0.0318753 | 0.0833601 | 0.0151697 | 0.123225 | 100 | 0 |
| C: align=peak_xcorr, scale_match=false | best_lag_samples | 0.32 | 0 | 1.08517 | -1 | 1 | -3 | 5 | 100 | 0 |
| C: align=peak_xcorr, scale_match=false | input_rms | 0.0486254 | 0.0451123 | 0.0161014 | 0.0315783 | 0.0660127 | 0.019738 | 0.104193 | 100 | 0 |
| C: align=peak_xcorr, scale_match=false | output_rms | 0.0483392 | 0.0450176 | 0.0156276 | 0.0308182 | 0.0658417 | 0.0198263 | 0.10747 | 100 | 0 |
| D: align=peak_xcorr, scale_match=true | waveform_snr_db | 5.26917 | 5.26077 | 1.19747 | 3.64527 | 6.74856 | 2.61615 | 8.29498 | 100 | 0 |
| D: align=peak_xcorr, scale_match=true | si_sdr_db | 3.63291 | 3.72536 | 1.73814 | 1.18953 | 5.71703 | -0.827686 | 7.5992 | 100 | 0 |
| D: align=peak_xcorr, scale_match=true | stft_l1 | 0.0572535 | 0.0513879 | 0.0212192 | 0.0318753 | 0.0833601 | 0.0151697 | 0.123225 | 100 | 0 |
| D: align=peak_xcorr, scale_match=true | best_lag_samples | 0.32 | 0 | 1.08517 | -1 | 1 | -3 | 5 | 100 | 0 |
| D: align=peak_xcorr, scale_match=true | input_rms | 0.0486254 | 0.0451123 | 0.0161014 | 0.0315783 | 0.0660127 | 0.019738 | 0.104193 | 100 | 0 |
| D: align=peak_xcorr, scale_match=true | output_rms | 0.0483392 | 0.0450176 | 0.0156276 | 0.0308182 | 0.0658417 | 0.0198263 | 0.10747 | 100 | 0 |
