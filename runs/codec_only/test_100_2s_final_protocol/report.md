# Codec-only SpeechTokenizer Baseline Diagnostics

- manifest: `manifests/mini_librispeech/test.jsonl`
- split: `test`
- decode_mode: `continuous_sum`
- metric_align: `peak_xcorr`

## Clean official reconstruction summary

| metric | mean | median | p10 | p90 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| waveform_mse | 0.000868707 | 0.000701251 | 0.000302345 | 0.00160631 | 0.000156947 | 0.00398414 |
| waveform_l1 | 0.0150534 | 0.0138617 | 0.00919605 | 0.0228908 | 0.00488306 | 0.031438 |
| waveform_snr_db | 5.26917 | 5.26077 | 3.64527 | 6.74856 | 2.61615 | 8.29498 |
| si_sdr_db | 3.63291 | 3.72536 | 1.18953 | 5.71703 | -0.827686 | 7.5992 |
| stft_l1 | 0.0572535 | 0.0513879 | 0.0318753 | 0.0833601 | 0.0151697 | 0.123225 |

## Clean continuous_sum reconstruction summary

| metric | mean | median | p10 | p90 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| waveform_mse | 0.000868707 | 0.000701251 | 0.000302345 | 0.00160631 | 0.000156947 | 0.00398414 |
| waveform_l1 | 0.0150534 | 0.0138617 | 0.00919605 | 0.0228908 | 0.00488306 | 0.031438 |
| waveform_snr_db | 5.26917 | 5.26077 | 3.64527 | 6.74856 | 2.61615 | 8.29498 |
| si_sdr_db | 3.63291 | 3.72536 | 1.18953 | 5.71703 | -0.827686 | 7.5992 |
| stft_l1 | 0.0572535 | 0.0513879 | 0.0318753 | 0.0833601 | 0.0151697 | 0.123225 |

## Official vs continuous_sum metric gap

| metric | mean | median | p10 | p90 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| waveform_mse | 0 | 0 | 0 | 0 | 0 | 0 |
| waveform_l1 | 0 | 0 | 0 | 0 | 0 | 0 |
| waveform_snr_db | 0 | 0 | 0 | 0 | 0 | 0 |
| si_sdr_db | 0 | 0 | 0 | 0 | 0 | 0 |
| stft_l1 | 0 | 0 | 0 | 0 | 0 | 0 |

## Alignment off vs alignment on metric gap

| metric | mean | median | p10 | p90 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| waveform_mse | -5.73346e-05 | -8.9558e-06 | -0.000207022 | 0 | -0.000411207 | 0 |
| waveform_l1 | -0.000171849 | 0 | -0.000728414 | 0.00010452 | -0.00134108 | 0.000451919 |
| waveform_snr_db | 0.274709 | 0.0808239 | 0 | 0.830374 | 0 | 2.06356 |
| si_sdr_db | 0.448238 | 0.105818 | 0 | 1.24924 | 0 | 4.7582 |
| stft_l1 | -0.00012016 | 0 | -0.000525736 | 0.000155145 | -0.00170936 | 0.000383049 |

## Worst sample list

| utt_id | used_decode_mode | si_sdr_db | waveform_snr_db | stft_l1 | audio_path |
|---|---|---:|---:|---:|---|
| 1089-134691-0001 | continuous_sum | -0.827686 | 2.61615 | 0.0482617 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1089/134691/1089-134691-0001.flac` |
| 1188-133604-0005 | continuous_sum | -0.0919556 | 2.96453 | 0.0283285 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1188/133604/1188-133604-0005.flac` |
| 1089-134691-0022 | continuous_sum | 0.505215 | 3.27026 | 0.0815854 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1089/134691/1089-134691-0022.flac` |
| 1188-133604-0026 | continuous_sum | 0.539355 | 3.28832 | 0.0417667 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1188/133604/1188-133604-0026.flac` |
| 1089-134686-0018 | continuous_sum | 0.579658 | 3.30981 | 0.0462969 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1089/134686/1089-134686-0018.flac` |
| 1089-134691-0023 | continuous_sum | 0.648909 | 3.34687 | 0.0829564 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1089/134691/1089-134691-0023.flac` |
| 1089-134686-0036 | continuous_sum | 0.801441 | 3.42952 | 0.0697382 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1089/134686/1089-134686-0036.flac` |
| 1188-133604-0012 | continuous_sum | 1.01401 | 3.54678 | 0.027578 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1188/133604/1188-133604-0012.flac` |
| 1089-134686-0025 | continuous_sum | 1.03243 | 3.55714 | 0.0558073 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1089/134686/1089-134686-0025.flac` |
| 1089-134686-0027 | continuous_sum | 1.16114 | 3.62957 | 0.0836956 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1089/134686/1089-134686-0027.flac` |
