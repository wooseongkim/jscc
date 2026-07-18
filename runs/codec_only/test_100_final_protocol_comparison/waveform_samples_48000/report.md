# Codec-only SpeechTokenizer Baseline Diagnostics

- manifest: `manifests/mini_librispeech/test.jsonl`
- split: `test`
- decode_mode: `both`
- metric_align: `peak_xcorr`

## Clean official reconstruction summary

| metric | mean | median | p10 | p90 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| waveform_mse | 0.000860899 | 0.000663619 | 0.000385575 | 0.00142782 | 0.000130896 | 0.00513596 |
| waveform_l1 | 0.0145287 | 0.0134112 | 0.00889862 | 0.0200505 | 0.00522327 | 0.041283 |
| waveform_snr_db | 5.19098 | 5.19564 | 3.82807 | 6.54061 | 2.72007 | 7.88301 |
| si_sdr_db | 3.54288 | 3.63231 | 1.50566 | 5.45166 | -0.601317 | 7.11135 |
| stft_l1 | 0.0542678 | 0.0503808 | 0.0343298 | 0.0762197 | 0.0193234 | 0.144088 |

## Clean continuous_sum reconstruction summary

| metric | mean | median | p10 | p90 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| waveform_mse | 0.000860899 | 0.000663619 | 0.000385575 | 0.00142782 | 0.000130896 | 0.00513596 |
| waveform_l1 | 0.0145287 | 0.0134112 | 0.00889862 | 0.0200505 | 0.00522327 | 0.041283 |
| waveform_snr_db | 5.19098 | 5.19564 | 3.82807 | 6.54061 | 2.72007 | 7.88301 |
| si_sdr_db | 3.54288 | 3.63231 | 1.50566 | 5.45166 | -0.601317 | 7.11135 |
| stft_l1 | 0.0542678 | 0.0503808 | 0.0343298 | 0.0762197 | 0.0193234 | 0.144088 |

## Official vs continuous_sum metric gap

| metric | mean | median | p10 | p90 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| waveform_mse | 5.82077e-13 | 0 | 0 | 0 | 0 | 5.82077e-11 |
| waveform_l1 | 1.86265e-11 | 0 | 0 | 0 | -1.86265e-09 | 1.86265e-09 |
| waveform_snr_db | 1.43051e-08 | 0 | 0 | 0 | -4.76837e-07 | 9.53674e-07 |
| si_sdr_db | 1.78814e-08 | 0 | 0 | 0 | -7.15256e-07 | 1.19209e-06 |
| stft_l1 | -1.49012e-10 | 0 | 0 | 0 | -7.45058e-09 | 7.45058e-09 |

## Alignment off vs alignment on metric gap

| metric | mean | median | p10 | p90 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| waveform_mse | -4.71307e-05 | -1.2059e-05 | -0.00012243 | 0 | -0.000882855 | 0 |
| waveform_l1 | -0.000139838 | 0 | -0.000451567 | 0 | -0.0020177 | 0.00036566 |
| waveform_snr_db | 0.212875 | 0.0765975 | 0 | 0.682429 | 0 | 1.48206 |
| si_sdr_db | 0.328805 | 0.112646 | 0 | 0.933914 | 0 | 3.26107 |
| stft_l1 | -4.1453e-05 | 0 | -0.000234236 | 0.00010728 | -0.00109003 | 0.000210546 |

## Worst sample list

| utt_id | used_decode_mode | si_sdr_db | waveform_snr_db | stft_l1 | audio_path |
|---|---|---:|---:|---:|---|
| 1089-134691-0005 | both | -0.601317 | 2.72007 | 0.0536578 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1089/134691/1089-134691-0005.flac` |
| 1188-133604-0005 | both | 0.259112 | 3.14171 | 0.0241253 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1188/133604/1188-133604-0005.flac` |
| 1089-134691-0022 | both | 0.828122 | 3.44408 | 0.0793587 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1089/134691/1089-134691-0022.flac` |
| 1089-134691-0001 | both | 0.937745 | 3.50446 | 0.0455466 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1089/134691/1089-134691-0001.flac` |
| 1089-134686-0030 | both | 1.18848 | 3.64507 | 0.0954121 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1089/134686/1089-134686-0030.flac` |
| 1089-134691-0002 | both | 1.20055 | 3.65193 | 0.0720751 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1089/134691/1089-134691-0002.flac` |
| 1089-134691-0023 | both | 1.21589 | 3.66067 | 0.0802412 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1089/134691/1089-134691-0023.flac` |
| 1089-134686-0016 | both | 1.32597 | 3.72371 | 0.0544615 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1089/134686/1089-134686-0016.flac` |
| 1188-133604-0012 | both | 1.33072 | 3.7264 | 0.0385117 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1188/133604/1188-133604-0012.flac` |
| 1089-134691-0024 | both | 1.45627 | 3.79919 | 0.0854797 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1089/134691/1089-134691-0024.flac` |
