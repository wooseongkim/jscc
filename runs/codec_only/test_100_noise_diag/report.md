# Codec-only SpeechTokenizer Baseline Diagnostics

- manifest: `manifests/mini_librispeech/test.jsonl`
- split: `test`
- decode_mode: `both`
- metric_align: `peak_xcorr`

## Clean official reconstruction summary

| metric | mean | median | p10 | p90 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| waveform_mse | 0.000886736 | 0.000676042 | 0.000265857 | 0.00174116 | 2.16756e-06 | 0.00686373 |
| waveform_l1 | 0.0152884 | 0.0148047 | 0.00786467 | 0.0243379 | 0.0010641 | 0.043767 |
| waveform_snr_db | 5.30472 | 5.1378 | 3.6817 | 7.18713 | 0.316732 | 9.08836 |
| si_sdr_db | 3.5668 | 3.5491 | 1.2526 | 6.26783 | -11.2098 | 8.51667 |
| stft_l1 | 0.0583944 | 0.0542556 | 0.0327295 | 0.0874616 | 0.00387945 | 0.182775 |

## Clean continuous_sum reconstruction summary

| metric | mean | median | p10 | p90 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| waveform_mse | 0.000886736 | 0.000676042 | 0.000265857 | 0.00174116 | 2.16756e-06 | 0.00686373 |
| waveform_l1 | 0.0152884 | 0.0148047 | 0.00786467 | 0.0243379 | 0.0010641 | 0.043767 |
| waveform_snr_db | 5.30472 | 5.1378 | 3.6817 | 7.18713 | 0.316732 | 9.08836 |
| si_sdr_db | 3.5668 | 3.5491 | 1.2526 | 6.26783 | -11.2098 | 8.51667 |
| stft_l1 | 0.0583944 | 0.0542556 | 0.0327295 | 0.0874616 | 0.00387945 | 0.182775 |

## Official vs continuous_sum metric gap

| metric | mean | median | p10 | p90 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| waveform_mse | 1.45519e-12 | 0 | -3.0559e-11 | 3.20142e-11 | -2.32831e-10 | 2.32831e-10 |
| waveform_l1 | 3.72529e-11 | 0 | 0 | 0 | -1.86265e-09 | 1.86265e-09 |
| waveform_snr_db | 5.48363e-08 | 0 | 0 | 4.76837e-07 | -9.53674e-07 | 9.53674e-07 |
| si_sdr_db | -4.85778e-08 | 0 | -4.76837e-07 | 2.6226e-07 | -1.93715e-06 | 1.54972e-06 |
| stft_l1 | 5.86733e-10 | 0 | -9.31323e-11 | 3.72529e-09 | -1.49012e-08 | 1.49012e-08 |

## Alignment off vs alignment on metric gap

| metric | mean | median | p10 | p90 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| waveform_mse | -9.27901e-05 | -2.9739e-05 | -0.000193671 | 0 | -0.000918072 | 0 |
| waveform_l1 | -0.000400361 | -0.00022532 | -0.000831312 | 3.4877e-06 | -0.00337577 | 0.000307824 |
| waveform_snr_db | 0.421381 | 0.22081 | 0 | 1.02924 | 0 | 3.17936 |
| si_sdr_db | 0.685601 | 0.364022 | 0 | 1.58909 | 0 | 5.42843 |
| stft_l1 | -0.000211743 | 0 | -0.000865833 | 0.000243984 | -0.00311747 | 0.000662155 |

## Worst sample list

| utt_id | used_decode_mode | si_sdr_db | waveform_snr_db | stft_l1 | audio_path |
|---|---|---:|---:|---:|---|
| 1188-133604-0012 | both | -11.2098 | 0.316732 | 0.0114114 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1188/133604/1188-133604-0012.flac` |
| 1089-134691-0022 | both | -1.09027 | 2.4993 | 0.0827367 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1089/134691/1089-134691-0022.flac` |
| 1089-134686-0022 | both | -1.01841 | 2.53098 | 0.0462262 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1089/134686/1089-134686-0022.flac` |
| 1089-134691-0005 | both | -0.952081 | 2.56032 | 0.0480183 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1089/134691/1089-134691-0005.flac` |
| 1089-134691-0002 | both | -0.308663 | 2.85872 | 0.0660615 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1089/134691/1089-134691-0002.flac` |
| 1089-134691-0017 | both | 0.320509 | 3.17352 | 0.0874106 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1089/134691/1089-134691-0017.flac` |
| 1089-134686-0018 | both | 0.611876 | 3.32701 | 0.0392923 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1089/134686/1089-134686-0018.flac` |
| 1089-134686-0001 | both | 0.911914 | 3.49019 | 0.0386072 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1089/134686/1089-134686-0001.flac` |
| 1188-133604-0026 | both | 0.981781 | 3.52886 | 0.0543489 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1188/133604/1188-133604-0026.flac` |
| 1188-133604-0005 | both | 1.07045 | 3.57834 | 0.0369722 | `/home/mike/jscc/data/mini_librispeech/LibriSpeech/test-clean/1188/133604/1188-133604-0005.flac` |
