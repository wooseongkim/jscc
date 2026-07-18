# SpeechTokenizer Codec-only Baseline Protocol

## Recommended protocol

- waveform_samples = 32000
- duration = 2 seconds at 16 kHz
- metric_align = peak_xcorr
- snr_scale_match = true
- metric_zero_mean = true
- main result keeps all samples and always reports SI-SDR <= -10 dB outlier count

## Main clean codec-only result

- evaluated waveform_samples = 48000
- evaluated duration = 3 seconds at 16000 Hz
- waveform SNR mean = 5.19098 dB
- SI-SDR mean = 3.54288 dB
- STFT L1 mean = 0.0542678
- outlier count = 0 / 100

## Reference 1-second protocol

- waveform_samples = 16000
- D protocol result from the fixed comparison run:
  - waveform SNR mean = 5.30472 dB
  - SI-SDR mean = 3.56680 dB
  - STFT L1 mean = 0.0583944
  - outlier count = 1 / 100

## Interpretation

- official SpeechTokenizer reconstruction and continuous_sum reconstruction are equivalent in clean codec-only evaluation.
- continuous_sum is not pre-quantization latent; it is post-quantization RVQ codebook embedding summed across layers.
- The advantage of continuous_sum for JSCC is not higher clean SDR, but graceful degradation under channel/latent perturbation.
- 2-second crop is selected as the stable main codec-only baseline because it removes the 1-second outlier and gives the best SI-SDR among tested stable settings.
- `latent_noise_snr_db` is latent-domain perturbation SNR, not wireless SNR.

## Outlier policy

- outlier threshold: SI-SDR <= -10 dB
- main result does not remove outliers
- 1-second condition keeps one known outlier: `1188-133604-0012` at `data/mini_librispeech/LibriSpeech/test-clean/1188/133604/1188-133604-0012.flac`
- 2-second condition has 0 outliers in the current test_100 protocol comparison
