# Speech-to-Speech JSCC under Rayleigh Fading and Jamming

This project is a minimal PyTorch research baseline for speech-to-speech joint
source-channel coding (JSCC). It sends continuous or soft neural speech-codec
representations directly as learned complex channel symbols and evaluates their
recovery under Rayleigh fading, AWGN, and adversarial jamming.

The communication path is:

```text
waveform
  -> shared neural speech codec encoder
  -> continuous representation [B,L,T,D]
  -> neural JSCC encoder
  -> complex symbols [B,M] or OFDM grid [B,K,N]
  -> Rayleigh fading + AWGN + jammer
  -> neural JSCC decoder
  -> reconstructed continuous representation [B,L,T,D]
  -> shared neural speech codec decoder
  -> reconstructed waveform
```

## Scope and explicit exclusions

This is an analog/neural representation-transmission system, not a conventional
digital speech link.

- ASR and TTS are **not** the main communication path. Optional WER may be
  computed after waveform reconstruction for evaluation only.
- Model weights, codec weights, and codebooks are **not transmitted**. The
  transmitter and receiver are assumed to share the trained models.
- The system does **not** implement digital FEC, channel-coded bitstreams,
  entropy-coded packets, or bit-level RVQ index transmission.
- Codec token IDs are not converted to bits for transport. Soft-codebook mode
  maps receiver logits to an embedding expectation with temperature-scaled
  softmax. Top-k token accuracy is analysis only.

The core research question is whether speech representations can be recovered
usefully when learned complex symbols are distorted by fading and jamming.

## Implemented components

- Continuous codec interface with deterministic mock codec
- Real SpeechTokenizer continuous-embedding adapter and EnCodec placeholder
- Complex JSCC encoder and decoder
- Exact per-example average transmit-power normalization
- Flat Rayleigh fading and optional OFDM resource grids
- Barrage, narrowband, burst, and pilot jamming
- Effective SINR, JSR, and jammer-mask statistics
- Uniform, deterministic rule-based, and trained learned layer gating
- Shared 8-D channel state containing post-equalization SINR, JSR, CSI NMSE,
  a four-class rule-based jammer posterior, and unreliable-resource ratio
- Joint learned-gate/JSCC training with budget and adjacent-layer smoothness losses
- Residual Conv1D latent refiner conditioned on channel state and oracle/estimated masks
- Invertible uniform, random, and reliability-greedy OFDM resource allocation
- Optional layer-wise power allocation
- Continuous latent loss and differentiable soft-codebook projection
- Config-driven training and SNR/JSR/jammer evaluation sweeps

## Repository layout

```text
configs/
  train.yaml                 Training distribution, loss, and output settings
  eval.yaml                  Evaluation sweep and adaptation settings
src/channels/
  rayleigh.py                Flat and OFDM complex Rayleigh channel
  jammer.py                  Jammer masks, power normalization, and metrics
  pilot.py                   Pilot insertion, LS CSI estimation, NMSE, and EVM
src/models/
  jscc_encoder.py            Latent-to-complex encoder and layer adaptation
  jscc_decoder.py            Complex-to-latent decoder
  soft_codebook.py           Continuous loss and soft embedding projection
  latent_refiner.py          Residual mask-conditioned latent denoiser
  resource_allocator.py      Reliability-based analog symbol permutation
src/codecs/
  base_codec.py              Requested abstract codec interface
  mock_codec.py              Seeded dependency-free mock codec
  *_wrapper.py               Optional external-codec placeholders
src/speech_jscc/             Install-safe package and compatibility API
train_latent_jscc.py         Latent JSCC training entry point
eval_jamming.py              Channel/adaptation sweep entry point
paired_eval.py               Explicit alias for deterministic paired evaluation
tests/                       Shape, power, channel, codec, and policy tests
```

Python already has a standard-library module named `codecs`. Therefore,
application code should import the install-safe API from `speech_jscc.codecs`.
The requested source-tree files remain available under `src/codecs/`.

## Quickstart

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

pytest -q
python train_latent_jscc.py --config configs/train.yaml
python eval_jamming.py --config configs/eval.yaml
# Equivalent paired-evaluation entry point:
python paired_eval.py --config configs/eval.yaml

# Evaluate the pretrained SpeechTokenizer itself:
python eval_speechtokenizer.py

# Smoke-test SpeechTokenizer through the complete JSCC path:
python paired_eval.py --config configs/eval_speechtokenizer.yaml

# Train JSCC on frozen SpeechTokenizer continuous embeddings:
python train_latent_jscc.py --config configs/train_speechtokenizer.yaml
```

The default configuration is intentionally small and uses the mock codec, so it
does not download pretrained weights or speech datasets.

> **Warning:** untrained SpeechTokenizer-latent checkpoint MSE is path-check only.
> A SpeechTokenizer latent MSE is a performance result only when the checkpoint
> metadata identifies it as `speechtokenizer_latent_jscc` trained from waveform
> corpus latents. Random initialization, missing metadata, and the earlier smoke
> test must not be reported as model performance.

## Mini LibriSpeech manifests

This repository expects the local Mini LibriSpeech extraction to use the standard
LibriSpeech layout:

```text
data/mini_librispeech/LibriSpeech/
  train-clean-5/<speaker>/<chapter>/*.flac
  dev-clean-2/<speaker>/<chapter>/*.flac
  test-clean/<speaker>/<chapter>/*.flac
  */*/*/*.trans.txt
```

Create JSONL manifests for SpeechTokenizer/JSCC training and evaluation:

```bash
python scripts/make_librispeech_manifest.py \
  --root data/mini_librispeech/LibriSpeech \
  --out_dir manifests/mini_librispeech
```

The command writes:

```text
manifests/mini_librispeech/train.jsonl
manifests/mini_librispeech/valid.jsonl
manifests/mini_librispeech/test.jsonl
```

Each JSONL record contains `utt_id`, repo-relative `audio_path`, `speaker_id`,
`chapter_id`, `split`, `text`, `duration_sec`, `sample_rate`, and
`num_samples`. Use `--absolute-paths` only when the manifest must be tied to one
machine-specific checkout.

Training can read the JSONL manifests directly:

```yaml
data:
  train_manifest: manifests/mini_librispeech/train.jsonl
  valid_manifest: manifests/mini_librispeech/valid.jsonl
  test_manifest: manifests/mini_librispeech/test.jsonl
  sample_rate: 16000
```

`train_manifest` and `valid_manifest` are used by the current training path;
`test_manifest` is preserved for held-out evaluation and ASR/WER tooling. Mini
LibriSpeech is useful for smoke tests and pipeline validation. Final quality
claims should be repeated on full LibriSpeech `test-clean` or a larger corpus.

## Training

Run the default latent experiment:

```bash
python train_latent_jscc.py --config configs/train.yaml
```

For each batch the trainer:

1. loads precomputed codec representations or encodes synthetic waveforms;
2. samples an SNR and JSR channel condition for every batch;
3. samples one jammer type using configured probabilities;
4. encodes continuous latents into normalized complex symbols;
5. applies Rayleigh fading, jamming, and complex AWGN;
6. decodes the received symbols into continuous latents; and
7. jointly optimizes JSCC and learned-gate parameters using reconstruction,
   gate-budget, gate-smoothness, and optional transmit-power penalties.

For real SpeechTokenizer training, place WAV files under `data/speech` (or set
explicit train/validation manifests) and run `configs/train_speechtokenizer.yaml`.
The split is deterministic. SpeechTokenizer is frozen and excluded from the
optimizer; only the JSCC model, learned gate, and latent refiner are trained.
Set `data.latent_cache_dir` to cache `[L,T,D]` continuous embeddings after their
first encoding. Set it to `null` to encode waveforms on every access.

To train from precomputed codec representations, set
`data.representations_path` to a `.pt` file containing a tensor with shape
`[N,L,T,D]`, or a dictionary whose `representations` value has that shape.

The mock and SpeechTokenizer outputs are intentionally separated:

```text
artifacts/checkpoints/mock_continuous_jscc.pt
artifacts/checkpoints/speechtokenizer_latent_jscc.pt
artifacts/train_metrics.jsonl        Per-log-step metrics
artifacts/reconstructions/*.pt       Target/reconstructed latent and waveform examples
```

The JSONL log includes total, raw latent, and refined latent loss, power penalty, transmit power,
effective SINR, requested and measured JSR, jammer type, mask ratio, and
layer-wise MSE.
It also logs per-layer alpha values and the complete encoder/decoder state vectors.
The checkpoint stores the learned-gate architecture and weights under
`learned_gate`. Its metadata records `codec_name`, latent shape, sample rate,
frame rate, normalization, source type, and whether SpeechTokenizer MSE is a
valid trained-checkpoint metric.

## Evaluation experiments

Run the configured sweep:

```bash
python eval_jamming.py --config configs/eval.yaml
```

Use a different checkpoint:

```bash
python eval_jamming.py \
  --config configs/eval.yaml \
  --checkpoint artifacts/jscc.pt
```

Evaluation sweeps the Cartesian product of:

- nominal SNR values;
- requested JSR values;
- barrage, narrowband, burst, and pilot jammers; and
- available layer-adaptation modes.

Every adaptation mode in a condition uses the exact same waveform batch,
Rayleigh coefficients, jammer waveform/fading, and AWGN tensor. The deterministic
batch seed is recorded in each CSV row. Pilot symbols are inserted before the
channel, LS CSI is estimated at the receiver, and estimated-CSI equalization is
the default. Add `oracle` to `eval.equalizer_modes` for a perfect-CSI upper bound.

The comparison modes are:

- `uniform`: all codec layers are active with uniform default allocation;
- `rule_based`: activates a layer prefix using nominal `SNR - JSR` and the
  configured thresholds;
- `learned`: evaluated only if the checkpoint contains a compatible
  `learned_gate` state dictionary. Otherwise it is reported and skipped.

Evaluation outputs:

```text
artifacts/eval.csv
artifacts/plots/layer_mse.png
artifacts/plots/waveform_metrics.png
artifacts/plots/effective_sinr.png
```

The CSV contains raw weighted latent MSE, normalized training-style latent loss,
one MSE column per codec layer, waveform MSE, SI-SDR, optional STOI, effective
SINR, measured JSR, active-layer count, and jammer-mask ratio.
It also reports CSI NMSE, pilot EVM, checkpoint kind, codec name, evaluation
data source, and `metric_interpretation` (`trained_checkpoint_performance` or
`smoke_test_path_check`).
PESQ/ViSQOL, speaker similarity, and WER columns remain optional placeholder or
future-work post-reconstruction evaluators. They are not part of encoding or
transmission.

## Audio quality metrics

SI-SDR is computed by default whenever both the reference waveform and decoded
waveform are available. It reports scale-invariant waveform distortion in dB
and is included in `eval_jamming.py` CSV rows as `si_sdr_db`.

STOI is an optional intelligibility metric backed by `pystoi`. It is disabled by
default so the base test and smoke-test environment does not need the package.
Install and enable it explicitly:

```bash
python -m pip install pystoi
# or install the project extra:
python -m pip install -e '.[audio-metrics]'
```

Then set:

```yaml
eval:
  enable_stoi: true
```

and run:

```bash
python eval_jamming.py --config configs/eval_speechtokenizer.yaml
```

`stoi_available` and `stoi_error` in the CSV describe whether STOI actually ran.
For synthetic smoke-test waveforms, SI-SDR is still mechanically valid, but
STOI should not be interpreted as speech intelligibility. PESQ/ViSQOL, speaker
similarity, and WER are still optional placeholder or future-work evaluators.
ASR/WER must remain a post-reconstruction evaluation tool, not part of the
communication path.

## Codec-only SpeechTokenizer evaluation with latent perturbation

Before training JSCC, evaluate the speech codec by itself. This path does not
instantiate the JSCC encoder/decoder, does not use Rayleigh fading, does not use
wireless AWGN, and does not report JSR or effective SINR.

Clean codec reconstruction means:

```text
waveform -> SpeechTokenizer encode -> continuous latent -> SpeechTokenizer decode
```

Latent noise sensitivity adds Gaussian perturbation directly to the continuous
latent tensor before decoding. `latent_noise_snr_db` is a latent-domain
perturbation SNR, not wireless channel SNR:

```text
waveform -> SpeechTokenizer encode -> latent + Gaussian noise -> SpeechTokenizer decode
```

Run a small validation check:

```bash
python eval_codec_only.py \
  --config configs/train_speechtokenizer.yaml \
  --manifest manifests/mini_librispeech/valid.jsonl \
  --split valid \
  --max_items 20 \
  --batch_size 2 \
  --enable_latent_noise_sweep true \
  --latent_noise_snr_db 30 20 10 5 0 \
  --output_dir runs/codec_only/valid_20_noise
```

Run a larger held-out check:

```bash
python eval_codec_only.py \
  --config configs/train_speechtokenizer.yaml \
  --manifest manifests/mini_librispeech/test.jsonl \
  --split test \
  --max_items 100 \
  --batch_size 4 \
  --enable_latent_noise_sweep true \
  --latent_noise_snr_db 30 20 10 5 0 \
  --output_dir runs/codec_only/test_100_noise
```

To diagnose clean reconstruction quality, compare the official SpeechTokenizer
RVQ-code path with the JSCC-facing continuous-latent path and optionally align
metrics by peak cross-correlation:

```bash
python eval_codec_only.py \
  --config configs/train_speechtokenizer.yaml \
  --manifest manifests/mini_librispeech/test.jsonl \
  --split test \
  --max_items 100 \
  --batch_size 4 \
  --decode_mode both \
  --metric_align peak_xcorr \
  --max_lag_samples 1000 \
  --snr_scale_match true \
  --metric_zero_mean true \
  --enable_latent_noise_sweep true \
  --latent_noise_snr_db 30 20 10 5 0 \
  --output_dir runs/codec_only/test_100_noise_diag
```

For fixed baseline reporting, enable the four-protocol comparison table. It
records A: no alignment/no scale matching, B: no alignment/scale matching,
C: peak cross-correlation/no scale matching, and D: peak cross-correlation/scale
matching. The recommended main protocol is D with `metric_zero_mean=true`; also
report the `metric_align=none` rows for comparison.

```bash
python eval_codec_only.py \
  --config configs/train_speechtokenizer.yaml \
  --manifest manifests/mini_librispeech/test.jsonl \
  --split test \
  --max_items 100 \
  --batch_size 4 \
  --decode_mode both \
  --metric_align peak_xcorr \
  --max_lag_samples 1000 \
  --snr_scale_match true \
  --metric_zero_mean true \
  --protocol_comparison true \
  --output_dir runs/codec_only/test_100_protocol_diag
```

To check crop-length sensitivity, pass multiple `--waveform_samples` values:

```bash
python eval_codec_only.py \
  --config configs/train_speechtokenizer.yaml \
  --manifest manifests/mini_librispeech/test.jsonl \
  --split test \
  --max_items 100 \
  --batch_size 4 \
  --decode_mode both \
  --metric_align peak_xcorr \
  --snr_scale_match true \
  --metric_zero_mean true \
  --protocol_comparison true \
  --waveform_samples 16000 32000 48000 \
  --output_dir runs/codec_only/test_100_length_sweep
```

`--decode_mode official` uses SpeechTokenizer `encode()` and `decode(codes)`
directly. `--decode_mode continuous_sum` uses the continuous `[B,L,T,D]`
representation used by JSCC experiments. `--decode_mode both` records both
metrics and their gap. The latent-noise sweep remains based on the
continuous-sum path, because that is the path JSCC transports.

Outputs include `summary.json`, `per_utterance_metrics.csv`,
`decode_comparison_metrics.csv`, optional `baseline_protocol_comparison.csv`,
`baseline_protocol_comparison.md`, `baseline_protocol_comparison.pdf`,
`latent_noise_sweep_metrics.csv`, `report.md`, up to five WAV examples by
default, worst-SI-SDR WAV examples in `worst_samples/`, detailed outlier
artifacts in `outliers/<utt_id>/`, and plots for clean waveform SNR, SI-SDR,
per-layer latent energy, and latent-noise sweeps. These results are codec
reconstruction upper bounds and latent error tolerance references for later
JSCC experiments.

## Codec-only baseline protocol

SpeechTokenizer official `encode()` -> `decode(codes)` reconstruction and the
current `continuous_sum` reconstruction are equivalent in clean codec-only
evaluation. The JSCC source representation in this project is not a digital RVQ
bitstream; it is the continuous post-quantized RVQ codebook embedding tensor
`[B,L,T,D]`, summed across layers at the SpeechTokenizer decoder input.

Use this protocol for the main clean codec-only baseline:

```text
waveform_samples = 32000
duration = 2 seconds at 16 kHz
metric_align = peak_xcorr
snr_scale_match = true
metric_zero_mean = true
outlier threshold = SI-SDR <= -10 dB
```

Report mean, median, std, p10, p90, min, max, and outlier count. Main results
must keep all samples; outlier-excluded values are diagnostic only. The fixed
test_100 result under this protocol is:

```text
waveform SNR mean = 5.26917 dB
SI-SDR mean = 3.63291 dB
STFT L1 mean = 0.0572535
outlier count = 0 / 100
```

The 1-second setting is retained for latency-oriented JSCC experiments, not as
the representative codec-only baseline:

```text
waveform_samples = 16000
protocol D waveform SNR mean = 5.30472 dB
protocol D SI-SDR mean = 3.56680 dB
protocol D STFT L1 mean = 0.0583944
outlier count = 1 / 100
```

The known 1-second outlier is `1188-133604-0012` at
`data/mini_librispeech/LibriSpeech/test-clean/1188/133604/1188-133604-0012.flac`.
The 3-second setting gives the lowest STFT L1 in the comparison, but its SI-SDR
is lower than the 2-second setting, so 2 seconds is the main baseline. Protocol
A (`metric_align=none`, `snr_scale_match=false`) is kept only as a raw
backward-compatible reference or appendix result.

Run the final baseline directly:

```bash
python eval_codec_only.py \
  --config configs/codec_only_baseline.yaml \
  --manifest manifests/mini_librispeech/test.jsonl \
  --split test \
  --max_items 100 \
  --batch_size 4 \
  --waveform_samples 32000 \
  --metric_align peak_xcorr \
  --snr_scale_match true \
  --metric_zero_mean true \
  --protocol_comparison true \
  --output_dir runs/codec_only/test_100_2s_final_protocol
```

Or regenerate the full length/protocol comparison:

```bash
PYTHON=.venv/bin/python scripts/run_codec_protocol_comparison.sh
```

Latent-noise sweeps should use the same 2-second protocol when they are compared
against the main clean baseline. `latent_noise_snr_db` is latent-domain
perturbation SNR, not wireless SNR.

## Full WAV-to-WAV JSCC inference

`infer_jscc_wav.py` runs one input WAV through the full continuous-latent JSCC
chain and exports the reconstructed Rx waveform. Unlike `eval_jamming.py`, this
entry point writes a WAV file. Unlike `eval_speechtokenizer.py`, it includes the
JSCC encoder/decoder, Rayleigh fading, AWGN, jammer, pilot CSI estimation, and
equalization.

Mock smoke test:

```bash
python infer_jscc_wav.py \
  --config configs/eval.yaml \
  --input data/example.wav \
  --output artifacts/inference/rx_mock.wav \
  --snr-db 8 \
  --jsr-db 0 \
  --jammer pilot \
  --adaptation-mode uniform \
  --allocation-mode uniform \
  --save-pt artifacts/inference/rx_mock.pt \
  --metrics-json artifacts/inference/rx_mock_metrics.json
```

SpeechTokenizer checkpoint use:

```bash
python infer_jscc_wav.py \
  --config configs/eval_speechtokenizer.yaml \
  --checkpoint artifacts/checkpoints/speechtokenizer_latent_jscc.pt \
  --input data/speech/example.wav \
  --output artifacts/inference/rx_speechtokenizer.wav \
  --snr-db 5 \
  --jsr-db 0 \
  --jammer pilot \
  --adaptation-mode learned_gate \
  --allocation-mode reliability_greedy \
  --refiner-mode refiner_estimated_mask
```

The CLI prints JSON metrics to stdout and can also save them with
`--metrics-json`; the metrics include SI-SDR and optional STOI when
`eval.enable_stoi` is true and `pystoi` is installed. Use `--save-pt` to store
source waveform, target latent, raw/final reconstructed latents, decoded
waveform, transmitted/received complex resources, estimated channel, jammer
tensors, pilot/jammer masks, channel-state vectors, layer gates, and the
metrics dictionary. If `learned_gate` or a refiner mode is requested, the
checkpoint must contain the corresponding module state.

## Configuration reference

### Shared model and codec settings

| Key | Description |
| --- | --- |
| `seed` | PyTorch and Python random seed |
| `device` | `auto`, `cpu`, or a PyTorch device such as `cuda:0` |
| `model.layers` | Number of codec representation layers `L` |
| `model.frames` | Representation frames `T` |
| `model.latent_dim` | Embedding dimension `D` |
| `model.channel_uses` | Complex symbol count `M`; model APIs also accept `(K,N)` grids |
| `model.channel_state_dim` | Set to 8 for `[SINR, JSR, NMSE, jammer posterior(4), mask ratio]` |
| `model.hidden_dim` | JSCC encoder/decoder hidden width |
| `model.target_power` | Required mean complex-symbol power `E[|x|^2]` |
| `codec.waveform_samples` | Mock-codec input/output waveform length |

### Training settings

| Key | Description |
| --- | --- |
| `data.representations_path` | Optional precomputed `[N,L,T,D]` tensor |
| `data.waveform_dir` / manifests | Real waveform corpus and deterministic train/val source |
| `data.val_fraction` | Deterministic validation fraction when using a directory |
| `data.latent_cache_dir` | Optional split-specific continuous-latent disk cache |
| `channel.snr_db_range` | Uniform training SNR range `[min,max]` |
| `channel.jsr_db_range` | Uniform training JSR range `[min,max]` |
| `channel.jammer_probabilities` | Nonnegative jammer probabilities summing to one |
| `channel.jammed_fraction` | Resource fraction used by sparse jammers |
| `channel.pilot_spacing` | Default pilot-mask spacing |
| `channel.pilot_time_spacing` | OFDM time-axis pilot spacing |
| `train.steps` | Optimizer steps |
| `train.batch_size` | Examples per step |
| `train.learning_rate` | Adam learning rate |
| `train.layer_weights` | One latent-MSE weight per codec layer |
| `train.latent_normalization` | `none`, `per_layer_power`, or `global_power` |
| `train.power_penalty_weight` | Multiplier for the average-power penalty |
| `train.lambda_budget` | Weight for `mean(sum(alpha))` |
| `train.lambda_smooth` | Weight for adjacent-layer alpha total variation |
| `train.learned_gate_hidden_dim` | Learned gate MLP hidden width |
| `train.transmitter_csi` | Feed pilot-derived channel state back to encoder and gate |
| `train.lambda_refine` | Weight for refined-latent reconstruction loss |
| `train.refiner_hidden_dim` | Residual Conv1D refiner width |
| `train.refiner_mask_mode` | `oracle` or reliability-estimated training mask |
| `train.allocation_mode` | Training allocation: uniform, random, or reliability-greedy |
| `train.layer_importance_order` | Codec-layer priority, highest importance first |
| `train.gradient_clip_norm` | Optional global gradient-norm limit |
| `train.log_every` | Metric and reconstruction logging interval |
| `train.metrics_jsonl` | JSONL metric path |
| `train.reconstruction_dir` | Saved reconstruction-example directory |
| `train.checkpoint` | Output checkpoint path |

### Evaluation settings

| Key | Description |
| --- | --- |
| `channel.snr_db` | SNR sweep values |
| `channel.jsr_db` | JSR sweep values |
| `channel.jammer_types` | Jammer types included in the sweep |
| `eval.batches` | Monte Carlo batches per condition |
| `eval.batch_size` | Examples per Monte Carlo batch |
| `eval.adaptation_modes` | `uniform`, `rule_based`, and/or checkpoint-backed `learned_gate` |
| `eval.transmitter_csi` | Feed pilot-derived state to encoder and adaptation gate |
| `eval.allocation_modes` | Paired uniform, random, and reliability-greedy allocation baselines |
| `eval.refiner_modes` | No-refiner, oracle-mask, and estimated-mask ablations |
| `eval.unreliable_fraction` | Fraction marked unreliable for estimated-mask refinement |
| `eval.paired_seed` | Base seed for mode-invariant evaluation realizations |
| `eval.equalizer_modes` | `estimated` and optional perfect-CSI `oracle` |
| `eval.rule_gate_thresholds_db` | `L-1` nondecreasing layer activation thresholds |
| `eval.layer_weights` | Layer weights for aggregate latent MSE |
| `eval.enable_stoi` | Enables optional `pystoi` STOI calculation after waveform reconstruction |
| `eval.output_csv` | Result table path |
| `eval.plot_dir` | Summary plot directory |
| `eval.enable_optional_wer` | Flags optional post-reconstruction WER; an external evaluator is still required |

## Codec integration

`BaseCodec` defines:

```python
encode_waveform(waveform)                 # [B,S] -> [B,L,T,D]
decode_representation(representation)     # [B,L,T,D] -> [B,S]
get_codebook()                            # optional [L,K,D] embeddings
representation_shape                     # (L,T,D)
```

`SpeechTokenizerWrapper` loads the official SpeechTokenizer config/checkpoint,
calls `forward_feature()` for all requested RVQ layers, and exposes the resulting
quantized embeddings as `[B,L,T,D]`. Decoding sums the continuous layer
embeddings and calls the pretrained waveform decoder directly. For codec-only
diagnostics, `official_reconstruct_waveform()` also exposes the upstream
`encode()` -> `decode(codes)` reconstruction path. RVQ indices are used only by
standalone codec diagnostics and are never transmitted by JSCC.

Install the upstream repository and its inference dependency, then provide
`codec.type: speechtokenizer`, `config_path`, `checkpoint_path`,
`waveform_samples`, and `n_q`. The `EnCodecWrapper` remains a placeholder.

## Baseline assumptions

- Evaluation uses pilot LS estimated CSI by default; perfect CSI is an optional
  oracle baseline.
- Fading is flat for `[B,M]` input. `[B,K,N]` enables an OFDM resource grid.
- JSR normalization is measured over all resources, so sparse jammers
  concentrate their energy on active mask elements.
- The mock codec is for pipeline and regression testing, not speech-quality
  benchmarking.
- Perceptual speech metrics require separate optional models or packages and are
  deliberately outside the communication pipeline.
