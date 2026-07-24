# R4 Waveform-Aware Clean-Channel Fine-Tuning Design

## Goal

Fine-tune the accepted Conv-Conformer JSCC encoder and decoder on the fixed R4
three-copy clean wireless path so the nominal-5-dB waveform gate passes without
changing SpeechTokenizer, the JSCC architecture, physical allocation, power
allocation, coherent MRC, or any production G/J path.

## Immutable inputs

- Initialization checkpoint:
  `runs/waveform_aware_wireless/clean_channel_training/best_waveform_si_sdr.pt`
- R4: 512-point FFT, 360 active subcarriers, 30 kHz SCS, 28 OFDM symbols,
  36-sample CP, 10.8 MHz occupied bandwidth.
- CF-2 source interface: 1,920 complex symbols.
- Physical transmissions: three copies and 5,760 total data RE.
- Delayed-CSI global balanced triplet allocation and bounded triplet/branch power.
- Current receiver CSI and the existing unbiased coherent MRC equation.
- Frozen SpeechTokenizer encoder, quantizer/codebooks, and waveform decoder.

## Architecture and data flow

A focused shared R4 forward function will be used by training, held-out
validation, and final evaluation:

1. Encode real waveform to `[B,8,50,1024]` with frozen SpeechTokenizer.
2. Encode to 1,920 complex CF-2 source symbols.
3. Allocate three physical copies using only a causally available, detached
   previous-TTI receiver CSI report. TTI 0 uses deterministic bootstrap.
4. Insert pilots, perform unitary time-domain OFDM, apply the correlated sparse
   six-tap channel, and add time-domain complex AWGN.
5. Estimate current CSI from pilots and coherently combine raw observations with
   the unchanged MRC equation.
6. Restore `[B,1920]`, decode the latent layers, sum the exact decoder-input
   layers, and decode waveform through frozen SpeechTokenizer.

The discrete allocation map and delayed CSI report are not differentiated.
Gradients flow through transmitted values, OFDM, channel, CSI-conditioned MRC,
the JSCC decoder, and the frozen codec waveform decoder into both JSCC modules.
Codec parameters remain `requires_grad=False` and receive no gradients.

For tractable training, all items in an optimizer batch share one causal channel
trajectory and allocation map for that TTI, while AWGN tensors remain independent
per item. The next report is derived from the current receiver estimate and
detached before it becomes transmitter input. This preserves the causal FDD
contract without requiring a distinct allocator state per utterance.

## Training protocol

- Total steps: 20,000.
- Batch size: default 4, configurable.
- Validation cadence: every 250 steps.
- SNR categorical distribution:
  - 5 dB: 0.50
  - 10 dB: 0.30
  - 15 dB: 0.20
- Optimizer and gradient clipping inherit the accepted checkpoint configuration.
- Only JSCC encoder and decoder parameters are passed to the optimizer.

Curriculum:

- Stage 1, steps 1–4,000: per-layer NMSE and summed-latent NMSE.
- Stage 2, steps 4,001–12,000: retain latent losses and gradually ramp the
  existing multi-resolution STFT weight from zero to its configured small value.
- Stage 3, steps 12,001–20,000: retain prior terms and gradually ramp a low-weight
  differentiable negative SI-SDR term.

Each loss value, effective weight, and component gradient norm is logged.
Gradient clipping remains enabled. A finite-value failure writes diagnostic
metadata and stops with nonzero status.

## Validation and checkpoint selection

Held-out validation uses fixed utterance IDs, fixed utterance order, and fixed
channel/noise seeds at 5, 10, and 15 dB. It is disjoint from each optimizer
batch. Validation records latent, waveform, physical, energy, and finite metrics.

Checkpoints:

- `best_5db_si_sdr.pt`: highest 5 dB delta SI-SDR.
- `best_clean_gate.pt`: highest minimum normalized margin across the three 5 dB
  waveform gates. Its metadata explicitly records whether all gates passed.
- `best_validation_average.pt`: highest mean delta SI-SDR across 5/10/15 dB.
- `last.pt`: final optimizer state.

The final full evaluation uses `best_clean_gate.pt` if its gate metadata is true.
Otherwise it uses `best_5db_si_sdr.pt` and reports that no training checkpoint
passed held-out clean gates. No checkpoint is called a passing checkpoint unless
the stored gate fields prove it.

## Final evaluation

Run the existing fixed protocol:

- 64 unseen-speaker utterances.
- Two deterministic realizations.
- 5, 10, and 15 dB.
- The unchanged R4 physical path.
- No jammer.

Report post-MRC SINR, aggregate and summed latent NMSE, SI-SDR, delta SI-SDR,
waveform SNR, delta waveform SNR, and STFT ratio per SNR. Compare 10 and 15 dB
against the accepted pre-fine-tuning R4 summary. A material regression is defined
as more than 0.5 dB loss in delta SI-SDR or delta waveform SNR, or more than 0.05
absolute increase in STFT ratio.

Channel-free validation reuses the accepted fixed unseen-speaker set and reports
the same waveform gates. It is a regression diagnostic and must not silently
replace the R4 checkpoint-selection metric.

The required 5 dB gate remains:

- delta SI-SDR >= -1.0 dB;
- delta waveform SNR >= -1.0 dB;
- STFT ratio <= 1.20.

## Outputs

Use a new root:

`runs/waveform_aware_wireless/r4_repetition3_mrc_finetune/`

It contains resolved configuration, environment and command records, metrics
JSONL, validation summaries, gradient diagnostics, all four checkpoints, a
channel-free regression, the 64×2×3 final evaluation, and a final comparison
against the accepted pre-fine-tuning R4 evidence.

External scripts refuse existing output directories unless `--overwrite` is
given, support exact `--resume`, preserve stdout/stderr with `tee`, and return
nonzero on failure.

## Tests

Tests must prove:

- exact 50/30/20 categorical SNR sampling under deterministic seeds;
- TTI 0 bootstrap and one-TTI delayed CSI causality;
- no current/future CSI is available to the allocator;
- the training forward retains the exact 5,760-RE and 5,760-energy contracts;
- waveform loss gives nonzero gradients to JSCC encoder and decoder;
- codec parameter gradients remain absent;
- staged weights ramp in the intended order;
- checkpoint metric names and selection rules are unambiguous;
- validation IDs and seeds are deterministic and held out;
- resume restores model, optimizer, scheduler/curriculum step, RNG, and CSI state;
- final evaluation uses the unchanged R4 path;
- existing R4, CF-2, FDD, channel-free, and production G/J tests remain passing.

## Non-goals

No SpeechTokenizer updates, model redesign, allocation/MRC change, retraining of
the physical layer, layer ablation, unequal repetition, jammer support, or
production G/J modification is permitted.
