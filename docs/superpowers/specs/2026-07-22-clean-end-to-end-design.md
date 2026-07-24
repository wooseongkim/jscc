# Clean End-to-End JSCC Diagnostic Design

## Scope

Freeze J1–J5 evidence and establish a clean speech upper bound in this order:
codec identity, existing-J5 direct bypass, ideal OFDM identity, clean impairment
ladder, then fresh channel-free Conv-Conformer training. No jammer curriculum,
waveform training loss, receiver redesign, or architecture change is permitted.

## Architecture

`src/evaluation/clean_end_to_end.py` owns pure tensor/path checks and metric
aggregation. `diagnose_clean_end_to_end.py` runs identity and existing-checkpoint
phases. `eval_clean_channel_ladder.py` runs paired C0–C4 evaluation.
`train_channel_free_conv_conformer.py` reuses the accepted G0 direct bypass and
real manifest/cache sampling while selecting independent best-latent and
best-waveform checkpoints. Shell launchers isolate long CUDA runs.

## Scientific contracts

- Real Mini LibriSpeech manifest/cache only; test split rejected.
- Codec, preprocessing, latent `[8,50,1024]`, Conv-Conformer, and 1920-symbol
  bottleneck remain unchanged.
- Neutral state is produced through the existing channel-state implementation.
- Identity failures stop later phases.
- Latent and waveform evidence are always reported separately.
- Existing J1–J5 summaries/checkpoints are read-only.
- Optional perceptual metrics are explicitly unavailable rather than fabricated.

## Outputs and classification

Each phase writes JSON/CSV/WAV evidence beneath
`runs/stage1_conv_conformer_clean_end_to_end/`. Root classification follows the
user-specified taxonomy and remains `INCONCLUSIVE` until required external runs
exist. J1–J5 remain latent-only evidence until the clean end-to-end baseline is
established.
