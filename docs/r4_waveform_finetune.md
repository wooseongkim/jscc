# R4 waveform-connected fine-tuning

This stage fine-tunes only the accepted Conv-Conformer JSCC encoder and decoder.
SpeechTokenizer stays frozen, but its waveform decoder is executed with input
autograd enabled. The physical forward uses the real R4 time-domain OFDM path,
comb pilot LS estimator, delayed-CSI global triplet allocator, and current-CSI
coherent MRC.

## Implementation audit

The accepted R4 evaluator previously held the physical equations directly in
`evaluate_repetition3_mrc.py` under evaluation-wide `torch.no_grad()`. Hashing and
CSI-report creation legitimately detach logging/mapping metadata. Reconstructed
latent decoding for training must instead use
`decode_frozen_representation_with_gradient`; it has no reconstructed-latent
detach and does not place the codec decoder under `no_grad`.

The shared differentiable forward is `R4WaveformForward` in
`src/speech_jscc/training/r4_waveform_finetune.py`. It directly reuses:

- `allocate_global_balanced_triplets`;
- `insert_physical_pilots`;
- `modulate_tti` / `demodulate_tti`;
- `apply_tti_multipath`;
- `estimate_comb_dft_ls`;
- `coherent_mrc`;
- `build_observable_receiver_state_v1`.

MRC uses current estimated CSI, `conj(a*h_hat)*y/sigma²` in the numerator,
`|a*h_hat|²/sigma²` in the denominator, and only the configured denominator
epsilon. It returns the original 1,920-symbol source order. The transmitter
allocator sees only a detached previous-TTI report; Stage A deliberately uses
the deterministic bootstrap map.

## Commands

Dry run:

```bash
bash scripts/run_r4_waveform_finetune_external.sh --device cuda --dry-run
```

Three-step smoke:

```bash
bash scripts/run_r4_waveform_finetune_external.sh \
  --device cuda --smoke-steps 3 \
  --output-dir runs/waveform_aware_wireless/r4_repetition3_mrc_finetune_smoke \
  --overwrite
```

Explicit waveform-gradient verification:

```bash
python train_r4_waveform_finetune.py \
  --config configs/train_r4_waveform_finetune.yaml \
  --device cuda --batch-size 1 --smoke-steps 1 \
  --verify-waveform-gradient \
  --output-dir runs/waveform_aware_wireless/r4_waveform_gradient_check \
  --overwrite
```

Full external 20,000-step training:

```bash
bash scripts/run_r4_waveform_finetune_external.sh --device cuda
```

Exact resume:

```bash
bash scripts/run_r4_waveform_finetune_external.sh \
  --device cuda \
  --resume runs/waveform_aware_wireless/r4_repetition3_mrc_finetune/last.pt
```

Light validation is executed by the trainer every 250 steps; full validation is
executed every 1,000 steps. Both use the fixed paths and seeds saved in
`validation_manifest.json`.

Final 64×2×3 evaluation:

```bash
bash scripts/run_r4_waveform_finetune_eval_external.sh \
  --device cuda \
  --checkpoint runs/waveform_aware_wireless/r4_repetition3_mrc_finetune/best_clean_gate.pt
```

If no checkpoint is eligible for `best_clean_gate.pt`, substitute
`best_5db_si_sdr.pt`. The final summary labels that choice as a fallback and does
not claim a clean-gate pass.
