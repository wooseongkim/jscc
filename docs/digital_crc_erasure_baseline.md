# Digital CRC-erasure baseline

The digital baseline is a paired, no-refiner comparison against the fixed R4
medium JSCC checkpoint. It loads—not reoptimizes—the stage-1 `x_best` UEP
profile and uses only the R4 CSI-only allocator.

## Packet and resource contract

SpeechTokenizer emits 8 × 50 official RVQ indices. Each loaded codebook size
determines the index bit width (currently 1024 entries → 10 bits). Each layer
is split into five independent 10-frame packets. A packet contains 100 index
bits plus CRC-16, then is encoded/decoded with Sionna's 3GPP TS 38.212 5G LDPC
implementation.

The fixed profile includes layers with `r=1`; a layer-local QPSK transport
cannot carry its 580 CRC-protected bits. Therefore the 40 independent packet
codewords are rate-matched to 288 bits each and globally interleaved over the
unchanged 5,760 data RE / 11,520 QPSK-bit R4 pool. The physical RE and energy
profile remains exactly inherited from `selected.x_best`. This is a bit-level
interleaver, not a source-bit drop, resource expansion, learned gate, latent
refiner, or jammer-aware placement.

Failed CRC packets zero precisely their original `[layer, ten-frame]` RVQ
embedding region. Valid packets use the shared SpeechTokenizer codebook lookup
and then the existing `decode_representation()` waveform path.

## Commands

CPU smoke:

```bash
python evaluate_digital_crc_erasure.py \
  --config configs/eval_digital_crc_erasure.yaml \
  --checkpoint runs/waveform_aware_wireless/r4_si_sdr_finetune/si_sdr_medium/local_step_003000.pt \
  --max-conditions 1 --max-utterances 1 --max-realizations 1 \
  --device cpu \
  --output-dir runs/waveform_aware_wireless/r4_digital_crc_erasure/cpu_smoke
```

CUDA full paired evaluation (run by the user):

```bash
python evaluate_digital_crc_erasure.py \
  --config configs/eval_digital_crc_erasure.yaml \
  --checkpoint runs/waveform_aware_wireless/r4_si_sdr_finetune/si_sdr_medium/local_step_003000.pt \
  --device cuda --allow-long-run \
  --output-dir runs/waveform_aware_wireless/r4_digital_crc_erasure/fixed_r4_csi_only
```
