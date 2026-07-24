# Channel-Free Conv-Conformer Implementation Audit

Date: 2026-07-23

This audit describes executable local code, not earlier design documents.

## Representation and symbol mapping

`SpeechTokenizerWrapper.encode_waveform()` calls the frozen model's
`forward_feature()` under `torch.no_grad()` and stacks eight tensors as
`[B,8,50,1024]`. Target extraction is intentionally nondifferentiable because
SpeechTokenizer is frozen.

`ConvConformerEncoder` applies:

1. per-frame `LayerNorm(1024)` and shared `Linear(1024,d_model)`;
2. a learned RVQ-layer embedding;
3. shared observable-state FiLM;
4. a shared depthwise local temporal convolution;
5. one shared temporal Conformer stack over `[B*8,50,d_model]`;
6. one optional shared cross-layer mixer over the eight RVQ layers;
7. a temporal symbol layout;
8. eight layer-specific symbol heads.

The historical/default layout linearly interpolates 50 features to
`symbol_frames=30`. Each layer-specific head emits 16 real values per frame,
viewed as eight complex values. Thus `30*8=240` complex symbols per layer and
`8*240=1920` total.

The encoder normalizes each branch and then the concatenated output with the
existing complex-power normalization. Uniform power allocation is used when no
allocation is supplied.

## Decoder mapping

The historical decoder partitions `[B,1920]` in layer-major order as
`[B,8,30,8]`, converts complex values to 16 real channels, applies a shared
projection and decoder Conformer stack, then linearly interpolates 30 temporal
positions to 50. Eight layer-specific `Linear(d_model,1024)` reconstruction
heads produce `[B,8,50,1024]`.

The encoder and decoder do not use eight independent Conformer stacks. Temporal
backbones and cross-layer mixers are shared; symbol and reconstruction heads are
layer-specific.

## Balanced ragged CF-2 layout

CF-2 does not call `F.interpolate`. It retains 50 temporal positions and uses a
fixed `[B,8,50,5]` complex tensor plus a boolean `[50,5]` validity mask. Frames
`[2,7,12,17,22,27,32,37,42,47]` contain four valid symbols; all other frames
contain five. The fifth slot in each short frame is invalid and exactly zero.
This produces 240 valid symbols per layer and 1,920 total.

Only valid symbols participate in complex-power normalization and packed
channel-use counting. The decoder reconstructs the same fixed-width tensor
before its mask-aware zero-padded projection. The mask and exact count pattern
are derived from resolved configuration rather than learned or reliability
allocation.

## Waveform gradient path

`SpeechTokenizerWrapper.decode_representation()` computes the actual decoder
input as:

```text
representation.permute(0,1,3,2).sum(dim=1)  # [B,1024,50]
```

Its normal method calls `model.eval()`. cuDNN RNN backward rejects an eval-mode
RNN, so waveform-connected training uses
`decode_frozen_representation_with_gradient()`. That helper:

- does not detach the reconstructed representation;
- does not enter `torch.no_grad()` or `torch.inference_mode()`;
- sums the eight reconstructed layers exactly as the wrapper does;
- sets only frozen RNN decoder modules to training mode to permit cuDNN
  backward;
- leaves every codec parameter at `requires_grad=False`.

The existing B/C training targets were extracted under no-grad, as expected.
Their reconstructed branch remained differentiable. The main defect in those
experiments was not a blocked gradient: the full waveform loss was enabled from
step 1 and the negative SI-SDR term dominated before a useful latent
reconstruction existed.

## Detach, cache, and inference audit

- Target SpeechTokenizer encoding uses `torch.no_grad()`; this is intentional.
- Clean waveform references are detached before loss comparison; reconstructed
  waveform decoding is not detached.
- The new revalidation config rejects `data.latent_cache_dir` and loads real
  waveforms directly.
- Validation uses no-grad because it does not update the JSCC model.
- No OFDM, pilot, channel, jammer, CSI, equalizer, learned gate, allocation, or
  latent-refiner function is called by the dedicated training entry point.

## Parameters and channel uses

Executable construction gives:

| Configuration | Encoder | Decoder | Total | Symbol frames | Complex uses |
|---|---:|---:|---:|---:|---:|
| A / B / CF-1 | 7,234,688 | 9,045,504 | 16,280,192 | 30 | 1,920 |
| C (old) | 7,333,376 | 9,057,792 | 16,391,168 | 30 | 7,680 |
| CF-2 | 7,222,352 | 9,043,968 | 16,266,320 | 50 ragged | 1,920 valid |
| CF-3 | 7,234,688 | 9,045,504 | 16,280,192 | 50 | 3,200 |
| CF-4 | 22,829,440 | 25,542,656 | 48,372,096 | 50 | 3,200 |

SpeechTokenizer trainable parameter count is zero in A/B/C and the new path.

## Existing A/B/C evidence

All three prior experiments used 4,096 steps:

- A: 1,920 uses, latent-only; best latent step 4,096 and best waveform step
  3,900.
- B: 1,920 uses, waveform-connected from step 1; best latent step 2,000 and
  best waveform step 1,900.
- C: 7,680 uses, waveform-connected from step 1; best latent step 4,000 and
  best waveform step 3,200.

Because A used a different objective from B/C, those runs cannot isolate channel
budget. The new CF matrix keeps the curriculum, dataset, seed policy, optimizer,
and scheduled steps controlled.
