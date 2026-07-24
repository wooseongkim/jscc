# G0 Architecture Screening Report

## Executive conclusion

The G0 direct-bypass failure was primarily caused by the flatten-MLP architecture, not by an intrinsically unusable 240-complex-symbol per-layer bottleneck.

The full-data `conv_conformer_v1` experiment passes every revised G0 gate. On unseen-speaker utterances it reduces Stage-1 normalized loss from the zero baseline of 1.0 to **0.50848**, reaches correlation **0.69377** and power ratio **0.50517**, and reconstructs enhancement layers 1–7 with mean loss **0.56199** and correlation **0.65994**. The completed PCA reference also reconstructs layers 1–7 meaningfully, establishing that the fixed 480-real-dimensional per-layer bottleneck retains useful linear information. The raw and normalized flat MLPs fail to exploit that information.

This is evidence for the following interpretation:

- the 1920-complex-symbol total budget is not a structural impossibility at G0;
- temporal/shared-layer modeling is materially better than flattening each latent layer;
- corpus normalization alone does not cure enhancement-layer collapse in the tested normalized-flat subset-16 experiment;
- `conv_conformer_v1` is the accepted architecture for the next G1 diagnostic;
- these results do not yet establish robustness to mapping, pilots, channels, AWGN, CSI error, or jammers.

## Protocol integrity

All full-subset comparisons use:

- representation `[B,8,50,1024]`;
- frozen SpeechTokenizer;
- 240 complex symbols per codec layer, 1920 total;
- direct encoder-to-decoder symbol bypass;
- no mapper, pilot, channel, noise, CSI estimator, equalizer, or jammer;
- uniform Stage-1 per-layer-power-normalized MSE;
- identical 64-presentation exposure for trainable models;
- identical validation suite, manifest, and latent cache.

Matching hashes for raw flat MLP, PCA full, and Conv-Conformer full:

| Provenance item | Hash |
|---|---|
| Validation suite | `d84a97c5908c8e878fcfbe5f9cd6f45397018a8d57f9274120e6967e970bdc71` |
| Train manifest | `de5c7c1057cd3d0c962f06953d13b0745010743ffe5aa4415b6f17b607b2ed40` |
| Latent cache | `de8278a597a31742853cfbd882c062aeb40b474f3a90e3b2845441e5aef0b834` |

PCA is an offline fitted reference and does not use optimizer epochs. The normalized-flat result currently available is subset 16 only, so it must not be treated as a full-corpus normalization comparison.

## Primary results

### Unseen-speaker unseen-utterance validation

| Architecture | Train subset | Train loss | Validation loss | Improvement over zero | Power ratio | Correlation | Screening status |
|---|---:|---:|---:|---:|---:|---:|---|
| Raw flat MLP | 16 | 0.98773 | 1.01187 | -1.19% | 0.01415 | 0.02325 | Fail |
| Raw flat MLP | 64 | 0.97779 | 0.99821 | 0.18% | 0.00941 | 0.06078 | Fail |
| Raw flat MLP | 256 | 0.97515 | 0.98060 | 1.94% | 0.01282 | 0.07689 | Fail |
| Raw flat MLP | full (1491) | 0.96812 | 0.96974 | 3.03% | 0.03087 | 0.08703 | Fail |
| Normalized flat MLP | 16 | 0.95703 | 0.98357 | 1.64% | 0.04809 | 0.07111 | Fail |
| Per-layer PCA-480 | full | n/a | 0.78955 | 21.04% | 0.21175 | 0.43396 | Pass, offline reference |
| Conv-Conformer | 16 | 0.83197 | 0.92787 | 7.21% | 0.15008 | 0.26180 | Enhancement gate fail |
| Conv-Conformer | 64 | 0.75149 | 0.78186 | 21.81% | 0.25413 | 0.45439 | Metrics pass; subset not eligible as final gate |
| Conv-Conformer | 256 | 0.67664 | 0.71584 | 28.42% | 0.31889 | 0.51442 | Pass |
| Conv-Conformer | full (1491) | 0.48541 | **0.50848** | **49.15%** | **0.50517** | **0.69377** | **Pass** |

The Conv-Conformer improvement grows consistently with train-set size. Full-data train and unseen losses remain close (`0.48541` versus `0.50848`), so the result is not explained by small-subset memorization.

### Same-speaker unseen-utterance validation

| Architecture | Subset | Loss | Improvement over zero | Power ratio | Correlation |
|---|---:|---:|---:|---:|---:|
| Normalized flat MLP | 16 | 0.98576 | 1.42% | 0.04969 | 0.07244 |
| PCA-480 | full | 0.76448 | 23.55% | 0.23671 | 0.45675 |
| Conv-Conformer | 16 | 0.91635 | 8.36% | 0.16663 | 0.27997 |
| Conv-Conformer | 64 | 0.75712 | 24.29% | 0.27308 | 0.47608 |
| Conv-Conformer | 256 | 0.68421 | 31.58% | 0.34311 | 0.54155 |
| Conv-Conformer | full | **0.47280** | **52.72%** | **0.53845** | **0.71868** |

Both same-speaker and unseen-speaker groups pass for Conv-Conformer 256 and full. The unseen-speaker penalty remains measurable at full (`0.50848` versus `0.47280`) but is not a gate failure.

## Layer-specific analysis

### Unseen-speaker group

| Architecture | Subset | Layer 0 loss | Layer 0 corr. | Layers 1–7 loss | Layers 1–7 corr. | Layers 1–7 power ratio |
|---|---:|---:|---:|---:|---:|---:|
| Raw flat MLP | full | 0.76158 | 0.48815 | 0.99948 | 0.02973 | approximately collapse |
| Normalized flat MLP | 16 | 0.76720 | 0.48297 | 1.01448 | 0.01227 | 0.01780 |
| PCA-480 | full | 0.37212 | 0.79223 | 0.84919 | 0.38278 | 0.15216 |
| Conv-Conformer | 16 | 0.65771 | 0.58550 | 0.96646 | 0.21556 | 0.12119 |
| Conv-Conformer | 64 | 0.50079 | 0.70724 | 0.82201 | 0.41827 | 0.21274 |
| Conv-Conformer | 256 | 0.29989 | 0.83699 | 0.77526 | 0.46834 | 0.25873 |
| Conv-Conformer | full | **0.13388** | **0.93062** | **0.56199** | **0.65994** | **0.45263** |

The decisive result is the enhancement-layer behavior. The raw flat MLP remains at the zero-predictor solution on layers 1–7. Corpus normalization on subset 16 improves Layer 0 but leaves layers 1–7 worse than zero. PCA proves that enhancement-layer information survives the bottleneck, while Conv-Conformer learns substantially stronger nonlinear temporal reconstruction.

The full Conv-Conformer no longer exhibits the prior Layer-0-only failure. Layers 1–7 improve by approximately 43.8% relative to their zero baseline and have meaningful positive correlation and reconstruction power.

## Exposure behavior

Conv-Conformer unseen-speaker validation loss by epoch:

| Epoch | Subset 16 | Subset 64 | Subset 256 | Full |
|---:|---:|---:|---:|---:|
| 1 | 1.0339 | 1.0003 | 0.9744 | 0.8796 |
| 2 | 1.0082 | 0.9898 | 0.9687 | 0.7844 |
| 4 | 1.0004 | 0.9748 | 0.9281 | 0.7552 |
| 8 | 0.9898 | 0.9682 | 0.8127 | 0.7294 |
| 16 | 0.9741 | 0.9286 | 0.7664 | 0.6816 |
| 32 | 0.9665 | 0.8202 | 0.7466 | 0.5896 |
| 64 | 0.9279 | 0.7819 | 0.7158 | **0.5085** |

The full run continues improving through epoch 64. This supports both increased data diversity and adequate exposure as important factors. It does not invalidate the architectural conclusion: under the same full-data exposure, the raw flat MLP stops near `0.9697`, while Conv-Conformer reaches `0.5085` and reconstructs layers 1–7.

## Constant-baseline comparison

For the full unseen-speaker suite:

- zero predictor: `1.00000`;
- global train-mean predictor: `0.96951`;
- raw flat MLP: `0.96974`, slightly worse than global mean;
- PCA-480: `0.78955`, clearly better than global mean;
- Conv-Conformer: `0.50848`, approximately 47.55% better than global mean.

For normalized flat MLP subset 16, the model beats its subset-specific global mean on unseen validation, but only because that small-subset global mean has loss `1.02384`. It still improves only 1.64% over zero and fails the enhancement and content gates.

## Revised gate outcome

| Architecture / subset | Layer 0 | Layers 1–7 | Same speaker | Unseen speaker | Beats global mean | Final screening |
|---|---|---|---|---|---|---|
| Normalized flat / 16 | Pass | Fail | Fail | Fail | Pass | Fail |
| Conv-Conformer / 16 | Pass | Fail | Pass | Pass | Pass | Fail |
| Conv-Conformer / 64 | Pass | Pass | Pass | Pass | Pass | Diagnostic metrics pass; final subset eligibility false |
| Conv-Conformer / 256 | Pass | Pass | Pass | Pass | Pass | Pass |
| Conv-Conformer / full | Pass | Pass | Pass | Pass | Pass | Pass |
| PCA-480 / full | Pass | Pass | Pass | Pass | Pass | Pass, offline reference only |

Subset 64 has `architecture_screening_pass=true` in the detailed metric gate but top-level `passed=false` because the final screening policy admits only sufficiently exposed subset 256 or full results. This is a policy distinction, not a metric contradiction.

## Scientific interpretation

The evidence supports two specified cases:

1. **Case B:** PCA meaningfully reconstructs enhancement layers while the raw flatten MLP fails. The bottleneck retains sufficient linear information, and flatten-MLP optimization/structure is a main limitation.
2. **Case F:** Conv-Conformer passes layers 1–7 and both unseen-content gates. The flat MLP was the primary G0 limitation, and G1 may be resumed with Conv-Conformer.

The evidence does not support Case D: PCA and Conv-Conformer do not both fail. It also does not support Case E: the full Conv-Conformer does not remain Layer-0-only.

Corpus normalization may still have value, but the current normalized-flat experiment covers only 16 utterances. It is sufficient to show that normalization alone does not solve the small-subset enhancement collapse; it is not sufficient to measure its full-corpus ceiling.

## Recommendation

Accept `conv_conformer_v1` as the G0 architecture and preserve the completed full checkpoint as diagnostic evidence. The next task should rerun **G1 pilot-reserved identity mapping** using Conv-Conformer, fresh initialization, the same exposure-normalized subsets and revised gates. Progress to G2 only after G1 passes on subset 256 or full.

Do not resume O6 or jammer curriculum directly from this result. G0 proves content reconstruction through a direct symbol bypass only; allocation/pilot mapping and channel integration must be revalidated in order.

No change to the 1920-symbol budget, uniform loss, SpeechTokenizer, channel model, jammer model, or production power semantics is justified by this experiment.

## Artifact locations

- Raw flat baseline: `runs/stage1_content_generalization/g0_exposure_normalized_v1/`
- PCA: `per_layer_pca_480/subset_full/`
- Normalized flat MLP: `normalized_flat_mlp/subset_16/`
- Conv-Conformer: `conv_conformer_v1/subset_{16,64,256,full}/`
- Aggregate rows: `aggregate_results.json`, `aggregate_by_architecture.csv`, `per_layer_results.csv`

