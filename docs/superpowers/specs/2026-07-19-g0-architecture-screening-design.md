# G0 Architecture Screening Design

The approved screening compares the unchanged `flat_mlp`, a train-only corpus-normalized wrapper around that model, an offline per-layer 480-component PCA reference, and a new `conv_conformer_v1` model. All trainable paths retain the frozen SpeechTokenizer `[B,8,50,1024]` representation, uniform layer-normalized MSE, and 1,920 complex-symbol direct bypass split as 240 symbols per layer.

`conv_conformer_v1` uses shared temporal encoder and decoder Conformer stacks, learned codec-layer embeddings, optional attention mixing across the eight layers, observable-state FiLM conditioning, deterministic 50→30 and 30→50 interpolation/convolution resampling, and layer-specific symbol/reconstruction heads. It preserves the existing encoder and decoder call signatures and existing complex-power semantics. Historical configs continue to construct the unchanged flat MLP unless `model.architecture` is explicit.

Normalization statistics are fit only from training-cache tensors and carry manifest/cache hashes. PCA is an offline reference fitted independently per layer with randomized low-rank SVD and never enters production transmission. Architecture-aware checkpoint metadata is strict and forbids partial or cross-architecture loading.

The screening CLI reuses deterministic content subsets, validation groups, epoch-aware exposure scheduling, baselines, and Stage-1 metrics. Revised gates independently require layer 0, layers 1–7, same-speaker unseen, and unseen-speaker unseen performance, plus beating the global mean. Long work is emitted as safe external commands; Codex runs only tests, dry runs, model construction, and a bounded smoke step.

