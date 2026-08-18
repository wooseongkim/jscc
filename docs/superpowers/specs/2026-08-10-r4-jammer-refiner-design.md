# R4 Learned Jammer Estimator and Adaptive Latent Refiner Design

## Goal

Add a receiver-only jammer estimator and posterior-gated mixture-of-experts latent refiner to the R4 waveform path without changing allocation, UEP, repetition mapping, or MRC.

## Scope and invariants

- The posterior classes are `no_jammer`, `broadband_awgn`, `subband`, `burst`, `block`, and `tone`.
- The estimator consumes only receiver-observable tensors: received grid, pilot grid, pilot mask, LS channel estimate, and noise variance.
- True jammer type and true jammer mask are labels for training losses only. They are never `JammerEstimator.forward` inputs.
- The refiner is post-decoder latent denoising only. It cannot affect transmitter allocation, source symbol mapping, UEP, power allocation, or MRC.
- `no_refiner` remains the default and returns the pre-existing raw decoder reconstruction exactly.

## Components

### JammerEstimator

`models.jammer_estimator.JammerEstimator` projects real/imaginary received-grid and pilot-residual features to a shared 2-D receiver feature map. It emits:

- `JammerEstimate.posterior`: `[B, 6]` class probability vector;
- `mask_logits`, `mask_prob`: `[B, K, N]` soft jammer mask in the physical active-grid shape;
- `mask_ratio`: `[B]`, the mask probability average.

`jammer_estimation_loss` combines posterior cross-entropy, mask BCE-with-logits, and Dice loss. Labels are passed only to that loss.

### MoEAdaptiveLatentRefiner

`models.adaptive_latent_refiner.MoEAdaptiveLatentRefiner` receives raw decoder latent `[B,L,T,D]`, decoder state, mask probability, and posterior. Each class expert predicts a residual latent delta; posterior weights form their mixture. The output is `raw_latent + delta`. Each residual head is zero-initialized, so the initial output is identity even for non-uniform posterior.

### R4 integration

`R4WaveformForward` optionally owns an estimator and refiner and supports:

- `no_refiner`: existing raw decoder reconstruction;
- `oracle_mask_refiner`: refiner receives `physical.jammer_mask` but posterior remains a receiver-side estimate (or a fixed no-jammer posterior only when no estimator is configured);
- `learned_mask_refiner`: learned estimator mask and posterior condition the refiner;
- `learned_posterior_moe_refiner`: same learned estimate, explicitly using the MoE routing path.

`R4WaveformOutput` retains `reconstruction` as the selected output and adds `raw_reconstruction`, `jammer_posterior`, `jammer_mask_prob`, and `refiner_mode`.

## Checkpoint and evaluation configuration

Optional configuration/CLI entries load separately serialized estimator and refiner states. Missing checkpoints are an error for learned modes; `no_refiner` needs neither. Existing evaluators remain comparable because their default remains `no_refiner`.

## Tests

1. Estimator shape and probability invariants.
2. Public estimator forward signature rejects true labels, proving label-leakage prevention.
3. Estimator loss uses labels independently of forward inputs.
4. MoE output shape and zero-initialized identity behavior.
5. R4 `no_refiner` reconstruction regression.
6. Oracle-mask and learned-mask forward smoke with finite tensors and unchanged physical allocation outputs.
