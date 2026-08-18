# R4 Jammer Estimator and MoE Refiner Trainer Design

## Goal

Train the existing receiver-only `JammerEstimator` and post-decoder `MoEAdaptiveLatentRefiner` on the fixed R4 physical path. The source medium JSCC checkpoint is loaded as the frozen baseline; no allocation or receiver-combining policy changes are permitted.

## Jammer taxonomy

The supervised vocabulary is exactly:

1. `no_jammer`
2. `broadband_awgn`
3. `subband`
4. `burst`
5. `tone`

`narrowband` is accepted only as a config input alias and is normalized to `subband` before any label is formed. Pilot-only jamming is excluded: broadband attacks the full grid including pilots, while tone may overlap pilot REs based on its deterministic location.

## Training boundary

The trainer creates an R4 forward engine with U0 and `no_refiner` physical allocation behavior. It runs the same forward once per sample, then applies estimator/refiner computation after the JSCC decoder output. The estimator sees received grid, pilots, pilot mask, LS channel estimate, and noise variance only. It never receives true jammer labels, true mask, or simulator jammer tensor. The true label/mask are used only by CE/BCE/Dice loss.

Codec stays frozen and in eval mode. JSCC encoder and decoder are frozen by default. A config-only `unfreeze_jscc_decoder` opt-in adds decoder parameters to the optimizer; the encoder remains frozen. Optimizer parameters otherwise contain only estimator and refiner parameters.

## Phases

### Phase 1: estimator_pretrain

R4 physical forward is evaluated under sampled jammer/SNR/JSR conditions. Optimize type CE + mask BCE + Dice. Refiner is not used for the training objective.

### Phase 2: oracle_mask_refiner

Estimator remains frozen. The refiner gets the physical mask and a one-hot class posterior only inside this named oracle upper-bound phase. It optimizes latent, optional STFT/SI-SDR, and no-jammer identity loss.

### Phase 3: learned_mask_moe

Estimator and refiner are optimized jointly. The refiner receives the estimator posterior and `mask_prob`; type/mask supervision and reconstruction losses are summed. No true jammer tensor/mask/type enters estimator forward or allocation.

## Loss

$$
L=\lambda_{type}L_{CE}+\lambda_{mask}L_{BCE}+\lambda_{dice}L_{Dice}
+\lambda_{latent}L_{latent}+\lambda_{stft}L_{STFT}
+\lambda_{si}L_{-SI\text{-}SDR}+\lambda_{identity}L_{identity}.
$$

The identity term is applied only to no-jammer examples. Mask BCE supports an optional positive-class weight; focal loss is not added in this first trainer to avoid silently changing the requested BCE baseline.

## Validation and checkpoints

Validation uses fixed seeds and the five jammer types. It reports estimator accuracy, mask BCE, mask IoU/F1, refined latent NMSE, SI-SDR, and no-jammer degradation. Checkpoints are isolated under the new output root and contain separate estimator/refiner states, optimizer/scheduler state, source checkpoint metadata, phase/global step, label vocabulary, config, and best metrics. `last.pt`, `best_validation_si_sdr.pt`, and `best_validation_latent.pt` are written without modifying existing SI-SDR fine-tuning artifacts.

## CLI and smoke

`train_r4_jammer_refiner.py` accepts config, source checkpoint, device, output directory, long-run guard, max-step override, and seed override. CPU smoke runs exactly two steps and verifies `last.pt`, reloadability, frozen codec/JSCC state, and finite loss. CUDA commands are supplied to the user but are not run by Codex.
