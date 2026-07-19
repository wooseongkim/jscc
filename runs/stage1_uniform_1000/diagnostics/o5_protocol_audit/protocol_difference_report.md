# O5 Protocol Difference Audit

**Classification:** different realization, not directly comparable

The historical O5 used batch seed 23003; C1 used seed 23023. Their 500-step losses are not a direct performance comparison.

| Field | Original O5 | New C1 | Classification |
|---|---|---|---|
| source_entry_point | `overfit_stage1_path.py` | `diagnose_o5_root_cause.py` | different |
| model_factory | `SpeechJSCC direct` | `build_components` | different |
| model_initialization_seed | `23` | `23` | same |
| latent_source | `RepresentationSource(train).next_batch(1)` | `RepresentationSource(train).next_batch(1)` | same |
| latent_hash | `None` | `None` | unknown |
| representation_shape | `[8, 50, 1024]` | `[8, 50, 1024]` | same |
| batch_size | `1` | `1` | same |
| optimizer | `Adam` | `Adam` | same |
| learning_rate | `0.0001` | `0.0001` | same |
| optimizer_reset_behavior | `fresh per ladder stage` | `fresh or exact resume` | different |
| optimization_steps | `500` | `requested CLI budget` | different |
| evaluation_interval | `final only` | `step 0/log interval/final` | different |
| loss_equation | `uniform per-layer power normalized MSE` | `uniform per-layer power normalized MSE` | same |
| layer_weights | `[1, 1, 1, 1, 1, 1, 1, 1]` | `[1, 1, 1, 1, 1, 1, 1, 1]` | same |
| normalization_epsilon | `1e-06` | `1e-06` | same |
| grid_shape | `[64, 32]` | `[64, 32]` | same |
| pilot_mask | `configured comb mask` | `configured comb mask` | same |
| resource_mapping | `pilot_reserved_v1` | `pilot_reserved_v1` | same |
| per_layer_channel_uses | `1920` | `1920` | same |
| snr_db | `10.0` | `10.0` | same |
| requested_jsr_db | `0.0` | `0.0` | same |
| jammer_type | `barrage` | `full_barrage_estimated_csi` | different |
| jammer_normalization | `_make_batch convention` | `total-grid normalized condition_batch` | different |
| fixed_batch_seed | `23003` | `23023` | different |
| fixed_realization_policy | `fixed within O5` | `fixed with hash assertion` | different |
| csi_estimator | `dft_tap_ls` | `dft_tap_ls` | same |
| estimator_num_taps | `6` | `6` | same |
| equalizer | `estimated legitimate CSI` | `estimated legitimate CSI` | same |
| receiver_state | `observable_v1` | `observable_v1` | same |
| transmitter_state | `zeros` | `zeros` | same |
| gates | `all ones` | `all ones` | same |
| allocation | `uniform equal layer power` | `uniform equal layer power` | same |
| decoder_input_ordering | `pilot-reserved deallocation` | `pilot-reserved deallocation` | same |
| model_mode | `default train mode` | `default train mode` | same |
| gradient_clipping | `none` | `none` | same |
| checkpoint_resume | `none` | `exact diagnostic resume supported` | different |
| metric_aggregation | `post-forward loss; final reconstruction before last update` | `logged pre-update states including final` | different |
| final_metric_step_semantics | `500th forward then optimizer update` | `step 500 forward; update occurs when extending` | different |
| step0_hash.latent_target | `87c6ce8a131b6b07a305f0868efb3dd13e6c13a9c17c2bda510f853503210b5a` | `87c6ce8a131b6b07a305f0868efb3dd13e6c13a9c17c2bda510f853503210b5a` | same |
| step0_hash.initial_model_parameters | `5d82e9ecc77dfed3d17ef4b880fa5119e3fef43258d217bc8b341f83a91fb116` | `c71e2b7e4403a5f16597c3dde1d8d3c65156fd788b17f990ea2bd3aedcdd81c1` | different |
| step0_hash.legitimate_channel | `b6f683a7a9056cea724cc37eddb6dd957d0b6e40fda05f37b195b0a4d9801428` | `f7332165f9b66b5d63a6fed487902aa3ccbaba2c5d4834b5ff69cce9bf65d0cf` | different |
| step0_hash.jammer_channel | `266c444ea8b4a1b669744b0b7db13ba8ccb7755325e91d6624d32c7fe7d9850c` | `aca6a9bef811c452a7c2aa911fe0bf24a04442c431ed5104eebebec10db323be` | different |
| step0_hash.awgn | `2ce6a7dfbb777b1336ea124814d754659f7ffe934ab2e68e92825931c31225b7` | `52d46ff81c30af13e91d7a35e9263f82525b7e24a7f5ae6f858c5514f963169b` | different |
| step0_hash.jammer_waveform | `a16a730937be103b865f5059e062433a721152ec14ed92e5ad893e9cfd642ac2` | `5218d44b4d93a64f8094fce0daf693f3859866f9a2c57e2879ccaafca26d884a` | different |
| step0_hash.jammer_mask | `85074aa95911014688e2f7d40b65bb6188303b65c0285aef22678b0e75a461fb` | `85074aa95911014688e2f7d40b65bb6188303b65c0285aef22678b0e75a461fb` | same |
| step0_hash.pilot_mask | `c0f2ae3ed9dee87238b6f3473667d9afc4409daf923c396aa5bbd67d323acc24` | `c0f2ae3ed9dee87238b6f3473667d9afc4409daf923c396aa5bbd67d323acc24` | same |
| step0_hash.transmitted_initial_data_symbols | `dafb06051b1c916179c69ed2bc4a200b21634a659ced165a920b22b2914b44c3` | `2f4ce9133e8c0624ba6a8b5ef49ec6bf1cf51d956337c353567e0788cefb6b7b` | different |
| step0_hash.receiver_state | `1a1435a93a4d5d5e05d9de7415a103c22365f06e6e78d7b7ec4d8a1abb2399ef` | `ea8118a4752666a2966be3d3f8c137c729624f9f61c5efc92f9dca8a197df385` | different |
| step0_hash.decoder_input | `1b69fe5863873827f950fd9f2e11d3c986c4b0497643eafeeb97d8ff69b12870` | `a569e7426da4628b6272d0085730b8cd9659b8eaeabfac028d5d2b47833da55f` | different |
