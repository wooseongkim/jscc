# R4 expanded validation

This evaluator corrects the checkpoint-selection bias in the completed R4
waveform fine-tuning run. It performs evaluation only; it does not train or
modify the JSCC, SpeechTokenizer, allocation, OFDM, CSI, or MRC implementations.

## Historical protocol audit

The completed trainer used six `dev-clean-2` utterances and one channel seed
every 250 steps for light validation. It used 16 utterances and three seeds
every 1,000 steps for full validation. Both light and full results could update
all named checkpoint selectors. The later 64-utterance final evaluation used
the same ordered `dev-clean-2` pool, so the 16 full-validation utterances were
also present in the final suite.

Only four weight artifacts remain:

- `best_5db_si_sdr.pt` at step 5,750
- `best_clean_gate.pt` at step 5,750
- `best_validation_average.pt` at step 5,750
- `last.pt` at step 20,000

Historical validation JSON exists every 250 steps, but it contains metrics
rather than model weights. Missing intermediate checkpoints cannot be
reconstructed.

The historical `best_clean_gate.pt` name refers to the pure-neural/noiseless-R4
regression constraints. It did not require the 5 dB wireless waveform gate.

## Dataset roles

- Existing training subset: `train-clean-5`, role `train`.
- Expanded selection: deterministic speaker-balanced 48 utterances from the
  unmodified `manifests/mini_librispeech/test.jsonl`, source split `test-clean`,
  assigned role `selection_validation`.
- Frozen legacy final: the exact existing ordered 64 `dev-clean-2` utterances,
  role `legacy_final_test`.

The evaluator rejects any pairwise speaker or utterance overlap. Selection
results from `test-clean` must be called selection-validation results, not test
results.

## Pairing

For each SNR and realization, channel taps and AWGN seeds are generated once.
Noise variance is fixed by the nominal unit-symbol SNR so the initial and
candidate checkpoints receive the same AWGN tensor. The initial checkpoint
generates the receiver pilot/LS-CSI report trajectory. Every candidate receives
that exact delayed report trajectory. Candidate-generated reports are compared
against it within the configured tolerance but never fed into later
allocation.

The existing `R4WaveformForward` remains the sole implementation of allocation,
pilot insertion, time-domain OFDM, multipath, AWGN, LS estimation, coherent
MRC, JSCC decoding, and SpeechTokenizer waveform decoding.

## Commands

Dry run:

```bash
python evaluate_r4_expanded_validation.py \
  --config configs/eval_r4_expanded_validation.yaml \
  --dry-run
```

Bounded smoke:

```bash
python evaluate_r4_expanded_validation.py \
  --config configs/eval_r4_expanded_validation.yaml \
  --device cuda \
  --max-candidates 2 \
  --max-utterances 2 \
  --max-realizations 1 \
  --output-dir runs/waveform_aware_wireless/r4_expanded_validation_smoke \
  --overwrite
```

Full expanded selection validation:

```bash
bash scripts/run_r4_expanded_validation_external.sh --device cuda
```

The output directory is protected by default. To intentionally replace an
incomplete expanded-validation directory, add `--overwrite`.

Run the frozen final suite only after expanded selection:

```bash
python evaluate_r4_expanded_validation.py \
  --config configs/eval_r4_expanded_validation.yaml \
  --device cuda \
  --run-final-test \
  --allow-long-run
```

This command requires the completed selection artifacts and evaluates only
`best_expanded_validation.pt`. It does not rerun or delete selection. Add
`--overwrite` only when intentionally replacing an existing `final_test/`
subdirectory. The final result is written separately and never changes
checkpoint ranking.
