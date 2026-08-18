# Digital-CRC-erasure baseline preflight

Date: 2026-08-17.  This report is a read-only inventory made before adding a
digital baseline.  No existing run directory, CSV, or checkpoint was changed.

## Repository and existing paths

* `git status --short` shows pre-existing tracked and untracked work, including
  the R4 jammer/refiner and allocator implementation.  This baseline must be
  added without modifying those outputs.
* `README.md` identifies the current system as **continuous/analog neural
  JSCC** and explicitly lists digital FEC/index transmission as not present.
* The requested `src/speech_jscc/evaluation/paired.py` path does not exist;
  the older generic helper is `src/evaluation/paired.py`.  The canonical R4
  waveform path is instead
  `src/speech_jscc/training/r4_waveform_finetune.py`:
  `R4WaveformForward` and `r4_physical_layer_forward` provide the existing
  OFDM, physical pilots, Rayleigh multipath, AWGN, LS CSI, equalization, and
  MRC path.  The fixed paired condition harness is
  `src/speech_jscc/evaluation/evaluate_r4_jammer_refiner_checkpoints.py`.
* Existing R4 result directories were inspected, including
  `runs/waveform_aware_wireless/r4_xbest_{existing,csi_only,interference}_speech_metrics/`.
  Their fixed plan has 16 utterances and paired seeds.  The three-way result
  joins have 864 rows each; those artifacts can supply the exact existing R4
  condition plan, but must not be overwritten.
* No local Notion export was found.  The prior Notion documentation refers to
  the same R4 artifacts; this report therefore treats the on-disk artifacts as
  the reproducible source of truth.

## Frozen codec and source representation

* Codec config: `configs/speechtokenizer_codec.yaml`.
* SpeechTokenizer config SHA256:
  `ea343ad69ca7e70c8febf8fc4cda683b1c4b1c36709e5e577936ffb05d62e6eb`.
* SpeechTokenizer checkpoint SHA256:
  `d04593b6c9a4b475f91ca481141a6ef5b23e6ac112f347dd2b2717f193c1c728`.
* A real CPU load verified 16 kHz, `n_q=8`, 16,000-sample crops,
  continuous representation `[B,8,50,1024]`, and RVQ codebooks
  `[8,1024,1024]`.  The official encoder returns integer codes shaped
  `[8,B,50]`; therefore the observed bit width is 10, but the future digital
  implementation must derive it from each loaded codebook size rather than
  hard-code it.
* `SpeechTokenizerWrapper` currently exposes only continuous embeddings and
  codebooks.  It needs a backward-compatible official-index API plus a
  codebook-lookup-to-`decode_representation()` API.  The existing continuous
  JSCC interface must remain unchanged.

## Fixed R4 comparison inputs

* Proposed JSCC checkpoint:
  `runs/waveform_aware_wireless/r4_si_sdr_finetune/si_sdr_medium/local_step_003000.pt`
  SHA256 `40ca7bd2cb8f774a71e8650b71838b89118d36bcf01b338882ac1ebe6c5e70fb`.
  Metadata reports `local_step=3000`, `source_global_step=5750`; its state
  format contains `config`, `model`, `optimizer`, `diagnostic_type`,
  `local_step`, and `source_global_step`.
* Selected UEP artifact:
  `runs/waveform_aware_wireless/r4_broadband_uep_optimization/stage1_screen/selected_profiles.json`
  SHA256 `9b1c51b8b81f9510829506aaadc73819c2805c2bbde459d4f42d2af451a41653`.
  `x_best` is the required input—not a new optimization—with repetition
  `[3,4,3,1,5,1,4,3]` (sum 24) and power shares
  `[0.1376379188,0.2077466505,0.2155657833,0.1023641692,0.1039495679,0.1115058737,0.0659890437,0.0552409929]`.
  This preserves 5,760 data RE and unit mean transmit power.
* The canonical fixed importance order in the active R4 code/config is
  `[1,0,2,5,3,4,6,7]` (`r4_waveform_finetune.LAYER_IMPORTANCE` and
  `configs/ofdm_nr_like_r4_repetition3_mrc.yaml`).  Its config provenance is
  marked provisional; the digital evaluator must record both that config
  source and SHA, rather than claim a newly derived importance order.
* The shared allocator must be
  `allocate_r4_csi_only`: it uses only `GlobalTripletCSIReport.reliability`.
  It must not receive interference reports, jammer masks/types/JSR, true
  channels, or RX residual risk.  `R4WaveformForward` already supports this
  as jammer-aware allocation mode `csi_only`.
* Existing physical profile is `configs/ofdm_nr_like_r4.yaml` (512 FFT,
  360 active subcarriers, 30 kHz SCS, 28 OFDM symbols, pilots at 3 and 17).
  `r4_uep_allocator.py` validates eight layers × 240 source symbols and
  exactly 5,760 data RE.

## Checkpoint inventory

All `runs/**/*.pt` file entries were enumerated and metadata-loading was
attempted before implementation.  The tree contains many historical aliases
and directories whose names end in `.pt`; directory entries are not treated as
corrupt checkpoints.  The relevant source, medium, UEP, and jammer-refiner
checkpoints above are present.  A new evaluator must validate each selected
checkpoint SHA and metadata before using it; it must never overwrite a
checkpoint or resume an existing result directory.

## LDPC backend and digital bit-budget gate

The environment was initially checked for `sionna`, `pyldpc`, `commpy`,
`fec`, and `torchldpc`: all were absent (`scipy` is present).  With explicit
approval, **Sionna 2.0.1** was installed and its 3GPP 38.212
`LDPC5GEncoder`/`LDPC5GDecoder` API was loaded successfully (BG2 selected by
the tested short-block configuration, with rate matching and iterative soft
decoding).

However, the fixed R4 source/RE contract makes the requested digital transport
infeasible before an LDPC rate can be chosen:

* Each RVQ layer carries 50 indices × `ceil(log2(1024)) = 500` source bits.
* `crc_block_frames=10` creates five independent CRC blocks per layer.  A
  16-bit CRC therefore raises this to **580 information bits per layer**.
* The fixed R4 contract has 240 distinct source positions per layer.  Under
  the explicit requirement that repetition copies are copies of the *same*
  QPSK coded bits, there are only 240 distinct QPSK symbols per layer, i.e.
  **480 uncoded bits per layer**.  Repetition `r_i` improves reliability but
  cannot increase information capacity.
* Thus 580 > 480 for every layer (and even 500 > 480 before CRC); any LDPC
  rate would only increase the required number of transmitted bits.  Across
  the crop, RVQ requires 4,000 index bits before CRC, while the 1,920 distinct
  QPSK positions carry only 3,840 bits before channel coding.

Using all 5,760 RE as fresh coded bits would violate the requirement that
repeated QPSK copies carry the same coded bit; increasing modulation order,
changing the R4 resource budget, dropping source bits, or sharing a block
between layers would each change an explicitly fixed comparison rule.

**Resolved specification decision (2026-08-17):** CRC packets remain
layer/time independent, but their LDPC rate-matched code bits are globally
interleaved over the unchanged 5,760 R4 data RE.  Repetition-budget RE are
therefore used for systematic and parity bits rather than identical QPSK bit
copies.  The total 11,520 QPSK-bit budget supports 40 packets × 116 bits at a
rate-matched length of 288 bits/packet (effective rate 0.4028).  The digital
implementation must continue to reject any alternate mode that silently drops
bits or changes RE/energy budgets.
