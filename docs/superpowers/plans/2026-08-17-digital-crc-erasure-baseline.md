# Digital-CRC-erasure baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare the fixed R4 proposed JSCC path against a CRC-protected,
3GPP 38.212 LDPC/QPSK digital RVQ-index baseline using identical paired
waveform/channel/jammer conditions.

**Architecture:** The digital source creates independent CRC packets for every
RVQ layer × ten-frame block.  Each packet is LDPC encoded and rate matched;
all code bits are deterministically interleaved over the complete 5,760-RE R4
CSI-only allocation pool.  Thus layer/time source packets remain independent,
while physical placement uses the entire fixed R4 UEP resource budget instead
of impossible per-layer payload silos.  Failed CRC packets zero only their
assigned `[layer, frame]` RVQ embeddings before the existing continuous
SpeechTokenizer decoder is used.

**Tech Stack:** PyTorch, installed Sionna 2.0.1 3GPP 38.212 LDPC, existing
R4 OFDM/LS CSI/MRC physical path, SpeechTokenizer wrapper, pytest, YAML.

## Global Constraints

- Use `refiner_mode=no_refiner` for proposed and digital methods.
- Never modify R4 OFDM, MRC, allocator, `UEPProfile`, repetition, or power
  logic; load the fixed importance/repetition/power source artifact.
- Use only `allocate_r4_csi_only` for allocation; no jammer/risk/oracle input.
- No source-bit dropping, concealment, oracle replacement, or RE/energy increase.
- Use independent layer × ten-frame CRC packets but globally interleave their
  LDPC rate-matched code bits across the existing 5,760 data RE pool.
- CUDA commands are supplied to the user; implementation verification is CPU only.

---

### Task 1: RVQ-index wrapper and digital primitives

**Files:**
- Modify: `src/speech_jscc/codecs/wrappers.py`
- Create: `src/speech_jscc/evaluation/digital_crc_erasure.py`
- Test: `tests/test_digital_crc_erasure.py`

**Interfaces:**
- Produces `SpeechTokenizerWrapper.encode_rvq_indices(waveform) -> Tensor`
  shaped `[B,L,T]` and `lookup_rvq_indices(indices) -> Tensor` shaped
  `[B,L,T,D]`.
- Produces `CRCBlockLayout`, `append_crc16`, `check_crc16`,
  `qpsk_modulate`, and `qpsk_llr`.

- [ ] **Step 1: Write failing wrapper/CRC tests**

```python
def test_rvq_indices_lookup_has_continuous_representation_shape(codec, wave):
    indices = codec.encode_rvq_indices(wave)
    assert tuple(indices.shape[1:]) == (8, 50)
    assert tuple(codec.lookup_rvq_indices(indices).shape[1:]) == (8, 50, 1024)

def test_crc_failure_zeros_only_its_layer_time_block():
    embedding = torch.ones(1, 8, 50, 4)
    erased = erase_failed_crc_blocks(embedding, [(3, 10, 20)])
    assert torch.equal(erased[:, 3, 10:20], torch.zeros_like(erased[:, 3, 10:20]))
    assert erased[:, 2].eq(1).all()
```

- [ ] **Step 2: Run to verify RED**

Run: `pytest -q tests/test_digital_crc_erasure.py -k 'rvq or crc'`

- [ ] **Step 3: Add minimal wrapper and CRC/QPSK implementation**

Use `SpeechTokenizer.model.encode(..., n_q=n_q)`, normalize the official
`[L,B,T]` code layout to `[B,L,T]`, validate index ranges against
`get_codebook()`, and perform lookup from the shared codebook.  CRC uses
CRC-16-CCITT over each fixed ten-frame packet.

- [ ] **Step 4: Run to verify GREEN**

Run: `pytest -q tests/test_digital_crc_erasure.py -k 'rvq or crc'`

### Task 2: Sionna LDPC blocks and global R4 bit interleaver

**Files:**
- Modify: `src/speech_jscc/evaluation/digital_crc_erasure.py`
- Test: `tests/test_digital_crc_erasure.py`

**Interfaces:**
- Produces `SionnaLDPCCodec(k, n, num_iter)` and
  `DigitalPacketTransport.encode_packets(...)` /
  `decode_packets(...)`.
- Produces `GlobalBitInterleaver` that maps exactly 11,520 QPSK bit slots to
  packet rate-matched codeword bits, preserving all data RE and total energy.

- [ ] **Step 1: Write failing transmission tests**

```python
def test_noiseless_ldpc_qpsk_round_trip_is_bit_perfect():
    result = transport.round_trip(indices, noiseless=True)
    assert torch.equal(result.indices, indices)
    assert result.crc_failure_count == 0

def test_global_interleaver_consumes_all_r4_qpsk_bit_slots():
    layout = interleaver.layout(profile)
    assert layout.data_re_count == 5760
    assert layout.qpsk_bit_count == 11520
    assert layout.total_energy == pytest.approx(5760)
```

- [ ] **Step 2: Run to verify RED**

Run: `pytest -q tests/test_digital_crc_erasure.py -k 'ldpc or interleaver'`

- [ ] **Step 3: Implement Sionna 5G LDPC adapter**

Instantiate Sionna `LDPC5GEncoder/LDPC5GDecoder` with explicit `k`, `n`,
base graph, rate, decoder iteration count, and Sionna version metadata.  Use
rate-matched codeword lengths whose sum is exactly 11,520; map packet bits
globally with a deterministic permutation derived solely from paired seed and
CSI-only allocation ordering.  Decode soft LLRs and CRC-check every source
packet independently.

- [ ] **Step 4: Run to verify GREEN**

Run: `pytest -q tests/test_digital_crc_erasure.py -k 'ldpc or interleaver'`

### Task 3: Paired R4 evaluator and artifacts

**Files:**
- Create: `src/speech_jscc/evaluation/evaluate_digital_crc_erasure.py`
- Create: `evaluate_digital_crc_erasure.py`
- Create: `configs/eval_digital_crc_erasure.yaml`
- Test: `tests/test_evaluate_digital_crc_erasure.py`

**Interfaces:**
- Produces a CLI that reloads the fixed UEP artifact, validates checkpoint
  hashes, creates no-refiner proposed JSCC and digital rows on identical
  `R4ForwardCondition`s, and saves JSONL/CSV summaries in a fresh directory.

- [ ] **Step 1: Write failing paired-condition tests**

```python
def test_methods_share_condition_hash_and_csi_only_mapping(plan):
    rows = evaluate_one_condition(plan)
    assert rows['proposed_jscc']['condition_hash'] == rows['digital_crc_erasure']['condition_hash']
    assert rows['proposed_jscc']['allocation_mode'] == 'csi_only'
    assert rows['digital_crc_erasure']['refiner_mode'] == 'no_refiner'
```

- [ ] **Step 2: Run to verify RED**

Run: `pytest -q tests/test_evaluate_digital_crc_erasure.py`

- [ ] **Step 3: Implement evaluator using canonical R4 physical path**

Reuse the fixed condition plan, `R4WaveformForward`, waveform SI-SDR helper,
and physical R4 functions.  Proposed and digital rows must record RE/pilot
counts, layer energy, allocator artifact SHA, LDPC metadata, CRC erasures,
effective SINR, CSI NMSE, pilot EVM, and waveform SI-SDR.  Do not write into
existing run directories.

- [ ] **Step 4: Run CPU smoke and regression tests**

Run: `pytest -q tests/test_digital_crc_erasure.py tests/test_evaluate_digital_crc_erasure.py`

### Task 4: Documentation and user-run CUDA command

**Files:**
- Modify: `preflight_report.md`
- Create: `docs/digital_crc_erasure_baseline.md`
- Update: existing R4 Notion research page

- [ ] **Step 1: Record the approved global interleaving decision**

Document the prior per-layer capacity failure, the global 11,520-bit
rate-matching budget, exact fairness invariants, Sionna metadata, and the
fact that CRC validity remains per layer/time block.

- [ ] **Step 2: Provide CUDA command without executing it**

```bash
python evaluate_digital_crc_erasure.py \
  --config configs/eval_digital_crc_erasure.yaml \
  --device cuda --allow-long-run
```

- [ ] **Step 3: Verify documentation links and no-overwrite guard**

Run: `pytest -q tests/test_digital_crc_erasure.py tests/test_evaluate_digital_crc_erasure.py`
