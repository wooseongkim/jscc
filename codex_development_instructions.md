# Codex 개발 계획 및 입력 프롬프트

## 핵심 방향

이 프로젝트는 디지털 FEC 기반 RVQ index 전송이 아니라, neural speech codec representation을 JSCC encoder가 complex channel symbol로 직접 매핑하는 speech-to-speech JSCC 연구이다. ASR/TTS/KG는 1차 구현의 본 경로에 넣지 않는다.

## 단계별 개발 계획

### Phase 0. 저장소 초기화
- Python/PyTorch 프로젝트 생성
- config 기반 실행 구조 구성
- pytest 기반 shape/power/channel test 작성

### Phase 1. Synthetic latent JSCC
- speech codec 없이 `[B, L, T, D]` 형태의 synthetic latent tensor 사용
- JSCC encoder/decoder 구현
- Rayleigh + AWGN + jamming channel 통과
- latent MSE와 power constraint 검증

### Phase 2. Channel/Jammer simulator
- flat Rayleigh channel
- optional OFDM grid channel
- barrage/narrowband/burst/pilot jammer
- effective SINR, JSR, jammer mask ratio, CSI NMSE 기록

### Phase 3. Codec wrapper
- `BaseCodec` interface 정의
- SpeechTokenizer 또는 EnCodec wrapper 연결
- pretrained model이 없을 때 mock codec으로 전체 pipeline 테스트 가능하게 작성

### Phase 4. Channel-adaptive layer gating
- RVQ layer별 gate `alpha_l(c)` 구현
- rule-based threshold policy 먼저 구현
- 이후 MLP 기반 learned gate 추가

### Phase 5. Soft codebook / latent refinement
- JSCC decoder output을 continuous embedding 또는 token logits로 복원
- soft codebook projection 구현
- optional latent refiner 추가

### Phase 6. Evaluation
- SNR/JSR/jammer type sweep
- uniform JSCC vs rule-based adaptive JSCC vs learned adaptive JSCC 비교
- layer-wise MSE, STOI/PESQ placeholder, speaker similarity placeholder, keyword accuracy optional 평가

## Codex CLI 입력 프롬프트

### Prompt 1
```text
Create a PyTorch research project for speech-to-speech JSCC under Rayleigh fading and jamming. Do not implement digital FEC or bit-level RVQ index transmission. The project must use continuous or soft codec representations as the JSCC source. Create the directory structure, config files, and minimal runnable training/evaluation scripts. Add pytest tests for tensor shapes and power normalization.
```

### Prompt 2
```text
Implement src/channels/rayleigh.py and src/channels/jammer.py. Use complex PyTorch tensors. The Rayleigh channel should support flat fading and optional OFDM resource grids. The jammer module should support barrage, narrowband, burst, and pilot jamming masks. Include functions to compute effective SINR, JSR, and jammer mask statistics. Add unit tests that verify signal power, jammer power, and output tensor shapes.
```

### Prompt 3
```text
Implement src/models/jscc_encoder.py and jscc_decoder.py. The encoder maps codec latent tensors of shape [B, L, T, D] plus channel-state vector c to complex channel symbols [B, M] or [B, K, N]. Enforce average power normalization. The decoder maps received complex symbols plus channel-state vector to reconstructed latent tensors [B, L, T, D]. Include layer gating and optional layer-wise power allocation, but keep the first version simple and deterministic.
```

### Prompt 4
```text
Implement src/models/soft_codebook.py. It should support two modes: (1) continuous latent reconstruction loss, and (2) soft codebook projection from token logits to codebook embeddings using temperature-scaled softmax. Include top-k token accuracy computation for analysis only. Do not convert the communication system into a digital index transmission pipeline.
```

### Prompt 5
```text
Create src/codecs/base_codec.py with an abstract interface: encode_waveform, decode_representation, get_codebook, representation_shape. Then create a placeholder SpeechTokenizerWrapper and EnCodecWrapper. If external pretrained models are unavailable, implement a mock codec that produces deterministic random latent tensors and reconstructs a dummy waveform so the JSCC pipeline can be tested end-to-end.
```

### Prompt 6
```text
Implement train_latent_jscc.py. It should load codec representations, sample random SNR/JSR/jammer types per batch, pass through JSCC encoder, Rayleigh+jamming channel, JSCC decoder, and compute layer-weighted latent MSE plus power penalty. Log loss, effective SINR, JSR, layer-wise MSE, and reconstruction examples. Add config options for layer weights, SNR range, JSR range, and jammer probabilities.
```

### Prompt 7
```text
Implement eval_jamming.py to sweep SNR, JSR, jammer type, and layer adaptation modes. Compare: uniform JSCC, rule-based channel-adaptive layer gating, and learned gating if available. Save CSV results and plots for layer-wise MSE, waveform metrics placeholders, and effective SINR. Keep ASR/WER only as optional evaluation, not part of the communication pipeline.
```

### Prompt 8
```text
Write README.md explaining that this project studies speech-to-speech JSCC using neural speech codec representations under Rayleigh fading and jamming. Clearly state that the system does not use ASR/TTS as the main communication path, does not transmit model weights, and does not implement digital FEC-based RVQ index transmission. Include quickstart commands, config descriptions, and experiment commands.
```
