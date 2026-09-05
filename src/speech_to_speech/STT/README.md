# STT Summary

This document summarizes the Speech-to-Text (STT) implementations in the `STT/` folder, including language support, language abbreviations, and usage in `s2s_pipeline.py`.

## Available STT Modes (`--stt`)

- `whisper` → `STT/whisper_stt_handler.py`
- `whisper-mlx` → `STT/lightning_whisper_mlx_handler.py`
- `mlx-audio-whisper` → `STT/mlx_audio_whisper_handler.py`
- `faster-whisper` → `STT/faster_whisper_handler.py`
- `parakeet-tdt` → `STT/parakeet_tdt_handler.py`
- `parakeet-unified` → `STT/nemo_asr_handler.py`
- `nemotron-streaming` → `STT/nemo_asr_handler.py`
- `paraformer` → `STT/paraformer_handler.py`
- `qwen3-asr` → `STT/qwen3_asr_handler.py`
- `openai` → `STT/openai_compatible_handler.py`

## Language Support by Handler

### 1) Whisper (`--stt whisper`)

- Handler: `WhisperSTTHandler`
- Language input flag: `--language` (from shared Whisper args)
- Supports fixed language (e.g. `en`) or `auto`
- Internal supported language list:
  - `en`, `fr`, `es`, `zh`, `ja`, `ko`, `hi`, `de`, `pt`, `pl`, `it`, `nl`
- Behavior:
  - Detects language from token output
  - If detected language is outside the supported list, it falls back to the previous language

### 2) Lightning Whisper MLX (`--stt whisper-mlx`)

- Handler: `LightningWhisperSTTHandler`
- Uses same shared `--language` argument as Whisper
- Internal supported language list:
  - `en`, `fr`, `es`, `zh`, `ja`, `ko`, `hi`, `de`, `pt`, `pl`, `it`, `nl`
- Behavior:
  - If `--language auto`, model auto-detects each utterance
  - If detected language is unsupported, falls back to last supported language

### 3) MLX Audio Whisper (`--stt mlx-audio-whisper`)

- Handler: `MLXAudioWhisperSTTHandler`
- Model flag: `--mlx_audio_whisper_model_name`
- Language still comes from shared `--language` flag (wired in pipeline)
- Internal supported language list:
  - `en`, `fr`, `es`, `zh`, `ja`, `ko`, `hi`, `de`, `pt`, `pl`, `it`, `nl`
- Behavior:
  - Uses fixed language unless `--language auto`
  - Falls back to last known supported language when needed

### 4) Faster-Whisper (`--stt faster-whisper`)

- Handler: `FasterWhisperSTTHandler`
- Language flag: `--faster_whisper_stt_gen_language`
- Default language: `en`
- Note:
  - This handler passes generation kwargs directly to `faster_whisper.WhisperModel.transcribe(...)`
  - Effective language coverage depends on selected Faster-Whisper/OpenAI Whisper model

### 5) Parakeet TDT (`--stt parakeet-tdt`)

- Handler: `ParakeetTDTSTTHandler`
- Language flag: `--parakeet_tdt_language` (optional)
- Supports auto language detection when language not specified
- Declared supported language list (25 European languages):
  - `en`, `de`, `fr`, `es`, `it`, `pt`, `nl`, `pl`, `ru`, `uk`, `cs`, `sk`, `hu`, `ro`, `bg`, `hr`, `sl`, `sr`, `da`, `no`, `sv`, `fi`, `et`, `lv`, `lt`
- Backend behavior:
  - On macOS/MPS: MLX (`mlx-community/parakeet-tdt-0.6b-v3`)
  - On CUDA/CPU: nano-parakeet (`nvidia/parakeet-tdt-0.6b-v3`)

### 6) Paraformer (`--stt paraformer`)

- Handler: `ParaformerSTTHandler`
- Model flag: `--paraformer_stt_model_name`
- Default model: `paraformer-zh`
- No dedicated language flag in current args class
- Practical support:
  - Depends on selected FunASR model checkpoint
  - Default setup is Chinese-oriented (`zh`)

### 7) Qwen3-ASR (`--stt qwen3-asr`)

- Handler: `Qwen3ASRSTTHandler`
- Model flag: `--qwen3_asr_model_name` (default `Qwen/Qwen3-ASR-0.6B-hf`; `Qwen/Qwen3-ASR-1.7B-hf` is more accurate)
- Language flag: `--qwen3_asr_language` (ISO code or name, default `auto`)
- Context flag: `--qwen3_asr_prompt` (optional hotwords or domain context)
- Supported languages: the 30 in the checkpoint's language table
  - `ar`, `yue`, `zh`, `cs`, `da`, `nl`, `en`, `fil`, `fi`, `fr`, `de`, `el`, `hi`, `hu`, `id`, `it`, `ja`, `ko`, `mk`, `ms`, `fa`, `pl`, `pt`, `ro`, `ru`, `es`, `sv`, `th`, `tr`, `vi`
- Behavior:
  - With `auto`, the model identifies the language of each final turn and reports it with the `-auto` suffix
  - Progressive (partial) windows are short and fool the language ID, so they reuse the language of the last final turn
  - A forced language is passed on every request and reported as is
- Transformers versions: `pyproject.toml` already requires a version that knows `qwen3_asr`. The prompt and true language forcing need `transformers>=5.15.1` (the Linux pin); with 5.14.1 (the macOS pin) the language is a hint and the prompt is ignored with a warning

### 8) OpenAI-compatible endpoint (`--stt openai`)

- Handler: `OpenAICompatibleSTTHandler`
- Endpoint: `POST /v1/audio/transcriptions`
- Upload: mono PCM16 WAV at 16 kHz
- Supports JSON (`{"text": "..."}`) and plain-text responses
- Keeps at most one best-effort progressive request in flight per pipeline while
  final requests are submitted independently; stale-turn filtering still applies
- See [`docs/openai-compatible-stt.md`](../../../docs/openai-compatible-stt.md)

### 9) Parakeet Unified (`--stt parakeet-unified`)

- Handler: `NemoASRSTTHandler`
- Install: `pip install "speech-to-speech[nemo]"`
- Model flag: `--parakeet_unified_model_name`
- Default model: `nvidia/parakeet-unified-en-0.6b`
- Language flag: `--parakeet_unified_language` (default `en`)
- Device flag: `--parakeet_unified_device` (default `auto`)
- The pipeline transcribes each VAD utterance with NeMo `ASRModel.transcribe` (offline API)

### 10) Nemotron Streaming (`--stt nemotron-streaming`)

- Handler: `NemoASRSTTHandler`
- Install: `pip install "speech-to-speech[nemo]"`
- Model flag: `--nemotron_streaming_model_name`
- Default model: `nvidia/nemotron-speech-streaming-en-0.6b`
- Override: `nvidia/nemotron-3.5-asr-streaming-0.6b` for multilingual
- Language flag: `--nemotron_streaming_language` (default `en`)
- Device flag: `--nemotron_streaming_device` (default `auto`)
- The pipeline transcribes each VAD utterance with NeMo `ASRModel.transcribe` (offline API)

## Language Abbreviations (ISO-style codes seen in STT handlers)

| Code | Language |
|---|---|
| `en` | English |
| `fr` | French |
| `es` | Spanish |
| `zh` | Chinese |
| `ja` | Japanese |
| `ko` | Korean |
| `hi` | Hindi |
| `de` | German |
| `pt` | Portuguese |
| `pl` | Polish |
| `it` | Italian |
| `nl` | Dutch |
| `ru` | Russian |
| `uk` | Ukrainian |
| `cs` | Czech |
| `sk` | Slovak |
| `hu` | Hungarian |
| `ro` | Romanian |
| `bg` | Bulgarian |
| `hr` | Croatian |
| `sl` | Slovenian |
| `sr` | Serbian |
| `da` | Danish |
| `no` | Norwegian |
| `sv` | Swedish |
| `fi` | Finnish |
| `et` | Estonian |
| `lv` | Latvian |
| `lt` | Lithuanian |
| `auto` | Per-utterance automatic language detection |

## Usage Examples

### Whisper (Transformers)

```bash
speech-to-speech serve --stt whisper --language en
speech-to-speech serve --stt whisper --language auto
```

### Whisper MLX (LightningWhisperMLX)

```bash
speech-to-speech serve --stt whisper-mlx --language auto --device mps
```

### MLX Audio Whisper

```bash
speech-to-speech serve --stt mlx-audio-whisper \
  --mlx_audio_whisper_model_name mlx-community/whisper-large-v3-turbo \
  --language auto
```

### Faster-Whisper

```bash
speech-to-speech serve --stt faster-whisper \
  --faster_whisper_stt_model_name large-v3 \
  --faster_whisper_stt_gen_language en
```

### Parakeet TDT

```bash
speech-to-speech serve --stt parakeet-tdt --parakeet_tdt_device auto
speech-to-speech serve --stt parakeet-tdt --parakeet_tdt_language de
```

With live transcription (MLX or CUDA/nano-parakeet backend):

```bash
speech-to-speech serve --stt parakeet-tdt \
  --enable_live_transcription \
  --live_transcription_update_interval 0.25
```

### Paraformer

```bash
speech-to-speech serve --stt paraformer --paraformer_stt_model_name paraformer-zh
```

### Qwen3-ASR

```bash
speech-to-speech serve --stt qwen3-asr
speech-to-speech serve --stt qwen3-asr --qwen3_asr_language fr
speech-to-speech serve --stt qwen3-asr \
  --qwen3_asr_model_name Qwen/Qwen3-ASR-1.7B-hf \
  --qwen3_asr_prompt "Vocabulary: Quilter, apostle."
```

### Parakeet Unified

```bash
pip install "speech-to-speech[nemo]"
speech-to-speech serve --stt parakeet-unified
```

The pipeline transcribes VAD utterances with NeMo `ASRModel.transcribe` (offline API).

### Nemotron Streaming

```bash
pip install "speech-to-speech[nemo]"
speech-to-speech serve --stt nemotron-streaming
speech-to-speech serve --stt nemotron-streaming \
  --nemotron_streaming_model_name nvidia/nemotron-3.5-asr-streaming-0.6b
```

The pipeline transcribes VAD utterances with NeMo `ASRModel.transcribe` (offline API).
