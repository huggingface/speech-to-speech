# STT Summary

This document summarizes the Speech-to-Text (STT) implementations in the `STT/` folder, including language support, language abbreviations, and usage in `s2s_pipeline.py`.

## Available STT Modes (`--stt`)

- `whisper` → `STT/whisper_stt_handler.py`
- `whisper-mlx` → `STT/lightning_whisper_mlx_handler.py`
- `mlx-audio-whisper` → `STT/mlx_audio_whisper_handler.py`
- `faster-whisper` → `STT/faster_whisper_handler.py`
- `parakeet-tdt` → `STT/parakeet_tdt_handler.py`
- `moonshine` → `STT/moonshine_handler.py`
- `paraformer` → `STT/paraformer_handler.py`

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

### 6) Moonshine (`--stt moonshine`)

- Handler: `MoonshineSTTHandler`
- No language CLI argument exposed
- Output language is hardcoded to `en`
- Practical support: English-focused in current integration

### 7) Paraformer (`--stt paraformer`)

- Handler: `ParaformerSTTHandler`
- Model flag: `--paraformer_stt_model_name`
- Default model: `paraformer-zh`
- No dedicated language flag in current args class
- Practical support:
  - Depends on selected FunASR model checkpoint
  - Default setup is Chinese-oriented (`zh`)

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
python s2s_pipeline.py --stt whisper --language en
python s2s_pipeline.py --stt whisper --language auto
```

### Whisper MLX (LightningWhisperMLX)

```bash
python s2s_pipeline.py --stt whisper-mlx --language auto --device mps
```

### MLX Audio Whisper

```bash
python s2s_pipeline.py --stt mlx-audio-whisper \
  --mlx_audio_whisper_model_name mlx-community/whisper-large-v3-turbo \
  --language auto
```

### Faster-Whisper

```bash
python s2s_pipeline.py --stt faster-whisper \
  --faster_whisper_stt_model_name large-v3 \
  --faster_whisper_stt_gen_language en
```

### Parakeet TDT

```bash
python s2s_pipeline.py --stt parakeet-tdt --parakeet_tdt_device auto
python s2s_pipeline.py --stt parakeet-tdt --parakeet_tdt_language de
```

With live transcription (MLX or CUDA/nano-parakeet backend):

```bash
python s2s_pipeline.py --stt parakeet-tdt \
  --enable_live_transcription \
  --live_transcription_update_interval 0.25
```

### Moonshine

```bash
python s2s_pipeline.py --stt moonshine
```

### Paraformer

```bash
python s2s_pipeline.py --stt paraformer --paraformer_stt_model_name paraformer-zh
```
