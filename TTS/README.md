# TTS Summary

## Available TTS modes (`--tts`)

Runtime-supported values in `s2s_pipeline.py`:

- `parler` → `parler_handler.py`
- `melo` → `melo_handler.py`
- `chatTTS` → `chatTTS_handler.py`
- `facebookMMS` → `facebookmms_handler.py`
- `pocket` → `pocket_tts_handler.py`
- `kokoro` → `kokoro_handler.py`

## Usage

### 1) Parler-TTS (`--tts parler`)

Recommended usage with Cuda

Primary args prefix: `--tts_*`

```bash
python s2s_pipeline.py \
  --tts parler \
  --tts_model_name parler-tts/parler-mini-v1-jenny \
  --tts_device cuda \
  --tts_torch_dtype float16
```

Useful options:
- `--description`
- `--tts_compile_mode`
- `--tts_gen_min_new_tokens`
- `--tts_gen_max_new_tokens`
- `--play_steps_s`

### 2) MeloTTS (`--tts melo`)

Primary args prefix: `--melo_*`

```bash
python s2s_pipeline.py \
  --tts melo \
  --melo_language en \
  --melo_device auto \
  --melo_speaker_to_id en
```

Language switching can occur automatically when STT emits `(text, language_code)` tuples.

### 3) ChatTTS (`--tts chatTTS`)

Primary args prefix: `--chat_tts_*`

```bash
python s2s_pipeline.py \
  --tts chatTTS \
  --chat_tts_device cuda \
  --chat_tts_stream true \
  --chat_tts_chunk_size 512
```

### 4) Facebook MMS (`--tts facebookMMS`)

Primary args prefix: `--facebook_mms_*` plus `--tts_language`

```bash
python s2s_pipeline.py \
  --tts facebookMMS \
  --facebook_mms_device cuda \
  --tts_language en
```

This handler maps STT language codes (e.g. `en`, `fr`, `es`) to MMS model suffixes (e.g. `eng`, `fra`, `spa`) and reloads the model on language changes.

### 5) Pocket TTS (`--tts pocket`)

Primary args prefix: `--pocket_tts_*`

```bash
python s2s_pipeline.py \
  --tts pocket \
  --pocket_tts_voice jean \
  --pocket_tts_device cpu \
  --pocket_tts_sample_rate 16000
```

Available preset voices include:
`alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, `azelma`.

### 6) Kokoro (`--tts kokoro`)

Primary args prefix: `--kokoro_*`

```bash
python s2s_pipeline.py \
  --tts kokoro \
  --kokoro_device auto \
  --kokoro_voice bm_fable \
  --kokoro_lang_code b
```

Behavior:
- Uses MLX backend on Apple Silicon (`mlx-community/Kokoro-82M-bf16`)
- Uses native kokoro pipeline otherwise (`hexgrad/Kokoro-82M`)
- Can auto-switch voice/language based on STT language code mapping

## Setup

### Low-latency GPU setup

```bash
python s2s_pipeline.py \
  --stt whisper \
  --tts parler \
  --tts_compile_mode default
```

### Apple Silicon setup

```bash
python s2s_pipeline.py \
  --local_mac_optimal_settings \
  --tts kokoro
```
