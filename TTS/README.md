# TTS Summary

## Available TTS modes (`--tts`)

Runtime-supported values in `s2s_pipeline.py`:

- `melo` → `melo_handler.py`
- `chatTTS` → `chatTTS_handler.py`
- `facebookMMS` → `facebookmms_handler.py`
- `pocket` → `pocket_tts_handler.py`
- `kokoro` → `kokoro_handler.py`
- `qwen3` → `qwen3_tts_handler.py`

## Usage

### 1) MeloTTS (`--tts melo`)

Primary args prefix: `--melo_*`

```bash
python s2s_pipeline.py \
  --tts melo \
  --melo_language en \
  --melo_device auto \
  --melo_speaker_to_id en
```

Language switching can occur automatically when STT emits `(text, language_code)` tuples.

Apple Silicon MPS note:
- If MeloTTS fails with `Output channels > 65536 not supported at the MPS device`, update macOS first.
- We reproduced this on an older macOS release and verified that the same MeloTTS code worked after updating to macOS `26.3.1`, without rebuilding the environment.

### 2) ChatTTS (`--tts chatTTS`)

Primary args prefix: `--chat_tts_*`

```bash
python s2s_pipeline.py \
  --tts chatTTS \
  --chat_tts_device cuda \
  --chat_tts_stream true \
  --chat_tts_chunk_size 512
```

### 3) Facebook MMS (`--tts facebookMMS`)

Primary args prefix: `--facebook_mms_*` plus `--tts_language`

```bash
python s2s_pipeline.py \
  --tts facebookMMS \
  --facebook_mms_device cuda \
  --tts_language en
```

This handler maps STT language codes (e.g. `en`, `fr`, `es`) to MMS model suffixes (e.g. `eng`, `fra`, `spa`) and reloads the model on language changes.

### 4) Pocket TTS (`--tts pocket`)

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

### 5) Kokoro (`--tts kokoro`)

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

### 6) Qwen3-TTS (`--tts qwen3`)

Primary args prefix: `--qwen3_tts_*`

```bash
python s2s_pipeline.py \
  --tts qwen3 \
  --qwen3_tts_model_name Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --qwen3_tts_device cuda \
  --qwen3_tts_ref_audio TTS/ref_audio.wav
```

## Setup

### Low-latency GPU setup

```bash
python s2s_pipeline.py \
  --stt whisper \
  --tts pocket
```

### Apple Silicon setup

```bash
python s2s_pipeline.py \
  --local_mac_optimal_settings \
  --tts melo
```

`--tts pocket` is also a valid option on macOS.
