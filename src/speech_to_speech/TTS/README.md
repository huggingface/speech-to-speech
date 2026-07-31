# TTS Summary

## Available TTS modes (`--tts`)

Runtime-supported values in `s2s_pipeline.py`:

- `chatTTS` → `chatTTS_handler.py`
- `facebookMMS` → `facebookmms_handler.py`
- `pocket` → `pocket_tts_handler.py`
- `kokoro` → `kokoro_handler.py`
- `qwen3` → `qwen3_tts_handler.py`

Deprecated TTS implementations, including MeloTTS, live in [`../../../archive/TTS`](../../../archive/TTS) and are no longer wired into `s2s_pipeline.py`.

## Usage

### 1) ChatTTS (`--tts chatTTS`)

Primary args prefix: `--chat_tts_*`

```bash
python s2s_pipeline.py \
  --tts chatTTS \
  --chat_tts_device cuda \
  --chat_tts_stream true \
  --chat_tts_chunk_size 512
```

### 2) Facebook MMS (`--tts facebookMMS`)

Primary args prefix: `--facebook_mms_*` plus `--tts_language`

```bash
python s2s_pipeline.py \
  --tts facebookMMS \
  --facebook_mms_device cuda \
  --tts_language en
```

This handler maps STT language codes (e.g. `en`, `fr`, `es`) to MMS model suffixes (e.g. `eng`, `fra`, `spa`) and reloads the model on language changes.

### 3) Pocket TTS (`--tts pocket`)

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

### 4) Kokoro (`--tts kokoro`)

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

### 5) Qwen3-TTS (`--tts qwen3`)

Primary args prefix: `--qwen3_tts_*`

```bash
python s2s_pipeline.py \
  --tts qwen3 \
  --qwen3_tts_model_name Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --qwen3_tts_device cuda \
  --qwen3_tts_backend ggml \
  --qwen3_tts_speaker Aiden \
  --qwen3_tts_language auto \
  --qwen3_tts_non_streaming_mode True
```

Behavior:
- Uses `faster-qwen3-tts` on non-macOS platforms, defaulting to the GGML backend. Pass `--qwen3_tts_backend torch` to use the CUDA-graphs backend instead.
- Supports GGML quantization selection via `--qwen3_tts_ggml_quantization BF16|Q8_0|Q4_K_M|F32`.
- Accepts a local talker/codec GGUF pair via `--qwen3_tts_gguf_talker_path` and `--qwen3_tts_gguf_codec_path`.
- Automatically caches `.spk` and `.rvq` voice references when GGML voice cloning uses raw reference audio. Set `--qwen3_tts_ref_cache_dir` to override the default `~/.cache/faster-qwen3-tts/qwentts_refs` location.
- Reuses precomputed GGML references via `--qwen3_tts_ref_spk` and the optional `--qwen3_tts_ref_rvq`.
- Uses `mlx-audio` on Apple Silicon and auto-maps `Qwen/...` model IDs to `mlx-community/...`, defaulting to the `6bit` MLX variant unless the model name already pins a suffix.
- Supports MLX quantization overrides on Apple Silicon via `--qwen3_tts_mlx_quantization bf16|4bit|6bit|8bit`.
- Keeps the existing voice-clone/custom-voice/voice-design handler flow intact.
- Defaults to the CustomVoice model with speaker `Aiden`, so no reference audio is required. Voice-clone/base models can still use `--qwen3_tts_ref_audio`.

Install notes for Linux GGML:
- The default PyPI `qwentts-cpp-python` wheel targets CUDA 12.8.
- If that wheel does not match your CUDA runtime, install one of the Hugging Face wheelhouse builds before installing `speech-to-speech`.

```bash
pip install "qwentts-cpp-python==0.3.1+cu130" \
  -f https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/tree/main/whl/cu130
pip install speech-to-speech
```

Available wheelhouse directories include `cu124`, `cu128`, `cu130`, and `cpu`.

Select a quantized GGUF from the public model resolver:

```bash
python s2s_pipeline.py \
  --tts qwen3 \
  --qwen3_tts_backend ggml \
  --qwen3_tts_model_name Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --qwen3_tts_ggml_quantization Q4_K_M \
  --qwen3_tts_speaker Aiden
```

Load local GGUF files by providing both the talker and codec paths. Keep the model name aligned with the local talker's model type so the handler selects the correct generation mode:

```bash
python s2s_pipeline.py \
  --tts qwen3 \
  --qwen3_tts_backend ggml \
  --qwen3_tts_model_name Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --qwen3_tts_gguf_talker_path /models/qwen-talker-1.7b-voicedesign-Q4_K_M.gguf \
  --qwen3_tts_gguf_codec_path /models/qwen-tokenizer-12hz-BF16.gguf \
  --qwen3_tts_instruct "Warm, confident narrator"
```

For GGML voice cloning, a raw reference is cached on its first use:

```bash
python s2s_pipeline.py \
  --tts qwen3 \
  --qwen3_tts_backend ggml \
  --qwen3_tts_model_name Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --qwen3_tts_ref_audio /voices/freeman.wav \
  --qwen3_tts_ref_text "The transcript for the reference audio." \
  --qwen3_tts_ref_cache_dir /voices/cache
```

Automatic cache entries use the same SHA-256 cache key for their `.spk`, `.rvq`, and `.json` filenames. Inspect the cache directory for the matching key and replace `CACHE_KEY` below with that filename stem. You can then pass its `.spk` by itself for speaker-only conditioning, or pair it with the corresponding `.rvq` codes and transcript for ICL conditioning:

```bash
python s2s_pipeline.py \
  --tts qwen3 \
  --qwen3_tts_backend ggml \
  --qwen3_tts_model_name Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --qwen3_tts_ref_spk /voices/cache/CACHE_KEY.spk \
  --qwen3_tts_ref_rvq /voices/cache/CACHE_KEY.rvq \
  --qwen3_tts_ref_text "The transcript for the reference audio."
```

Raw `--qwen3_tts_ref_audio` and cached `--qwen3_tts_ref_spk`/`--qwen3_tts_ref_rvq` inputs are mutually exclusive. `.rvq` input requires both `.spk` and reference text.

Example for Apple Silicon using the default 6-bit MLX variant:

```bash
python s2s_pipeline.py \
  --tts qwen3 \
  --qwen3_tts_model_name Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --qwen3_tts_speaker Aiden
```

You can override the default and select `bf16`, `4bit`, or `8bit` explicitly:

```bash
python s2s_pipeline.py \
  --tts qwen3 \
  --qwen3_tts_model_name Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --qwen3_tts_mlx_quantization 4bit \
  --qwen3_tts_speaker Aiden
```

To benchmark the Apple Silicon MLX variants side by side:

```bash
.venv/bin/python benchmark_tts.py \
  --handlers qwen3 \
  --iterations 3 \
  --qwen3_mlx_quantizations bf16 4bit 6bit 8bit
```

This will run separate benchmark entries for `qwen3[bf16]`, `qwen3[4bit]`, `qwen3[6bit]`, and `qwen3[8bit]`.

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
  --local_mac_optimal_settings
```

`--tts pocket` and `--tts kokoro` are also valid options on macOS.
