# STT Summary

This document summarizes the Speech-to-Text (STT) implementations in the `STT/` folder, including language support, language abbreviations, and usage in `s2s_pipeline.py`.

## Available STT Modes (`--stt`)

- `whisper` → `STT/whisper_stt_handler.py`
- `whisper-mlx` → `STT/lightning_whisper_mlx_handler.py`
- `mlx-audio-whisper` → `STT/mlx_audio_whisper_handler.py`
- `faster-whisper` → `STT/faster_whisper_handler.py`
- `parakeet-tdt` → `STT/parakeet_tdt_handler.py`
- `paraformer` → `STT/paraformer_handler.py`
- `qwen3-asr` → `STT/qwen3_asr_handler.py`
- `qwen3-asr-http` → `STT/qwen3_asr_http_handler.py`

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

### 7) Qwen3-ASR (`--stt qwen3-asr`) — in-process, macOS: currently blocked

- Handler: `Qwen3ASRSTTHandler`
- Requires the optional `qwen3-asr` extra: `pip install speech-to-speech[qwen3-asr]`
- Model flag: `--qwen3_asr_model_name`, default `Qwen/Qwen3-ASR-1.7B` (`Qwen/Qwen3-ASR-0.6B` also available for lower latency)
- Language flag: `--qwen3_asr_language` (ISO 639-1 code, e.g. `en`, `vi`, `zh`; omit or set to `auto` for automatic language detection)
- Supported language list (30 languages, `yue` = Cantonese): `zh`, `en`, `yue`, `ar`, `de`, `fr`, `es`, `pt`, `id`, `it`, `ko`, `ru`, `th`, `vi`, `ja`, `tr`, `hi`, `ms`, `nl`, `sv`, `da`, `fi`, `pl`, `cs`, `fil`, `fa`, `el`, `hu`, `mk`, `ro`
- Backend: `qwen-asr` package (transformers backend), one VAD-segmented utterance per call
- **`transformers` version conflict, confirmed unresolved on macOS:** `qwen-asr==0.0.6` hard-pins
  `transformers==4.57.6`, which conflicts with this project's `transformers==5.6.2` pin on macOS (needed by
  Qwen3-TTS/mlx-audio) — installing the extra requires `pip install --no-deps qwen-asr==0.0.6` (plus its
  non-`transformers` deps: `nagisa`, `librosa`, `soundfile`, `accelerate`) on top of the normal macOS install.
  On top of `transformers==5.6.2`, `qwen_asr`'s vendored modeling/config code
  (`qwen_asr/core/transformers_backend/`) hits **at least four** separate transformers 4→5 breaking changes
  it hasn't adapted to upstream: a deprecated `check_model_inputs()` call signature, `Qwen3ASRConfig.__init__`
  reading `self.thinker_config` before it's assigned (config validation now runs eagerly inside
  `super().__init__()`), the removed `"default"` key in `ROPE_INIT_FUNCTIONS` (replaced by a
  `rope_parameters`-based API — the replacement also drops `mrope_interleaved`/`mrope_section`, i.e. even a
  correct-looking patch here risks silently wrong positional encoding rather than a crash), and
  `Qwen3ASRThinkerConfig` similarly missing `pad_token_id` at model-build time. `qwen3_asr_handler.py` applies
  runtime shims for the first three (see `_apply_transformers_5x_compat` in that file; each checks current
  state first, so they no-op once `qwen-asr` fixes things upstream) but **model construction still fails** on
  the fourth, confirmed live against a real macOS install (Python 3.12, Apple Silicon). Do not rely on this
  handler on macOS until `qwen-asr` supports transformers 5.x upstream — use `--stt qwen3-asr-http` (below)
  instead, which sidesteps the whole conflict.
- On Linux/CUDA, the base `transformers>=4.57.0` requirement is a *range*, so `transformers==4.57.6` may
  resolve and run cleanly there without any of the above — untested, would appreciate confirmation.

### 8) Qwen3-ASR over HTTP (`--stt qwen3-asr-http`) — the actually-working path on macOS

- Handler: `Qwen3ASRHTTPSTTHandler`
- No extra needed in the main install — this handler only makes plain HTTP calls (via `httpx`, already a
  base dependency) and never imports `qwen_asr` or cares which `transformers` version this project has.
- Talks to a separately-running server, started in its **own** virtualenv pinned to
  `transformers==4.57.6` (the version `qwen-asr` actually needs). That server is
  [`scripts/qwen3_asr_server.py`](../../../scripts/qwen3_asr_server.py) — a small Flask wrapper around
  `Qwen3ASRModel.from_pretrained(...).transcribe(...)`, **not** the `qwen-asr` package's own
  `qwen-asr-serve` command, which wraps `vllm serve` and hard-requires the `vllm` package. `vllm` has no
  wheels for macOS/Apple Silicon at all, so `qwen-asr-serve` cannot run there regardless of backend flag —
  confirmed live (`ModuleNotFoundError: No module named 'vllm'` even with `--backend transformers`
  requested). `scripts/qwen3_asr_server.py` calls the exact same in-process transformers-backend API
  confirmed working end-to-end on Apple Silicon MPS, just wrapped in an HTTP endpoint instead of imported
  in-process:

  ```bash
  # Terminal 1 — separate venv, only for Qwen3-ASR
  uv venv --python 3.12 .venv-qwen3-asr
  source .venv-qwen3-asr/bin/activate
  uv pip install qwen-asr==0.0.6
  python scripts/qwen3_asr_server.py --device mps --host 127.0.0.1 --port 8000
  ```

  ```bash
  # Terminal 2 — the normal speech-to-speech venv (transformers==5.6.2 on macOS)
  speech-to-speech --stt qwen3-asr-http --qwen3_asr_http_base_url http://127.0.0.1:8000
  ```

- Flags: `--qwen3_asr_http_base_url` (default `http://127.0.0.1:8000`), `--qwen3_asr_http_timeout_s`
  (default `30.0`).
- Both processes run on `localhost`, so the added latency is a local HTTP round-trip (single-digit
  milliseconds), negligible next to model inference time.
- Known gap: per-request language forcing (`--qwen3_asr_language`-equivalent) isn't wired up for this path —
  `scripts/qwen3_asr_server.py` always calls `.transcribe(language=None)`, relying on Qwen3-ASR's own
  language auto-detection tag in the response. Easy to add if needed (a `language` field on the request
  JSON, threaded through to `.transcribe(language=...)`); not done here for lack of a concrete need yet.
- On a CUDA host with `vllm` installed, `qwen-asr-serve` (the real one) would likely outperform this via
  vLLM's batching — this handler's HTTP contract only assumes an OpenAI-chat-completions-shaped response, so
  swapping the server side later doesn't require touching `qwen3_asr_http_handler.py`.
- Prefer Docker over a native venv for the server side? See "CPU-only / Qwen3-ASR Docker setup" in
  the top-level README's [Docker section](../../../README.md#docker) for a `docker compose` alternative that
  runs both this and the pipeline as containers — CPU-only, no NVIDIA Container Toolkit required.

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

### Paraformer

```bash
python s2s_pipeline.py --stt paraformer --paraformer_stt_model_name paraformer-zh
```

### Qwen3-ASR

```bash
python s2s_pipeline.py --stt qwen3-asr --qwen3_asr_language en
python s2s_pipeline.py --stt qwen3-asr --qwen3_asr_language auto
python s2s_pipeline.py --stt qwen3-asr --qwen3_asr_model_name Qwen/Qwen3-ASR-0.6B
```

### Qwen3-ASR over HTTP (macOS-friendly)

```bash
# with scripts/qwen3_asr_server.py already running in its own venv, per section 8 above
python s2s_pipeline.py --stt qwen3-asr-http --qwen3_asr_http_base_url http://127.0.0.1:8000
```
