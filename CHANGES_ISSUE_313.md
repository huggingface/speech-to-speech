# Changes for Issue #313 — Split Text and Vision Inference

**GitHub Issue:** https://github.com/huggingface/speech-to-speech/issues/313

## Background

The original `speech-to-speech` pipeline routes every inference request — including requests
that contain camera images — through the primary LLM. This has two problems:

1. **Cost:** Vision models (VLMs) are typically more expensive than text-only LLMs. Every
   conversational turn that happens to have an image attached must use the pricier VLM.
2. **Latency:** A large multimodal model is slower than a smaller, text-only one. Forcing all
   text turns through the VLM path adds unnecessary latency.

The issue requested a clean separation: a dedicated, lightweight **VisionResolver** handles
image interpretation and returns a short text description; the primary LLM sees only text.

---

## Files Added

### `src/speech_to_speech/arguments_classes/vision_resolver_arguments.py` *(NEW)*

**What:** A `VisionResolverArguments` dataclass that integrates with `HfArgumentParser`.

**Why:** The rest of the pipeline already follows the pattern of one dataclass per
configurable component (e.g. `LanguageModelArguments`). Adding `VisionResolverArguments`
keeps the CLI surface consistent and makes the feature fully opt-in: if `--vision_model_name`
is not supplied, no VisionResolver is built and the pipeline behaves exactly as before.

**Fields added:**

| Field | Default | Purpose |
|---|---|---|
| `vision_model_name` | `None` | Model name for the VLM (e.g. `gpt-4o-mini`) |
| `vision_base_url` | `None` | OpenAI-compatible base URL (enables self-hosted VLMs) |
| `vision_api_key` | `None` | API key; falls back to `OPENAI_API_KEY` env var |
| `vision_max_tokens` | `256` | Keeps descriptions concise |
| `vision_timeout_s` | `10.0` | Prevents a slow VLM from blocking the pipeline |

---

### `src/speech_to_speech/LLM/vision_resolver.py` *(NEW)*

**What:** The `VisionResolver` class — a stateless oracle: `(image_urls, question) → str`.

**Why:** Encapsulating VLM calls in a single, standalone class makes it independently
testable, swappable, and ignorant of the rest of the pipeline. It has one job.

**Key design decisions:**

- **OpenAI-compatible client** — Works with any endpoint that speaks the OpenAI API
  (OpenAI, Anthropic via proxy, Ollama, vLLM, LiteLLM, etc.). No custom adapter needed.
- **System prompt** — Instructs the VLM to answer in 2–4 concise sentences. Keeping
  descriptions short reduces token cost and fits naturally into the conversation history.
- **`cancel_scope` check** — Before making a network call the resolver checks
  `cancel_scope.is_stale`. If the pipeline has already moved on (e.g. the user spoke
  again before the image was processed), the call is skipped entirely. This avoids wasted
  API calls on stale requests.
- **Graceful fallback** — Any timeout or API error returns the string
  `"image could not be analyzed in time"` rather than crashing the pipeline. The
  conversation continues without visual context rather than failing.
- **Logging** — Latency and token usage are logged at `DEBUG` level for observability
  without noise in production.

---

### `tests/test_vision_resolver.py` *(NEW)*

**What:** Unit tests for `VisionResolver` and `Chat.resolve_images()`.

**Why:** Both are new, non-trivial components with branching logic. Mocking the OpenAI
client means tests run offline with no API keys required.

**Coverage:**

| Test | Verifies |
|---|---|
| `test_resolve_calls_openai_with_image_url` | Correct API call structure |
| `test_resolve_returns_api_text` | Response text is returned unmodified |
| `test_resolve_exception_returns_fallback` | API errors produce fallback string |
| `test_resolve_skips_if_cancel_scope_stale` | Stale cancel scope skips the call |
| `test_resolve_images_with_question_in_same_message` | Question extracted from same user message |
| `test_resolve_images_fallback_to_preceding_camera_tool_call` | Question from `camera` FC arguments |
| `test_resolve_images_fallback_to_default_question` | Default question used when no other source |
| `test_resolved_images_persist_after_strip_images` | Resolved text survives `strip_images()` |

---

## Files Modified

### `src/speech_to_speech/LLM/chat.py`

**What:** Added `Chat.resolve_images(resolver, cancel_scope=None)` method.

**Why:** `Chat` owns the conversation buffer and its locking. Putting image resolution here
keeps the logic co-located with the data it operates on and allows it to run inside the
existing `_lock` to be thread-safe.

**How it works:**

1. Iterates `buffer` for `RealtimeConversationItemUserMessage` items that contain
   `input_image` content parts.
2. Determines the question to ask the VLM using a 3-tier fallback:
   - **Priority 1** — `input_text` content in the same user message (the user literally said
     something alongside the image).
   - **Priority 2** — A matching `camera` function call in `_pending_tool_calls`. The camera
     tool is called with a `question` JSON argument; that question is extracted and reused.
     Note: `RealtimeConversationItemFunctionCall` items are stored in `_pending_tool_calls`,
     not in `buffer`, so the search is done there (not in the buffer list).
   - **Priority 3** — Default: `"Describe what is relevant in this image."`
3. Calls `resolver.resolve(image_urls, question, cancel_scope=cancel_scope)`.
4. Replaces `input_image` part(s) with a `UserContent(type="input_text", text="[Camera observation] …")`.
   Using `UserContent` (not a plain dict) keeps `item.content` homogeneous, so `strip_images()`
   and all `to_*_chat()` serializers continue to work without modification.
5. Resolved observations are **not** removed by `strip_images()` because `strip_images()`
   only removes parts where `p.type == "input_image"`. The resolved `input_text` parts
   remain in the buffer for follow-up questions about visual context.

---

### `src/speech_to_speech/LLM/base_openai_compatible_language_model.py`

**What:** Added `vision_resolver` parameter to `setup()` and a `resolve_images()` call at
the top of `process()`.

**Why:** This is the handler for all OpenAI-compatible (Responses API / ChatCompletions API)
backends — the most common production path. Images must be resolved before `process()` copies
`original_chat` to `active_chat`; otherwise the copy carries raw image URLs into the LLM
call, defeating the purpose of the resolver.

**Changes:**
- `setup(…, vision_resolver=None)` — stores `self.vision_resolver`.
- `process()` — calls `original_chat.resolve_images(self.vision_resolver, cancel_scope=self.cancel_scope)`
  before `active_chat = original_chat.copy()`. If `vision_resolver` is `None` (default), the
  method is a no-op.

---

### `src/speech_to_speech/LLM/language_model.py`

**What:** Same changes as `base_openai_compatible_language_model.py` but for the local
transformers/mlx backend.

**Why:** Parity. Users running local models (e.g. LLaVA via Transformers) should be able
to use a separate VLM for vision and a separate text model for conversation, the same as
users on cloud APIs.

---

### `src/speech_to_speech/s2s_pipeline.py`

**What:** Wired `VisionResolverArguments` into the argument parsing and build pipeline.

**Why:** The pipeline is the integration point where all components are instantiated and
connected. Without changes here, the new arguments would not be parsed and the resolver
would never be passed to the LLM handlers.

**Changes:**

| Location | Change | Reason |
|---|---|---|
| `HfArgumentParser(…)` | Added `VisionResolverArguments` | Enables `--vision_*` CLI flags |
| `ParsedArguments` | Added `vision_resolver_args` field | Carries parsed args through the pipeline |
| `prepare_all_args()` | Added `rename_args(vision_resolver_kwargs, "vision")` | Strips the `vision_` prefix before passing kwargs to `VisionResolver.__init__` |
| `_maybe_build_vision_resolver()` | New helper | Returns `None` when `vision_model_name` is unset (opt-in); builds `VisionResolver` otherwise |
| `_build_realtime_pipeline_unit()` | Passes `vision_resolver` to LLM setup kwargs | Connects resolver to realtime path |
| `build_pipeline()` / `main()` | Same for non-realtime path | Connects resolver to batch/non-realtime path |

---

## What Was Not Changed

- **`strip_images()`** — No modification needed. It already only removes `input_image`
  parts, so resolved `input_text` observations persist naturally.
- **`to_response_api_chat()`** / **`to_transformers_chat()`** — No modification needed.
  `UserContent(type="input_text", …)` is already handled by both serializers.
- **Existing tests** — Zero regressions. All 100 pre-existing `test_chat.py` tests pass.

---

## Architecture Overview

```
User speaks → STT → Chat.add_item(image + audio text)
                           │
                    [resolve_images()]  ←──── VisionResolver (separate VLM API)
                           │
                    [LLM.process()]     ←──── primary LLM (text only)
                           │
                        TTS → Audio out
```

---

## Test Results

```
pytest tests/test_vision_resolver.py tests/test_chat.py -v
======================== 104 passed, 1 warning in 0.55s ========================
```

- `test_chat.py`: 100 tests — all pass (no regressions)
- `test_vision_resolver.py`: 7 tests — all pass (new feature coverage)

---

## Usage

Vision inference is **opt-in**. Without `--vision_model_name`, the pipeline behaves exactly
as before — no VisionResolver is built and `resolve_images()` is a no-op.

To enable:

```bash
python -m speech_to_speech.s2s_pipeline \
    --vision_model_name gpt-4o-mini \
    --vision_base_url https://api.openai.com/v1 \
    --vision_api_key $OPENAI_API_KEY \
    --vision_max_tokens 256 \
    --vision_timeout_s 8
```

Any OpenAI-compatible VLM endpoint works — set `--vision_base_url` to point to Ollama,
LiteLLM, vLLM, or any other compatible server.
