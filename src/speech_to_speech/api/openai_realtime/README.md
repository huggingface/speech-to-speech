## Realtime Engine -- High-Level Architecture

The server runs as a handler thread inside the existing pipeline. A FastAPI/uvicorn instance serves the WebSocket endpoint at `/v1/realtime`, while the same queue-driven handler chain (VAD, STT, LLM, TTS) processes audio in parallel threads.

```mermaid
flowchart LR
    subgraph client [Client]
        WS["WebSocket /v1/realtime"]
    end

    subgraph server [Realtime Server]
        Router["WebSocket Router"]
        Service["RealtimeService"]
        Config["RuntimeConfig"]
    end

    subgraph pipeline [Pipeline Threads]
        VAD["VAD"]
        STT["STT"]
        TN["TranscriptionNotifier"]
        LLM["LLM"]
        Proc["LMOutputProcessor"]
        TTS["TTS"]
    end

    WS -- "client events (JSON)" --> Router
    Router -- "parse / dispatch" --> Service
    Service -- "PCM chunks" --> VAD
    Service -- "session config" --> Config
    Config -. "read by" .-> VAD & LLM & TTS
    VAD -- "speech segments" --> STT
    STT -- "transcript" --> TN
    TN -- "transcript + notify" --> LLM
    LLM -- "text + tools" --> Proc
    Proc -- "clean text" --> TTS
    TTS -- "PCM audio" --> Router
    Proc -- "assistant_text / tools" --> Router
    VAD -- "speech_started/stopped" --> Router
    TN -- "transcription events" --> Router
    Router -- "server events (JSON)" --> WS
```

**Key flow:**

1. **Inbound audio**: Client sends `input_audio_buffer.append` with base64 PCM. `RealtimeService` decodes, resamples to 16 kHz, splits into 512-sample chunks, and puts them on the `recv_audio_chunks_queue` for VAD.
2. **Speech detection**: VAD detects speech boundaries and emits `speech_started` / `speech_stopped` events on the `text_output_queue`. Full utterance audio goes to STT.
3. **Transcription**: STT output passes through `TranscriptionNotifier`, which taps the transcript for `transcription.delta` / `transcription.completed` events before forwarding to the LLM.
4. **Generation**: The LLM generates text (and optional tool calls). `LMOutputProcessor` splits the output: clean text goes to TTS, and `assistant_text` + tool call dicts go to the `text_output_queue`.
5. **Outbound audio**: TTS writes PCM chunks to `send_audio_chunks_queue`. The router's async `_send_loop` drains both queues, encoding PCM as `response.output_audio.delta` events and translating internal messages into protocol events.
6. **Session config**: `session.update` events deep-merge into `RuntimeConfig`, which is a shared Pydantic model read by VAD (turn detection thresholds), LLM (instructions, tools), and TTS (voice) at processing time.

---

## Supported OpenAI Realtime Events

### Client -> Server

| Event | Description |
|---|---|
| `input_audio_buffer.append` | Stream base64 PCM audio. Decoded, resampled to 16 kHz, and chunked for the VAD. |
| `session.update` | Deep-merge session config (instructions, tools, voice, turn detection, audio format). |
| `conversation.item.create` | Inject `input_text` or `function_call_output` into the LLM context without triggering generation. |
| `response.create` | Trigger LLM generation. Supports per-response `instructions` and `tool_choice` overrides. |
| `response.cancel` | Cancel the in-progress response and re-enable listening. |

### Server -> Client

| Event | Description |
|---|---|
| `session.created` | Sent on connection with current session config. |
| `error` | Protocol errors (`session_limit_reached`, `unknown_or_invalid_event`, `invalid_session_type`, `conversation_already_has_active_response`, etc.) |
| `input_audio_buffer.speech_started` | VAD detected user speech. |
| `input_audio_buffer.speech_stopped` | End of user speech segment. |
| `conversation.item.created` | Acknowledges injected `input_text` from `conversation.item.create`. |
| `conversation.item.input_audio_transcription.delta` | Streaming partial transcript (when live transcription is enabled). |
| `conversation.item.input_audio_transcription.completed` | Final transcript for the user turn (with duration usage). |
| `response.created` | Emitted on the first outbound audio chunk (response is `in_progress`). |
| `response.output_audio.delta` | Base64 PCM audio chunk from TTS. |
| `response.output_audio.done` | Audio stream complete for the current output item. |
| `response.output_audio_transcript.done` | Full assistant text transcript for the turn. |
| `response.function_call_arguments.done` | Tool call with `call_id`, `name`, and JSON `arguments`. |
| `response.done` | Response finished (`completed`, `cancelled` with reason `turn_detected` or `client_cancelled`). |

---

## Tool Calling Design

Tool calling works through two distinct paths depending on the LLM backend, but both converge to the same wire protocol for the client.

### Local LLM path (`LanguageModelHandler` -- transformers / mlx-lm)

Tools defined in `session.update` are converted to `FunctionTool` objects. Each tool's JSON Schema `parameters` are turned into a Python `inspect.Signature` (via `signature_from_schema`), and `to_code_prompt()` renders a human-readable `def name(...): """docstring"""` block.

These tool signatures are injected into the system prompt using a Jinja2 template (`tool_prompt.py`) that instructs the model to wrap tool calls in `<code>...</code>` delimiters:

```
<code>
function_name(arg_name_1=value1, arg_name_2='string_value')
</code>
```

After generation, `_extract_tools` uses a regex to find `<code>` blocks, then `extract_function_calls_from_text` parses each `name(kwargs)` call and validates it against registered tools. Valid calls become `ResponseFunctionToolCall` dicts with generated `call_id`s.

### OpenAI API path (`OpenApiModelHandler`)

Tools are passed natively as the `tools=` parameter to `client.responses.create`. The API returns structured `function_call` items directly -- no prompt engineering or regex parsing needed. Per-response `tool_choice` overrides from `response.create` are supported.

### Common output path

Both handlers yield `(text, language_code, tools)` tuples. `LMOutputProcessor` forwards clean text to TTS and puts `{"type": "assistant_text", "text": ..., "tools": [...]}` on the `text_output_queue`. The router's `_send_loop` translates these into:
- `response.output_audio_transcript.done` for the text
- `response.function_call_arguments.done` for each tool call

### Tool result flow

1. Client executes the tool and sends `conversation.item.create` with `type: "function_call_output"` and `output: "<result>"`
2. `RealtimeService` enqueues `("__FUNCTION_RESULT__", "<result>")` on `text_prompt_queue`
3. The LLM handler appends it as a user message (context only, no generation)
4. Client sends `response.create` to trigger the follow-up generation

---

## Interruption Handling

Barge-in (user speaks while the assistant is playing audio) is handled cooperatively between the VAD, the `_send_loop`, and the LLM/TTS handlers via a shared `CancelScope` object (`cancel_scope.py`).

### CancelScope design

`CancelScope` replaces the old two-signal pattern (`cancel_response` Event + `discard_stale_output` boolean) with a single object that manages:

- **Generation counter** (`cancel_scope.generation`): pipeline threads (LLM, TTS) capture the current generation at the start of each response and check `cancel_scope.is_stale(gen)` on every streaming token. When `cancel()` is called, the generation increments and all prior generations become stale -- no timing games required.
- **Discard flag** (`cancel_scope.discarding`): the async `_send_loop` checks this to drop stale audio/text that arrives between `cancel()` and `response_done()`.

```mermaid
sequenceDiagram
    participant User
    participant VAD
    participant SendLoop as _send_loop
    participant LLM
    participant TTS
    participant Client

    Note over TTS,Client: Assistant audio is playing (response_playing=set)
    User->>VAD: speaks
    VAD->>SendLoop: speech_started on text_output_queue
    SendLoop->>Client: input_audio_buffer.speech_started
    SendLoop->>Client: response.done (status=cancelled, reason=turn_detected)
    SendLoop->>SendLoop: cancel_scope.cancel() (gen++ & discarding=True)
    SendLoop->>SendLoop: flush output_queue + text_output_queue
    SendLoop->>SendLoop: response_playing.clear()
    LLM->>LLM: is_stale(gen) → True, aborts generation
    TTS->>TTS: is_stale(gen) → True, aborts generation
    TTS->>SendLoop: __RESPONSE_DONE__
    SendLoop->>SendLoop: cancel_scope.response_done() (discarding=False)
    Note over VAD,Client: Pipeline is now processing the new user utterance
```

**Step by step:**

1. **VAD detects speech**: puts `{"type": "speech_started"}` on `text_output_queue`.
2. **`_send_loop` processes text events first** (priority over audio): translates `speech_started` into protocol events. If an active response was in progress, `RealtimeService.dispatch_pipeline_event` emits `response.output_audio.done` + `response.done` with `status="cancelled"` and `reason="turn_detected"`.
3. **Cancel + queue flush**: if `response_playing` is set, the send loop calls `cancel_scope.cancel()` (increments generation, enables discard), drains both `output_queue` (preserving `__RESPONSE_DONE__` sentinels) and `text_output_queue`, then clears `response_playing`.
4. **LLM/TTS cancellation**: handlers capture `gen = cancel_scope.generation` at the start of each response and check `cancel_scope.is_stale(gen)` on every streaming token, aborting early when stale.
5. **Discard guard**: while `cancel_scope.discarding` is True, the send loop silently drops stale audio chunks and assistant text. The guard clears when `__RESPONSE_DONE__` arrives (via `cancel_scope.response_done()`).
6. **Client-initiated cancel**: `response.cancel` calls `cancel_scope.cancel()` (only if a response was active), triggers `finish_audio_response(status="cancelled", reason="client_cancelled")`, and re-enables `should_listen`.
7. **Spurious cancel safety**: if no response is active, `cancel_scope.cancel()` is not called, preventing the discard guard from being set without a `__RESPONSE_DONE__` to clear it.

---

## Testing

### Local LLM with Transformers

```bash
.venv/bin/python s2s_pipeline.py \
  --mode realtime \
  --stt parakeet-tdt \
  --llm transformers \
  --tts kokoro \
  --lm_model_name "Qwen/Qwen3-4B-Instruct-2507" \
  --lm_device mps \
  --lm_torch_dtype float16 \
  --enable_live_transcription
```

### Local LLM with MLX-LM

```bash
.venv/bin/python s2s_pipeline.py \
  --mode realtime \
  --stt parakeet-tdt \
  --llm mlx-lm \
  --tts kokoro \
  --lm_model_name "mlx-community/Qwen3-4B-Instruct-2507-bf16" \
  --lm_device mps \
  --lm_torch_dtype float16 \
  --enable_live_transcription
```

### Remote LLM with OpenAI-compatible API

```bash
.venv/bin/python s2s_pipeline.py \
  --mode realtime \
  --stt parakeet-tdt \
  --llm open_api \
  --tts kokoro \
  --open_api_model_name "openai/gpt-oss-20b:groq" \
  --open_api_base_url "https://router.huggingface.co/v1" \
  --open_api_api_key "$HF_TOKEN" \
  --open_api_stream \
  --enable_live_transcription
```