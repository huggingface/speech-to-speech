# Stateful Qwen3-ASR with vLLM

A standalone API for applications that own their audio transport, VAD and turn
policy. One warmed engine serves independent 16 kHz mono PCM16 streams through
`qwen-asr`'s stateful vLLM backend. The framework's existing STT backends are
unchanged.

## Install and use

Use a separate, compatible Linux CUDA environment for inference:

```bash
uv sync --project packages/qwen-vllm-stt --extra vllm --no-dev
```

Only the `vllm` extra installs the inference runtime. The API targets
`qwen-asr` 0.0.6 with vLLM 0.14.0. That backend pins Transformers 4.57.6, which
cannot satisfy the full framework's Transformers 5 requirement. Keep the native
inference component in a separate environment.

The CUDA path has been exercised on Linux x86_64, Python 3.12 and an H100 with
`Qwen/Qwen3-ASR-0.6B`. The base package's Python test matrix does not imply that
every interpreter is supported by the optional CUDA dependencies.

```python
from hf_s2s_qwen_vllm import QwenVllmEngine, QwenVllmSettings

engine = QwenVllmEngine(
    QwenVllmSettings(
        model="Qwen/Qwen3-ASR-0.6B",
        language="English",
        gpu_memory_utilization=0.8,
        max_new_tokens=32,
        unfixed_chunk_num=2,
        unfixed_token_num=5,
        chunk_size_sec=0.5,
        stream_step_ms=250,
        max_segment_sec=30,
        inference_timeout_s=8,
        inference_rt_margin=3,
        max_concurrency=1,
        max_streams=64,
    )
)


async def transcribe(audio_chunks):
    stream = await engine.new_stream()
    try:
        async for pcm16 in audio_chunks:
            for update in await stream.push(pcm16):
                print(update.text, update.final, update.forced)
        for update in await stream.commit():
            print(update.text, update.final, update.forced)
    finally:
        await stream.close()
```

Await `push()` in audio order and call `commit()` at the intended boundary.
Partials arrive during audio delivery as full transcript snapshots; later
partials may revise earlier words. A segment limit produces a marked forced
final; every explicit commit still returns one final, including an empty final
when no audio follows a forced boundary.

`model` accepts a repository ID or local snapshot directory. Download and verify
a pinned snapshot before startup when reproducible model assets are required.

## Lifecycle and resource limits

Keep an engine on one event loop. `start()` loads and warms its model before
accepting requests; `new_stream()` also awaits this shared startup. Cancelling a
startup waiter does not cancel model loading. The hosting service owns the
startup deadline and must abort a process that cannot become ready.

`max_streams` bounds open streams; `max_concurrency` bounds simultaneous backend
calls across those streams. A stream retains its reservation until closed or
failed. Start with `max_concurrency=1`; larger values require backend validation.
The caller must bound its input queues and close every stream.

Decode/finalize deadlines include queue time and use
`inference_timeout_s + inference_rt_margin * segment_duration`. A zero base
timeout disables the deadline. Expiry before inference raises
`QwenVllmAdmissionError` without poisoning the model. Expiry during inference
makes `health()` unhealthy and requires a process restart. Cancelled blocking
work retains its inference slot until drained; cancellation or failure after
entering a stream operation closes that stream to protect its mutable state.

`close()` discards late output and waits for the current operation. Cancel the
operation's task first for immediate teardown. Neither action stops an executing
GPU kernel. Input must be little-endian PCM16 with complete samples; each frame
must fit the positive `max_segment_sec` limit. The limit is enforced across
frames; zero disables it. The caller owns session identity and result delivery.

## Development

```bash
uv sync --project packages/qwen-vllm-stt --group dev
uv run --project packages/qwen-vllm-stt pytest packages/qwen-vllm-stt/tests -q
```

Tests use a deterministic model double to exercise stream isolation, sample
delivery, admission, cancellation and deadlines. They do not establish ASR
quality or capacity; validate the intended model, decoding settings and offered
load on the target GPU runtime.
