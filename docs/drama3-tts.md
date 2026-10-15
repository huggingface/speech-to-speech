# Fish Audio Drama-3

`--tts drama3` streams [Fish Audio Drama-3](https://huggingface.co/fishaudio/drama-3)
from a separate [SGLang Omni](https://sgl-project.github.io/sglang-omni/cookbook/fishaudio_drama3.html)
server. The server can run on the same machine or a remote GPU host. The
speech-to-speech client needs no extra dependencies or local TTS weights.

## Start the model server

Install SGLang Omni in its own environment, following its
[installation guide](https://sgl-project.github.io/sglang-omni/get_started/installation.html),
then launch the model:

```bash
sgl-omni serve --model-path fishaudio/drama-3 --port 8000
```

Drama-3 has 35B parameters with 3B active per token. The standard checkpoint
needs a large NVIDIA GPU; see the upstream cookbook for memory requirements,
quantized checkpoints, and GPU placement. Its weights have a separate
[Fish Audio license](https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE.md).

## Use it in the pipeline

```bash
speech-to-speech local \
  --tts drama3 \
  --drama3_base_url http://localhost:8000/v1
```

Use `serve` instead of `local` for a Realtime server without local microphone
and speaker playback. The other pipeline stages keep their usual defaults.

For an authenticated endpoint, set `DRAMA3_API_KEY` or pass `--drama3_api_key`.
The adapter does not use `OPENAI_API_KEY`. `--drama3_model` selects the model ID
sent to the server, defaulting to `fishaudio/drama-3`.

## Keep a consistent voice

Without a reference, Drama-3 chooses a voice for each synthesis request. To
keep the same speaker across replies, supply a short reference clip and its
transcript:

```bash
speech-to-speech local \
  --tts drama3 \
  --drama3_ref_audio /voices/speaker.wav \
  --drama3_ref_text "The exact words spoken in the reference clip."
```

The file lives on the speech-to-speech machine. The adapter reads it at startup
and sends its encoded audio with each request; the model server does not need
access to that path. Both reference options are required together. SGLang Omni
limits each reference to 60 seconds by default. Realtime named-voice overrides
do not replace the configured reference.

## Streaming and delivery

The adapter requests mono PCM16 at Drama-3's native 44.1 kHz, converts it to
16 kHz, and publishes 512-sample blocks as audio arrives. Startup includes a
short synthesis to check the endpoint. Barge-in and session teardown close the
active HTTP request through the shared remote-TTS transport.

Drama-3 infers language from the text. Inline acting tags such as `[laughs]`
pass through unchanged. Sampling defaults match the official Space:
temperature 1.0, top-p 0.9, and top-k 4097. Override them with
`--drama3_gen_temperature`, `--drama3_gen_top_p`, and `--drama3_gen_top_k`.
`--drama3_timeout` bounds each request, including warmup (300 seconds by default).

Use the SGLang Omni engine's `/v1` URL. The public Gradio Space, Fish Speech's
`/v1/tts` server, and Fish Audio's hosted API use different request protocols.
