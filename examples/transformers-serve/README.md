# Speech-to-speech on `transformers serve`

This example runs every model of the voice loop in one `transformers serve`
process, through its OpenAI-compatible endpoints:

```text
speech-to-speech      VAD + Smart Turn -> STT ------------------> LLM ----------------> TTS -> Realtime client
transformers serve                        /v1/audio/transcriptions /v1/chat/completions /v1/audio/speech
                                          openai/whisper-small.en  Qwen/Qwen3-0.6B      gradium/phonon-31m-en
```

speech-to-speech keeps turn-taking, interruption, conversation state and the
Realtime API. It uses its existing `openai` STT, `chat-completions` LLM and
`openai` TTS backends, so no code change is needed on either side.

`/v1/audio/speech` comes from
[huggingface/transformers#49681](https://github.com/huggingface/transformers/pull/49681),
which is not merged yet. [Gradium Phonon](https://huggingface.co/gradium/phonon-31m-en)
31M is Apache-2.0 and is not gated.

## Install

Use two environments: speech-to-speech pins its own Transformers release.

```bash
python3 -m venv .venv-serve
.venv-serve/bin/pip install \
  "transformers[serving,audio] @ git+https://github.com/huggingface/transformers@refs/pull/49681/head" \
  python-multipart soundfile
```

`python-multipart` is needed for `/v1/audio/transcriptions` and is not part of
the `serving` extra. Install speech-to-speech as in the
[Quickstart](../../README.md#quickstart), in its own environment.

## Run

Start the model server. Models load on their first request, and the pipeline
warms each one up at startup.

```bash
.venv-serve/bin/transformers serve --dtype float32 --model-timeout 3600
```

`--dtype float32` follows Phonon's CPU recommendation; it applies to every
model, so leave it out on a GPU.

Then talk to it with a microphone and speakers, from the repository root:

```bash
speech-to-speech local -f examples/transformers-serve/transformers-serve.yaml
```

Or run the server and the scripted conversation, which needs no audio device:

```bash
speech-to-speech serve -f examples/transformers-serve/transformers-serve.yaml
python examples/transformers-serve/conversation_demo.py --out-dir demo-out
```

The script synthesizes each user turn with Phonon in a catalog voice (Zoey),
streams it to the Realtime WebSocket at real-time pace, and writes
`conversation.wav`, `conversation.md` and `turns.json` with per-turn timings.

## Phonon through `/v1/audio/speech`

As of the pull request:

- Speech is not streamed. The configuration requests whole WAV replies
  (`openai_tts_response_format: wav`, `openai_tts_stream: false`). With the
  default `pcm` format, speech-to-speech sends `stream_format=audio`, which the
  server rejects with `400 This model does not support streaming audio.` Each
  sentence is therefore heard only once all of it is synthesized.
- `voice` is ignored: the assistant always uses the checkpoint's voice, and
  `--openai_tts_voice`, the Realtime `session.audio.output.voice` and the
  script's `--assistant-voice` have no effect. A catalog voice can only be
  passed as `ref_audio`, a path to its `.safetensors` file on the server's own
  disk, which is how the script gives the user a different voice.
- Fields outside OpenAI's speech request are rejected with 422, so leave
  `openai_tts_language` and `openai_tts_task_type` unset.

## Measured

The scripted five-turn conversation, on a 4-core x86 CPU container with 15 GB
of RAM and no GPU, all models in float32. Stage times are the server's turn
log. First audio is measured by the client from the end of the user's audio.
Only the scripted conversation was run there, not `local` with a microphone.

| Turn | STT | LLM (full reply) | TTS first audio | Client first audio |
|---|---:|---:|---:|---:|
| 1 | 1.51 s | 2.81 s | 1.19 s | 3.90 s |
| 2 | 1.52 s | 2.10 s | 1.06 s | 4.35 s |
| 3 | 1.33 s | 3.70 s | 1.67 s | 4.61 s |
| 4 | 1.48 s | 3.15 s | 1.32 s | 4.15 s |
| 5 | 1.53 s | 2.72 s | 1.16 s | 5.05 s |

Every transcript was correct, and Whisper transcribes the recorded replies back
to the assistant's text. Qwen3-0.6B answered the factual turns, but did not
recall how the user takes their tea; use a larger model, such as the default
Qwen3.5-4B, on a GPU or Apple Silicon. The non-continuous-batching
path of `transformers serve` does not reuse the KV cache between requests, so
the whole conversation is prefilled again on every turn.
