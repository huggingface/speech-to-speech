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

## Clone a voice: record, listen, and talk

Without a reference, Drama-3 chooses a voice for each synthesis request. To
keep the same speaker across replies, supply a short reference clip and its
exact transcript. No training step is needed.

The commands below use Bash or zsh, an installed `speech-to-speech`, and a
running model server from the previous section. Recording and conversion use
[FFmpeg](https://ffmpeg.org/download.html). Choose a clear 5–15 second clip with
one speaker and no background music.

### 1. Record a reference or use an existing clip

Create a directory for this voice and set your model server's URL:

```bash
S2S_VOICE_DIR="$HOME/s2s-voices/my-voice"
S2S_DRAMA_URL="http://localhost:8000/v1"
mkdir -p "$S2S_VOICE_DIR"
```

On **macOS**, run this in Terminal and allow microphone access if prompted.
Once the recording timer starts, read: “Hello there. This is my natural speaking
voice. I like building things that make conversations feel easy.” Recording
stops after twelve seconds.

```bash
test ! -e "$S2S_VOICE_DIR/reference.wav" &&
ffmpeg -hide_banner -n -f avfoundation -i "none:default" \
  -t 12 -ar 44100 -ac 1 -c:a pcm_s16le "$S2S_VOICE_DIR/reference.wav" &&
  printf '%s\n' \
  'Hello there. This is my natural speaking voice. I like building things that make conversations feel easy.' \
  > "$S2S_VOICE_DIR/reference.txt"
```

On **Linux with PulseAudio or PipeWire's PulseAudio interface**, use
`-f pulse -i default` in place of `-f avfoundation -i "none:default"` and read
the same sentence.

Alternatively, record with your usual audio app or use an existing clip. Convert
it to a mono WAV, then put its exact spoken words in `reference.txt`:

```bash
test ! -e "$S2S_VOICE_DIR/reference.wav" &&
ffmpeg -hide_banner -n -i /path/to/your-clip.m4a \
  -ar 44100 -ac 1 -c:a pcm_s16le "$S2S_VOICE_DIR/reference.wav" &&
  printf '%s\n' 'The exact words spoken in your clip.' \
  > "$S2S_VOICE_DIR/reference.txt"
```

Choose one method. These commands leave an existing recording untouched; use
a new voice directory for another take. Listen to the clip before continuing.
If you changed or missed words, correct the transcript or record another take.
Keep the WAV and transcript together so you can reuse the voice later.

### 2. Start a conversation with that voice

```bash
speech-to-speech local \
  --tts drama3 \
  --drama3_base_url "$S2S_DRAMA_URL" \
  --drama3_ref_audio "$S2S_VOICE_DIR/reference.wav" \
  --drama3_ref_text "$(cat "$S2S_VOICE_DIR/reference.txt")" \
  --stream_batch_sentences 1 \
  --init_chat_prompt 'Keep replies to one short sentence. Reply in the language the user requests. For requested sighs or laughter, include [sighs] or [laughs] in the spoken text. Preserve those tags when repeating text; do not explain them.'
```

This uses the usual local STT and LLM defaults; add your existing STT/LLM flags
if you use different backends. Wear headphones, speak, and pause for the reply.

The reference file must live on the machine running `local` or `serve`. If you
recorded on another machine, copy the WAV and transcript there first. The
adapter reads the WAV at startup and sends its encoded audio with each request;
the separate model server does not need access to that file path. Both reference
options are required together. SGLang Omni limits each reference to 60 seconds
by default.

To change voices, stop the pipeline with `Ctrl+C`, choose another reference and
transcript, and rerun the command. Realtime named-voice overrides and the browser's
Voice menu do not replace the configured reference.

### 3. Try a short demo

| Say to the assistant | What to listen for |
|---|---|
| “Say: We finally got it working.” | A reply using the reference voice. |
| “Say that again, with a sigh before the sentence and a laugh afterwards.” | The LLM emits `[sighs]` and `[laughs]`; Drama-3 receives them as acting cues. |
| “Now say it in Spanish.” | The requested language with the same reference. |

The language demo changes the **output** language. Speaking another language
into the microphone also requires an STT backend/model that supports it.
Acting cues depend on the LLM emitting the tags; inspect the assistant transcript
if a direction is missing.

For a browser recording, replace `local` with `serve` and follow the
[browser demo setup](../demo/README.md#quick-start-local). Paste the same
instructions into **Settings → Instructions**, since the browser sends its own
session instructions. Conversation history exposes **Server timings**; the TTS
time to first audio is only one part of the full response latency.

## More voice workflows

You can design a character voice from a description in the
[official Space](https://huggingface.co/spaces/fishaudio/Drama-3), download a
short generated take, and use that WAV plus its spoken words as the reference
above. This lets you reuse a designed voice in conversation without sending
voice-design instructions through the adapter.

The upstream model also supports multi-speaker scenes and editing from previous
takes. This backend currently configures one reference and does not expose a
speaker roster, voice-design parameters, or retake history. Those workflows
remain available in upstream tools rather than the speech-to-speech Voice menu.

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
