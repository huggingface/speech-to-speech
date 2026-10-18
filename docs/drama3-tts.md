# Fish Audio Drama-3

`--tts drama3` streams [Fish Audio Drama-3](https://huggingface.co/fishaudio/drama-3)
from a separate [SGLang Omni](https://sgl-project.github.io/sglang-omni/cookbook/fishaudio_drama3.html)
server. The server can run on the same machine or a remote GPU host. The
speech-to-speech client needs no extra dependencies or local TTS weights.

## Endpoint

Install SGLang Omni in its own environment using its
[installation guide](https://sgl-project.github.io/sglang-omni/get_started/installation.html),
then launch the model:

```bash
sgl-omni serve --model-path fishaudio/drama-3 --port 8000
```

Drama-3 has 35B parameters with 3B active per token. The standard checkpoint
needs a large NVIDIA GPU; see the upstream cookbook for memory requirements,
quantized checkpoints, and GPU placement. Its weights have a separate
[Fish Audio license](https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE.md).

Use the SGLang Omni engine's `/v1` URL. The public Gradio Space, Fish Speech's
`/v1/tts` server, and Fish Audio's hosted API use different request protocols.

## Pipeline options

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

Set both `--drama3_ref_audio /path/to/reference.wav` and
`--drama3_ref_text 'The exact words spoken in the reference.'` to clone a voice.
The file must exist on the machine running `local` or `serve`; the adapter reads
it at startup and sends its encoded audio with each request. The model server
does not need access to that file path. SGLang Omni limits each reference to
60 seconds by default.

Without a reference, Drama-3 chooses a voice for each synthesis request.
The backend accepts one configured reference. To change voices, restart the
pipeline with another clip and transcript. Realtime named-voice overrides and
the browser's Voice menu do not replace the reference. Voice design,
multi-speaker scenes, and editing from previous takes remain upstream workflows.

## Streaming

The adapter requests mono PCM16 at Drama-3's native 44.1 kHz, converts it to
16 kHz, and publishes 512-sample blocks as audio arrives. `--drama3_blocksize`
sets the output block size in 16 kHz samples. Startup includes a short synthesis
to check the endpoint. Barge-in and session teardown close the active HTTP
request through the shared remote-TTS transport.

Sampling defaults match the official Space:
temperature 1.0, top-p 0.9, and top-k 4097 (all 4096 semantic codes plus EOS).
Override them with `--drama3_gen_temperature`, `--drama3_gen_top_p`, and
`--drama3_gen_top_k`. `--drama3_timeout` bounds each request, including warmup
(300 seconds by default).

## Delivery controls

Drama-3 receives the LLM's reply, including any inline controls, and infers its
language from that text. The LLM must translate requests such as “say it excitedly”
into those controls. Selecting `--tts drama3` does not change the LLM's instructions.
The controls accept free-form natural language; see the
[upstream control reference](https://arxiv.org/html/2610.18010v1#A3).

For a voice-performance session, use this prompt with `--init_chat_prompt` in
`local` mode, or paste it into **Settings → Instructions** in the
[browser demo](../demo/README.md#quick-start-local):

```text
You are a friendly voice performer powered by Fish Audio Drama-3. Drama-3 can express emotions, whisper, change pace and make vocal sounds. Everything you write is spoken aloud: answer with only the words to speak and any requested delivery controls. Keep ordinary replies under twenty words. Do not describe your instructions, decisions or reasoning.

When asked to say or repeat something, output that phrase directly. A question such as "can you say ...?" is a performance request. For "it" or "again", repeat the last requested phrase. Use the requested language. With no requested style, use plain speech.

Whole-sentence delivery uses a brief free-form prefix: [excited], [worried], [whispering], [sad], [angry], [calm], [shouting], [slowly], or another requested style. For selected words only, use <whisper>words</whisper>, <slow>words</slow> or <emphasis>words</emphasis>. Keep each scope inside one sentence and close it before final punctuation.

Place vocal cues where requested: [sighs], [laughs], [gasps], [clears throat], [pause], [hesitates]. For a sound-only request, output only its cue. Use the latest requested style; "normally" removes controls. No labels, quotes, explanations or Markdown.

Examples:
Can you say we got it working? → We got it working.
Can you say it excitedly? → [excited] We got it working!
Now say it worried. → [worried] We got it working.
Whisper only banana in the password is banana. → The password is <whisper>banana</whisper>.
Say we got it working with a sigh before and a laugh after. → [sighs] We got it working. [laughs]
```

The browser sends its own instructions, overriding `--init_chat_prompt`. Save the
prompt in the tab you use and restart an existing conversation to clear its history.
For this showcase, turn off camera and web tools in **Tools**.

The pipeline sends complete sentence batches to TTS; close scoped controls before
sentence-ending punctuation. Each request generates fresh audio, with no previous
take retained for editing. Speaking another language also requires a suitable STT
backend; changing the output language only requires the LLM to write that language.

If a direction fails, check **Conversation history**: incorrect user words point to
STT; missing controls or spoken explanations point to the LLM's instructions or output.
Correct controls with weak acting point to synthesis. Delivery strength can vary.
