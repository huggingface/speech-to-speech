# Speech to visemes

This optional stage adapts Fabio Catania's (@fabiocat93)
[PR #99](https://github.com/huggingface/speech-to-speech/pull/99) for the current
pipeline and addresses [issue #37](https://github.com/huggingface/speech-to-speech/issues/37).
It retains the original phoneme-to-viseme map and Wav2Vec2 extraction approach.

Enable it with your existing server or local configuration:

```bash
speech-to-speech serve --enable_visemes --stv_device cpu
```

The default checkpoint is `bookbot/wav2vec2-ljspeech-gruut`. It downloads on first
use and is trained on English phonemes. Its [model card](https://huggingface.co/bookbot/wav2vec2-ljspeech-gruut)
describes the training data and phoneme vocabulary. The mapping includes other
languages, but this does not make the checkpoint multilingual. Select a compatible
16 kHz CTC phoneme checkpoint with `--stv_model_name` for other languages.
`--stv_device auto` chooses an available CUDA, MPS, or CPU device; the global
`--device` overrides it. The stage is absent by default, so disabling visemes
requires no additional model download, memory, warmup, or buffering.

## Client events

The stage adds a JSON extension event on the existing Realtime WebSocket or
WebRTC data channel. It adds no endpoints and uses no pickle serialization.
The standard audio events and RTP audio are unchanged.

```json
{
  "type": "speech_to_speech.output_audio.visemes",
  "event_id": "event_...",
  "response_id": "resp_...",
  "item_id": "item_...",
  "output_index": 0,
  "content_index": 0,
  "visemes": [
    {"viseme": 21, "start_s": 0.02, "end_s": 0.08},
    {"viseme": 6, "start_s": 0.08, "end_s": 0.15}
  ]
}
```

IDs 0 through 21 use the original Microsoft-compatible mouth-shape map. Each cue
has an interval in seconds relative to the start of the response's generated
audio, across all its audio items. The offset continues across inference batches
and resets for the next response. `item_id` identifies the assistant item carrying
that batch. Compound phonemes produce consecutive mouth-shape intervals. IPA
stress marks are removed before mapping. Unmapped phonemes produce no cue.

Events precede their corresponding audio. Schedule shapes against actual audio
playback, including any client buffering, rather than the event's arrival time.
On interruption or response cancellation, discard scheduled shapes with buffered
audio and return the mouth to rest. Ignore extension events if your client only
needs audio. Output resampling changes sample counts but not these time units.

The browser `S2sRealtimeClient` exposes the event for avatar consumers:

```javascript
client.addEventListener("visemes", ({ detail }) => {
  avatar.queueVisemes(detail.response_id, detail.visemes);
});
```

This hook forwards cues; the avatar owns its animation and playback clock.

## Buffering and failure behavior

Extraction buffers up to 0.5 seconds of audio plus model inference time before
sending each batch. Short replies and final tails flush at response completion
or before ordered text/tool events. Inference may be less accurate on short
segments. Padding very short inputs affects model inference only; client audio
samples are preserved exactly.

Extraction failures log an error and preserve audio playback and completion.
Listening, interruption, and response completion remain controlled by the
current pipeline. Cancellation drops stale audio and cues together. Session end
clears all buffered audio and timing state before the unit can be reused.
