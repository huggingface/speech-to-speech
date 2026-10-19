# VAD

Turn detection still runs in `VADHandler`. The speech-probability source is selectable.

## Backends (`--vad`)

- `silero` (default) — Silero VAD via `VADIterator`
- `firered` — FireRed streaming VAD via `FireRedStreamVad` start and end events

FireRed uses `FireRedStreamVad` speech-start and speech-end events. It does not use Silero trigger or min-silence hangover logic.

Streaming FireRedVAD has no published benchmark yet. Silero remains the default.

## Progressive audio snapshots

`VADHandler` can emit cumulative audio snapshots while the user continues speaking.
This capability is independent of the downstream transcription backend. Pipeline
code can request a cadence by setting `transcription_window_samples` on the handler,
in samples at the VAD input sample rate. It is disabled by default (`None`). For example:

```python
vad_handler.transcription_window_samples = 120 * vad_handler.sample_rate
```

Once speech meets the normal acceptance threshold, VAD emits a
`VADAudio(mode="progressive")` after each additional interval of accumulated turn
audio since the previous snapshot. Snapshots contain the current accumulated
segment plus any retained prefix from reopened speech; they are not separate,
fixed-length segments. With short-fragment merging enabled, held fragments can be
stitched in at finalization. Consumers must therefore handle changed audio rather
than assume every snapshot is an exact prefix of the final audio.
Detection and capture continue, and the buffer is retained.
Taking a snapshot does not emit a speech-stop event or finalize the turn. The
normal speech-end path still emits the final audio. Snapshot progress is retained
across reopenings and reset for a new turn or session.

`enable_realtime_transcription` can schedule additional snapshots for live
transcription. The sample-based cadence also works when that setting is disabled.
Downstream consumers decide how to slice audio, reuse completed work, or publish
captions. The cadence neither caps VAD memory nor changes `max_speech_ms`.

Currently, pipeline wiring enables this cadence for the
[OpenAI-compatible HTTP STT window option](../../../docs/openai-compatible-stt.md#bounded-long-turns-opt-in).
Its upload limits, quiet-point cuts, and transcript cache belong to that STT handler.

## FireRedVAD

Install the optional extra and download Stream-VAD weights:

```bash
pip install "speech-to-speech[fireredvad]"
hf download FireRedTeam/FireRedVAD --local-dir ./pretrained_models/FireRedVAD
```

Then:

```bash
speech-to-speech local \
  --vad firered \
  --vad_firered_model_dir ./pretrained_models/FireRedVAD/Stream-VAD
```

`--vad_firered_use_gpu` runs the detector on CUDA. Audio is 16 kHz mono PCM, same as Silero.

Missing extra or `--vad_firered_model_dir` fails at handler setup, before a session is accepted.
