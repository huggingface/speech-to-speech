# VAD

Turn detection still runs in `VADHandler`. The speech-probability source is selectable.

## Backends (`--vad`)

- `silero` (default) — Silero VAD via `VADIterator`
- `firered` — FireRed streaming VAD via `FireRedStreamVad` start and end events

FireRed uses `FireRedStreamVad` speech-start and speech-end events. It does not use Silero trigger or min-silence hangover logic.

Streaming FireRedVAD has no published benchmark yet. Silero remains the default.

## Opening a turn on confident speech (opt-in, Silero)

By default a new turn opens once a segment holds `--min_speech_ms` (384 ms) of active
speech, where a chunk is active at `--thresh` to start and 0.15 below it to continue.
That single number trades two errors against each other: lower it and short noises open
turns, raise it and a quick "yes" is dropped.

`--min_confident_speech_ms` scores a new-turn candidate instead:

```text
confident_ms / min_confident_speech_ms + (active_ms - confident_ms) / min_speech_ms >= 1
```

`confident_ms` is the active speech at or above `--confident_thresh` (0.9). A few very
confident chunks open a turn on their own, enough ordinary ones still do, and mixtures add
up. This lets `--min_speech_ms` be raised against low-confidence noise without losing
short, clearly spoken replies:

```bash
speech-to-speech local --min_speech_ms 800 --min_confident_speech_ms 128
```

`--idle_after_s` tightens the rule when nobody has been talking. A new turn that starts
more than that many seconds after the last accepted speech or finished assistant response
uses `--idle_min_speech_ms` and `--idle_min_confident_speech_ms` in place of the two values
above:

```bash
  --idle_after_s 60 --idle_min_speech_ms 1600 --idle_min_confident_speech_ms 384
```

Short replies happen inside a conversation, so they keep the lenient rule; a stray sound
after a minute of silence has to look like a sentence. The cost is that a one-word wake-up
after a long pause may be ignored.

Only new turns are scored. Speech that continues a reopenable turn keeps
`--min_speech_continuation_ms`, and stitched short segments keep the plain rule. Every
scored candidate logs its score, active and confident time and idle seconds at INFO.
These options are implemented for Silero only and are rejected with FireRed. Speech
that is confidently human but not addressed to the assistant still opens a turn; this
separates speech from noise, not addressees.

## Progressive audio snapshots

`VADHandler.transcription_window_samples` sets an optional cadence for emitting
`VADAudio(mode="progressive")`, independently of the transcription backend or live
transcription setting. It is measured in input samples and defaults to `None` (disabled):

```python
vad_handler.transcription_window_samples = 120 * vad_handler.sample_rate
```

Once speech is accepted, snapshots are emitted after each additional interval of
audio since the previous snapshot. They contain the current segment plus any
retained prefix from reopened speech. Capture and buffering continue; snapshots
do not split speech or end the turn. Finalization can add held short fragments,
so consumers must handle changed audio rather than assume an exact final prefix.

The pipeline currently enables this capability for the
[OpenAI-compatible HTTP STT window option](../../../docs/openai-compatible-stt.md#bounded-long-turns-opt-in).

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
