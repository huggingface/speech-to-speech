# VAD

Turn detection still runs in `VADHandler`. The speech-probability source is selectable.

## Backends (`--vad`)

- `silero` (default) — Silero VAD via `VADIterator`
- `firered` — FireRed streaming VAD via `FireRedStreamVad` start and end events

FireRed uses `FireRedStreamVad` speech-start and speech-end events. It does not use Silero trigger or min-silence hangover logic.

Streaming FireRedVAD has no published benchmark yet. Silero remains the default.

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
