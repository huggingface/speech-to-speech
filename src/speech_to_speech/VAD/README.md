# VAD

Turn detection still runs in `VADHandler`. The speech-probability source is selectable.

## Backends (`--vad`)

- `silero` (default) — Silero VAD via `VADIterator`
- `firered` — FireRed streaming VAD, same trigger/silence/pad logic as Silero

Streaming FireRedVAD has no published benchmark yet. Silero remains the default.

## FireRedVAD

Install the optional extra and download Stream-VAD weights:

```bash
pip install "speech-to-speech[fireredvad]"
huggingface-cli download FireRedTeam/FireRedVAD --local-dir ./pretrained_models/FireRedVAD
```

Then:

```bash
speech-to-speech local \
  --vad firered \
  --vad_firered_model_dir ./pretrained_models/FireRedVAD/Stream-VAD
```

`--vad_firered_use_gpu` runs the detector on CUDA. Audio is 16 kHz mono PCM, same as Silero.

Missing extra or `--vad_firered_model_dir` fails at handler setup, before a session is accepted.
