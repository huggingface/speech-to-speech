# Speaker-aware live conversations

Speech-to-speech can attach speaker activity from
[Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)
to transcripts in the `local` and `serve` pipelines. The model identifies up
to eight anonymous speaker slots within a session. It does not transcribe speech
or establish a speaker's real-world identity.

## Setup

Transformers 5.18.0 includes the model implementation from
[PR #49056](https://github.com/huggingface/transformers/pull/49056).
The application requires that release or newer and pins 5.18.0 on macOS.
Install normally; no source checkout or dependency override is needed:

```bash
uv venv .venv-diarized-pipeline
uv pip install --python .venv-diarized-pipeline/bin/python -e .

.venv-diarized-pipeline/bin/speech-to-speech local --mac-optimal-settings --diarization
```

`--diarization` selects the official model's main branch and
`low_latency` streaming mode. Diarization defaults to `auto`, selecting CUDA,
then MPS, then CPU according to availability. The Mac preset selects MPS.
`--device` overrides the diarization device along with other components;
`--diarization_device` can select a device for diarization alone.
CUDA or MPS is recommended for live sessions. CPU may keep up initially but
fall behind as the retained speaker cache grows during sustained speech,
overflowing the worker queue. Explicit `--diarization_device cpu` remains
available, and startup logs warn whenever CPU is selected.
The model loads once **per pipeline**, so
increasing the pipeline count also increases model memory usage. An STT backend
is required. Omit both `--diarization` and `--diarization_model_name` to
disable speaker attribution.

The model and streaming settings can be changed with:

```bash
--diarization_model_name nvidia/Nemotron-3-Diarization \
--diarization_device mps \
--diarization_dtype float32 \
--diarization_streaming_mode low_latency
```

The model loader rejects missing or unexpected weight keys, avoiding silent
use of an incompatible checkpoint.

## How it works

```text
Incoming PCM → VAD ─┬→ selected speech chunks → background diarization
                   └→ final utterance → STT → transcript + speaker labels → LLM → TTS
```

VAD sends each selected audio chunk once to a dedicated diarization worker.
Long idle silence and cumulative progressive STT buffers are not replayed.
At each utterance boundary the worker flushes the model's look-ahead and
restarts its audio windows while retaining its speaker cache. That cache is
cleared at session end. STT does not wait for diarization: it uses the labels
already available when transcription finishes.

The LLM receives a compact prefix, such as `[speaker_0] Hello.` or
`[speaker_0, speaker_1; words not attributed] Hello.`. `partial` means the
model has not finished scoring all audio for that utterance. With no reliable
label, the LLM receives the plain transcript. The client-visible transcript
never includes speaker metadata. Mixed-speaker turns have no word-level
attribution; the live path does not run a second ASR model or split voices.

The worker queue is bounded. Queue overflow or inference failure disables
speaker labels for the **rest of the session**, until the client reconnects.
Late results never change an answer already started. The logs show
`Diarization at transcription` with speaker durations and timing, without
adding a diarization wait to the response path.

## Validation

```bash
pytest tests/test_streaming_diarization.py tests/test_diarization_transformers.py \
  tests/test_diarization_worker.py tests/openai_realtime/test_speaker_conversation.py -q
```

These tests cover streaming boundaries, cache retention, device selection,
VAD integration, speaker metadata delivery, and failure handling. The
Transformers-specific tests run against the released model implementation
using small, randomly initialized weights. No checkpoint is downloaded.
