# Streaming speaker activity

This demo shows **who spoke when** from a microphone or an audio file using the
Transformers implementation of Nemotron 3 Diarization. It supports up to eight
speaker slots, including simultaneous speakers. Speaker IDs follow first arrival
within a session; they are not names and do not identify someone across sessions.

The live terminal shows active speakers and the twelve most recently completed
segments. JSON Lines output is available for building a browser visualization or
saving results. File mode can additionally attach speaker candidates to Whisper
word timestamps.

## Setup

This example uses the public [Transformers implementation](https://github.com/huggingface/transformers/pull/49056)
and the model weights in the `refs/pr/1` revision of
[`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization).
The model and code are still being updated; use matching revisions until the
Transformers release and the model main branch contain the final files.

Use a dedicated environment from the repository root. This avoids changing the
main application's Transformers pin (particularly the macOS pin). The demo loads
the package directly from `src`, so it does not require the full voice-agent stack.

```bash
python -m venv .venv-diarization
source .venv-diarization/bin/activate
python -m pip install torch numpy librosa sounddevice rich
python -m pip install 'git+https://github.com/huggingface/transformers.git@4caba7a55a71993c72b6f7aeac8fdffff134b530'
```

The examples below use the model revision shown in the upstream usage snippets.
The loader checks for missing or unexpected weight keys so an incompatible model
revision fails instead of silently running with partially initialized weights.

## Microphone

```bash
PYTHONPATH=src python -m speech_to_speech.diarization.demo \
  --microphone \
  --model nvidia/Nemotron-3-Diarization --revision refs/pr/1 \
  --device cuda --dtype bfloat16 --streaming-mode ultra_low_latency
```

Press Ctrl-C to finish and flush the last audio window. `--input-device INDEX`
selects a microphone; list devices with `python -m sounddevice`. The microphone
must support the checkpoint's sample rate (currently 16 kHz). CPU/float32 is the
default; real-time throughput depends on hardware and has not been established by
the unit tests. The demo deliberately uses eager inference; compilation is not
required.

Input buffering is approximately 1.04 s for `low_latency`, 0.64 s for
`very_low_latency`, and 0.32 s for `ultra_low_latency`, **plus compute time**.
The actual value is read from the processor and printed at startup. A full audio
queue or device overflow stops the demo rather than silently dropping samples
and shifting speaker timestamps.

## Recording and transcript

```bash
PYTHONPATH=src python -m speech_to_speech.diarization.demo \
  --audio conversation.wav --realtime \
  --model nvidia/Nemotron-3-Diarization --revision refs/pr/1 \
  --device cuda --dtype bfloat16

PYTHONPATH=src python -m speech_to_speech.diarization.demo \
  --audio conversation.wav --json --transcribe openai/whisper-small \
  --model nvidia/Nemotron-3-Diarization --revision refs/pr/1 \
  --device cuda > conversation.jsonl
```

Files are converted to mono and resampled to the model rate. Omit `--realtime` to
run as fast as inference permits. File loading holds the recording in memory;
the streaming component itself keeps only an audio window and model cache.

JSON updates contain `processed_seconds`, `active_speakers`, newly completed
`segments` (`speaker`, `start`, `end`), and a `final` flag. Times are seconds from
the beginning of the input. Ongoing segments are emitted once they close, even if
they span many chunks. Silence has an empty active-speaker list. Raw frame activity
can produce short segments; `--threshold` controls the speech probability cutoff.

Optional transcription runs **after** file diarization, not live. `word` records
contain text, start/end times, and every temporally overlapping speaker. Multiple
IDs indicate ambiguous attribution; an empty list means unknown. This does not
separate overlapping voices or guarantee which speaker said a word. ASR words
with missing timestamps cause an explicit error. Transcript/audio content stays
local to inference; the commands download model files from the Hub.

## Use the component

```python
from speech_to_speech.diarization import StreamingDiarizer

diarizer = StreamingDiarizer.from_pretrained(
    "nvidia/Nemotron-3-Diarization",
    revision="refs/pr/1",
    device="cuda",
    dtype="bfloat16",
    streaming_mode="ultra_low_latency",
)
for block in audio_blocks:  # mono float arrays, continuous audio including silence
    for segment in diarizer.push(block, sample_rate=16000):
        print(segment)
    print(diarizer.active_speakers)
for segment in diarizer.finish():
    print(segment)
diarizer.reset()  # discard speaker identities before a different session
```

Use one instance per session and a single worker per instance. Feed each audio
sample once, in order. The component validates sample rate and shape, handles
processor overlap/look-ahead, carries the speaker cache, and zero-pads the final
analysis window while clipping output timestamps to the actual audio duration.

Diarization is a side channel alongside STT, not an STT backend: it produces
speaker activity rather than text. The standalone demo feeds continuous audio.
The conversation integration below instead streams only VAD-selected audio on a
background worker. Cumulative progressive VAD outputs must not be fed directly:
they repeat previously processed audio. Use `finish_utterance()` between disjoint
speech regions to flush the tail and preserve speaker identity without joining
unrelated waveforms into one spectrogram.

## Speaker-aware conversation (`serve` and `local`)

The integration can now be enabled on the real conversation pipeline. With the
supporting Transformers version installed, run:

```bash
speech-to-speech local --mac_optimal_settings --diarization
```

`--diarization` selects the official model PR revision with `low_latency`
streaming. The Mac preset selects MPS. Both `--mac_optimal_settings` and
`--mac-optimal-settings` are supported. To customize the model, use:

```bash
--diarization_model_name nvidia/Nemotron-3-Diarization \
--diarization_revision refs/pr/1 \
--diarization_device mps \
--diarization_dtype float32 \
--diarization_streaming_mode low_latency
```

Use `cpu` on an Intel Mac or `cuda` on an NVIDIA GPU. Omit
both `--diarization` and `--diarization_model_name` to disable the feature. `--device`, if provided,
overrides the diarization device too. An STT backend is required (`--stt none`
is rejected). Local and remote STT backends share the same metadata path.

The standalone demo environment above does not install the full voice-agent
stack. Until a supporting Transformers release is available, use the explicit
override below for the full pipeline:

```bash
uv venv .venv-diarized-pipeline
uv pip install --python .venv-diarized-pipeline/bin/python \
  --overrides examples/streaming-diarization/transformers-pr-overrides.txt -e .

.venv-diarized-pipeline/bin/speech-to-speech local --mac_optimal_settings --diarization
```

This overrides the macOS Transformers pin only in this environment. The override
pins the tested commit from Transformers PR #49056; update it when the upstream
implementation or published model revision changes.

The live path is:

```text
Incoming PCM → VAD ─┬→ speech chunks → background diarization → speaker labels
                   └→ final utterance → STT → transcript + ready speaker labels
                                            → LLM → TTS
```

VAD queues each selected chunk once, including its buffered speech onset and short
trailing silence. Diarization starts while someone is speaking; final audio and
cumulative progressive STT buffers are never replayed to it. Long idle silence
is skipped. Short gaps retained by VAD for stitching speech fragments can still
be processed as context.

The dedicated worker performs all inference and utterance flushing. VAD never
waits for it. At a final boundary the remaining model look-ahead is flushed, with
zero-padding only for the final analysis window. Speaker cache is retained, but
spectrogram windows and local frame counters restart at the next onset. Absolute
sample offsets map selected regions back to VAD audio, including skipped gaps
and overlapping pre-roll. Each pipeline owns one model/worker; disconnect
invalidates old pending results immediately, and only the worker resets its
model before handling new-session audio.

The model is warmed before capture, with synthetic speaker state discarded. The
worker is managed with the pipeline threads and stops on pipeline shutdown.

The LLM receives anonymous `speaker_0`, `speaker_1`, etc. labels alongside the
transcript and an explanation that they are session-local metadata, not words
to read aloud. The client-visible transcription remains the recognized text.
A single-speaker utterance is labelled directly. If several speakers occur in
one VAD utterance, the LLM is explicitly told that individual words are **not**
attributed. The live path does not run a second ASR model, split voices, or invent
word timestamps; the file demo's optional word alignment remains separate.

Each final VAD message carries a promise tied to its audio and session. STT
resolves it **after transcription finishes**, with **zero additional wait**.
If the tail is still flushing, a detached snapshot of already-scored activity is
used and marked incomplete. If the worker has not reached that boundary yet,
attribution is unknown. A late result never changes a response that has already
started. Completed
results become detached snapshots. Reopened turns combine their earlier promises
and replace the previous transcript in conversation history.

The worker queue is bounded to 128 blocks (each at most 100 ms; normally VAD
supplies 32 ms). Producers never block. An overflow or inference error disables
speaker attribution for the rest of that session, rather than dropping samples
and silently changing identities. New sessions start cleanly. Activity history
is bounded to 120 seconds; unretained audio is marked incomplete. Speaker-aware
reply targeting and speaker fields in the public Realtime protocol are not
implemented.

## Validation

```bash
pytest tests/test_streaming_diarization.py tests/test_diarization_transformers.py \
  tests/test_diarization_worker.py tests/openai_realtime/test_speaker_conversation.py -q
```

The first module checks chunk boundaries, overlaps, cache resets, short/final
audio, transcript alignment, JSON output, and microphone overflow with deterministic
fixtures. The second runs real Transformers processors and small randomly
initialized models for all three streaming modes; it skips if that API is not
installed. The conversation tests exercise PCM → VAD → STT metadata → LLM history,
including distinct speakers, revised transcripts, reset, mixed speakers, and
missing activity. These tests require no checkpoint download.

An earlier preview checkpoint was smoke-tested on eight seconds of the bundled
`src/speech_to_speech/TTS/ref_audio.wav`, producing one speaker segment with valid
timestamp bounds. The official checkpoint at `refs/pr/1` loaded with the exact
Transformers PR commit above and processed a two-second silent input on CPU.
Multi-speaker accuracy, noisy microphones, and GPU throughput still need
evaluation on representative recordings.

The trained checkpoint has also been exercised through the real VAD stage,
confirming that finalized utterances carry speaker metadata and session teardown
invalidates pending results and clears model state on the worker. Tests cover
blocked inference without blocking VAD/STT, queue overflow, disconnect during
inference, skipped silence, pre-roll deduplication, and cache continuity across
utterances. Conversation-routing tests use deterministic speech and
transcription fixtures rather than making external LLM requests.

A paced 12-second smoke recording on this Mac (two copies of a four-second speech
sample separated by silence) sent 7.36 seconds to the diarization model and kept
`speaker_0` across both utterances. With MPS diarization in the background, VAD
calls had a 1.35 ms median and 2.19 ms 95th percentile in that run. Final labels
were ready about 104–280 ms after the VAD boundary; the response path does not
wait for them. This is an isolated smoke measurement, not a full STT/LLM/TTS
contention or multi-speaker accuracy benchmark.
