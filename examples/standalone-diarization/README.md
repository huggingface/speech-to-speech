# Streaming speaker activity

This standalone example uses the streaming component introduced in
[PR #583](https://github.com/huggingface/speech-to-speech/pull/583).

This demo shows **who spoke when** from a microphone or an audio file using the
Transformers implementation of Nemotron 3 Diarization. It supports up to eight
speaker slots, including simultaneous speakers. Speaker IDs follow first arrival
within a session; they are not names and do not identify someone across sessions.

The live terminal shows active speakers and the twelve most recently completed
segments. JSON Lines output is available for building a browser visualization or
saving results. File mode can additionally attach speaker candidates to Whisper
word timestamps.

## Setup

This example uses the merged [Transformers implementation](https://github.com/huggingface/transformers/pull/49056)
and the weights on the main branch of
[`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization).
Until a Transformers package release includes this model, install the tested
merge commit below.

Use a dedicated environment from the repository root. This avoids changing the
main application's Transformers pin (particularly the macOS pin). The demo loads
the package directly from `src`, so it does not require the full voice-agent stack.

```bash
python -m venv .venv-diarization
source .venv-diarization/bin/activate
python -m pip install torch numpy librosa sounddevice rich
python -m pip install 'git+https://github.com/huggingface/transformers.git@8f080cb5aa480f794283c6e8618f917cc9a6506d'
```

The examples below use the model's main branch. The loader checks for missing
or unexpected weight keys so an incompatible model revision fails instead of
silently running with partially initialized weights.

## Microphone

```bash
PYTHONPATH=src python -m speech_to_speech.diarization.demo \
  --microphone \
  --model nvidia/Nemotron-3-Diarization \
  --device cuda --dtype bfloat16 --streaming-mode low_latency
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
  --model nvidia/Nemotron-3-Diarization \
  --device cuda --dtype bfloat16

PYTHONPATH=src python -m speech_to_speech.diarization.demo \
  --audio conversation.wav --json --transcribe openai/whisper-small \
  --model nvidia/Nemotron-3-Diarization \
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
