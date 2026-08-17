"""Whisper-family STT handlers must tag progressive audio as PartialTranscription.

`BaseSTTHandler.before_emit_output` marks a turn revision complete as soon as it sees a
`Transcription`, and `should_process_input` then rejects every later chunk of that
revision as "input-after-final". A handler that emits `Transcription` for a progressive
chunk therefore closes the turn while the user is still speaking, and the full utterance
is discarded: the LLM only ever receives the first fragment.

`ParakeetTDTSTTHandler` branches on `vad_audio.mode` and yields `PartialTranscription`
for progressive audio. The four Whisper-family handlers did not, so they truncated every
turn in `--mode realtime` for any language Parakeet does not cover.

Only VAD emits progressive chunks, and only when live transcription is enabled
(`s2s_pipeline` sets `enable_realtime_transcription` from `--enable_live_transcription`,
and `vad_handler` gates the progressive yield on it), so branching on `mode` alone is
sufficient here -- the handlers need no flag of their own.

No Whisper, faster-whisper, MLX, or model download is needed: every model is faked, and
the Darwin-only / extra-only backends are stubbed so all four handlers are covered on
Linux CI as well as macOS.
"""

from __future__ import annotations

import importlib
import sys
import types
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest

from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, VADAudio

SAMPLE_RATE = 16000

# (mode, seconds of speech captured so far, what the model returns for it)
CHUNKS = [
    ("progressive", 1, "one"),
    ("progressive", 2, "one two"),
    ("final", 3, "one two three"),
]
FULL_UTTERANCE = CHUNKS[-1][2]

TRANSCRIPTS = {int(SAMPLE_RATE * seconds): text for _, seconds, text in CHUNKS}


def _ensure_module(name: str, **attrs: object) -> None:
    """Make `name` importable, faking it when the optional backend is not installed."""
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
    except ImportError:
        module = types.ModuleType(name)
        for attr, value in attrs.items():
            setattr(module, attr, value)
        sys.modules[name] = module


def _prepare(handler):
    handler.queue_in = Queue()
    return handler


def build_whisper(monkeypatch):
    from speech_to_speech.STT import whisper_stt_handler
    from speech_to_speech.STT.whisper_stt_handler import WhisperSTTHandler

    monkeypatch.setattr(whisper_stt_handler.console, "print", lambda *a, **k: None)

    # Row 0 is [number of audio samples, language token]: the handler reads index 1 for
    # the language code and the fake decoder reads index 0 to pick the transcript.
    handler = object.__new__(WhisperSTTHandler)
    handler.gen_kwargs = {}
    handler.start_language = "en"
    handler.last_language = "en"
    handler.prepare_model_inputs = lambda audio: audio
    handler.model = SimpleNamespace(generate=lambda features, **kw: np.array([[len(features), 0]]))
    handler.processor = SimpleNamespace(
        tokenizer=SimpleNamespace(decode=lambda token: "<|en|>"),
        batch_decode=lambda ids, **kw: [TRANSCRIPTS[int(ids[0][0])]],
    )
    return _prepare(handler)


def build_faster_whisper(monkeypatch):
    _ensure_module("faster_whisper", WhisperModel=object)
    from speech_to_speech.STT import faster_whisper_handler
    from speech_to_speech.STT.faster_whisper_handler import FasterWhisperSTTHandler

    monkeypatch.setattr(faster_whisper_handler.console, "print", lambda *a, **k: None)

    def transcribe(audio, **kw):
        segment = SimpleNamespace(start=0.0, end=1.0, text=TRANSCRIPTS[len(audio)])
        return [segment], SimpleNamespace(language="en")

    handler = object.__new__(FasterWhisperSTTHandler)
    handler.gen_kwargs = {}
    handler.model = SimpleNamespace(transcribe=transcribe)
    return _prepare(handler)


def build_lightning_whisper_mlx(monkeypatch):
    _ensure_module("lightning_whisper_mlx", LightningWhisperMLX=object)
    from speech_to_speech.STT import lightning_whisper_mlx_handler
    from speech_to_speech.STT.lightning_whisper_mlx_handler import LightningWhisperSTTHandler

    monkeypatch.setattr(lightning_whisper_mlx_handler.console, "print", lambda *a, **k: None)
    # torch.mps.empty_cache() is unguarded and is a no-op only where Metal exists.
    monkeypatch.setattr(lightning_whisper_mlx_handler.torch.mps, "empty_cache", lambda: None)

    handler = object.__new__(LightningWhisperSTTHandler)
    handler.start_language = "en"
    handler.last_language = "en"
    handler.model = SimpleNamespace(transcribe=lambda audio, **kw: {"text": TRANSCRIPTS[len(audio)], "language": "en"})
    return _prepare(handler)


def build_mlx_audio_whisper(monkeypatch):
    from speech_to_speech.STT import mlx_audio_whisper_handler
    from speech_to_speech.STT.mlx_audio_whisper_handler import MLXAudioWhisperSTTHandler

    monkeypatch.setattr(mlx_audio_whisper_handler.console, "print", lambda *a, **k: None)

    handler = object.__new__(MLXAudioWhisperSTTHandler)
    handler.model_name = "mlx-community/whisper-large-v3-turbo"
    handler.start_language = "en"
    handler.last_language = "en"
    handler.gen_kwargs = {}
    handler.model = SimpleNamespace(
        generate=lambda audio, verbose=False, **kw: SimpleNamespace(text=TRANSCRIPTS[len(audio)])
    )
    return _prepare(handler)


BUILDERS = {
    "whisper": build_whisper,
    "faster-whisper": build_faster_whisper,
    "whisper-mlx": build_lightning_whisper_mlx,
    "mlx-audio-whisper": build_mlx_audio_whisper,
}


def vad_chunk(mode: str, seconds: int) -> VADAudio:
    return VADAudio(
        audio=np.zeros(SAMPLE_RATE * seconds, dtype=np.float32),
        mode=mode,
        turn_id="turn_1",
        turn_revision=0,
    )


def drive_turn(handler) -> list:
    """Feed one utterance through the handler using BaseHandler.run()'s hook order.

    The input queue is left empty between chunks, which is what the pipeline looks like
    while the user is still speaking: the final chunk does not exist yet, so the
    `progressive-before-final` guard in `should_process_input` cannot mask the defect.
    """
    emitted = []
    for mode, seconds, _ in CHUNKS:
        item = vad_chunk(mode, seconds)
        if not handler.should_process_input(item):
            continue
        for output in handler.process(item):
            if not handler.should_emit_output(output):
                continue
            handler.before_emit_output(output)
            emitted.append(output)
    return emitted


@pytest.mark.parametrize("backend", sorted(BUILDERS))
def test_progressive_audio_yields_partial_transcription(backend, monkeypatch):
    handler = BUILDERS[backend](monkeypatch)

    outputs = list(handler.process(vad_chunk("progressive", 1)))

    assert len(outputs) == 1
    assert isinstance(outputs[0], PartialTranscription)
    assert outputs[0].text == "one"
    assert outputs[0].turn_id == "turn_1"
    assert outputs[0].turn_revision == 0


@pytest.mark.parametrize("backend", sorted(BUILDERS))
def test_final_audio_yields_transcription(backend, monkeypatch):
    handler = BUILDERS[backend](monkeypatch)

    outputs = list(handler.process(vad_chunk("final", 3)))

    assert len(outputs) == 1
    assert isinstance(outputs[0], Transcription)
    assert outputs[0].text == FULL_UTTERANCE


@pytest.mark.parametrize("backend", sorted(BUILDERS))
def test_progressive_chunks_do_not_finalize_the_turn(backend, monkeypatch):
    """The regression: the whole utterance must survive, not just the first fragment."""
    handler = BUILDERS[backend](monkeypatch)

    emitted = drive_turn(handler)

    finals = [item for item in emitted if isinstance(item, Transcription)]
    partials = [item for item in emitted if isinstance(item, PartialTranscription)]

    assert [item.text for item in partials] == ["one", "one two"]
    assert len(finals) == 1
    assert finals[0].text == FULL_UTTERANCE


@pytest.mark.parametrize("backend", sorted(BUILDERS))
def test_absent_mode_still_yields_transcription(backend, monkeypatch):
    """Non-realtime pipelines send VADAudio with mode=None and must be unaffected."""
    handler = BUILDERS[backend](monkeypatch)

    item = VADAudio(audio=np.zeros(SAMPLE_RATE * 3, dtype=np.float32), turn_id="turn_1", turn_revision=0)
    outputs = list(handler.process(item))

    assert len(outputs) == 1
    assert isinstance(outputs[0], Transcription)
    assert outputs[0].text == FULL_UTTERANCE


def test_lightning_whisper_mlx_keeps_unsupported_language_transcription(monkeypatch):
    handler = build_lightning_whisper_mlx(monkeypatch)
    handler.start_language = "auto"
    calls = []

    def transcribe(audio, **kwargs):
        calls.append(kwargs)
        return {"text": "Privet, kak dela?", "language": "ru"}

    handler.model = SimpleNamespace(transcribe=transcribe)

    outputs = list(handler.process(vad_chunk("final", 3)))

    assert outputs[0].text == "Privet, kak dela?"
    assert outputs[0].language_code == "ru-auto"
    assert calls == [{}]
