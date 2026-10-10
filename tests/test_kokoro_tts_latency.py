from __future__ import annotations

import sys
from collections.abc import Iterator
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.pipeline.turn_latency import TurnLatencyStore
from speech_to_speech.TTS import kokoro_handler
from speech_to_speech.TTS.kokoro_handler import KokoroTTSHandler


def _native_handler(monkeypatch: pytest.MonkeyPatch, *, stream) -> KokoroTTSHandler:
    class FakePipeline:
        def __init__(self, *, lang_code: str, device: str) -> None:
            self.lang_code = lang_code

        def __call__(self, _text: str, *, voice: str, speed: float) -> Iterator[tuple[str, str, np.ndarray]]:
            return stream()

    monkeypatch.setitem(sys.modules, "kokoro", SimpleNamespace(KPipeline=FakePipeline))
    monkeypatch.setattr(kokoro_handler, "platform", "linux")
    monkeypatch.setattr(KokoroTTSHandler, "warmup", lambda self: None)
    monkeypatch.setattr(kokoro_handler.console, "print", lambda *_args, **_kwargs: None)
    handler = KokoroTTSHandler.__new__(KokoroTTSHandler)
    handler.setup(Event(), device="cpu", lang_code="b", voice="bm_fable", blocksize=4)
    return handler


def test_kokoro_records_ttfa_at_first_provider_chunk_and_preserves_first_audio(monkeypatch):
    """TTFA is provider audio before resample; later segments must not overwrite."""
    clock = [100.0]
    monkeypatch.setattr(kokoro_handler, "perf_counter", lambda: clock[0])

    def stream():
        clock[0] = 100.05
        yield ("Hello", "həˈloʊ", np.full(8, 0.1, dtype=np.float32))
        clock[0] = 100.50
        yield ("again", "əˈɡɛn", np.full(8, 0.2, dtype=np.float32))

    handler = _native_handler(monkeypatch, stream=stream)
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_response("resp-1", turn_id="turn_1", turn_revision=0)

    outputs = list(
        handler.process(
            TTSInput(
                text="Hello",
                response_key="resp-1",
                turn_id="turn_1",
                turn_revision=0,
                speech_stopped_at_s=99.0,
            )
        )
    )

    assert outputs
    assert tracker.tts_ttfa_s == pytest.approx(0.05)
    assert tracker.e2e_s == pytest.approx(1.05)

    clock[0] = 200.0

    def later_stream():
        clock[0] = 200.40
        yield ("Again", "əˈɡɛn", np.full(8, 0.3, dtype=np.float32))

    handler.pipeline = lambda *_args, **_kwargs: later_stream()
    list(
        handler.process(
            TTSInput(
                text="Again",
                response_key="resp-1",
                turn_id="turn_1",
                turn_revision=0,
                speech_stopped_at_s=99.0,
            )
        )
    )

    assert tracker.tts_ttfa_s == pytest.approx(0.05)
    assert tracker.e2e_s == pytest.approx(1.05)


def test_kokoro_mlx_records_ttfa_before_silence_trim(monkeypatch):
    """MLX TTFA uses provider audio arrival, not post-trim yield time."""
    clock = [50.0]
    monkeypatch.setattr(kokoro_handler, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(kokoro_handler, "platform", "darwin")
    monkeypatch.setattr(KokoroTTSHandler, "warmup", lambda self: None)
    monkeypatch.setattr(kokoro_handler.console, "print", lambda *_args, **_kwargs: None)

    class FakePipeline:
        def load_voice(self, voice: str) -> None:
            return None

        def __call__(self, *, text: str, voice: str, speed: float):
            clock[0] = 50.08
            # Leading silence then speech: trim would delay the first yielded sample.
            audio = np.concatenate([np.zeros(2400, dtype=np.float32), np.full(2400, 0.2, dtype=np.float32)])
            yield SimpleNamespace(audio=audio.reshape(1, -1))
            clock[0] = 50.40

    handler = KokoroTTSHandler.__new__(KokoroTTSHandler)
    handler.should_listen = Event()
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.device = "mps"
    handler.backend = "mlx"
    handler.voice = "bm_fable"
    handler.lang_code = "b"
    handler._initial_voice = "bm_fable"
    handler._initial_lang_code = "b"
    handler.speed = 1.0
    handler.blocksize = 512
    handler.model = SimpleNamespace(_get_pipeline=lambda lang: FakePipeline())
    handler._pipeline = FakePipeline()
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_response("mlx-1", turn_id="turn_1")

    outputs = list(
        handler.process(
            TTSInput(
                text="Hello",
                response_key="mlx-1",
                turn_id="turn_1",
                speech_stopped_at_s=49.0,
            )
        )
    )

    assert outputs
    assert tracker.tts_ttfa_s == pytest.approx(0.08)
    assert tracker.e2e_s == pytest.approx(1.08)
