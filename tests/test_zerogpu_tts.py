"""One parent proxy supports different providers without importing either model."""

from queue import Queue
from threading import Event, Thread
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pytest
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import ResponseFailedEvent
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.pipeline.turn_latency import TurnLatencyTracker


@pytest.fixture(params=[("qwen3", "Aiden", "Vivian", "Serena"), ("kokoro", "af_heart", "af_bella", "am_adam")])
def adapter(monkeypatch, request):
    from speech_to_speech.TTS import zerogpu_handler as module

    backend, initial, alternate, response_voice = request.param
    voices = [initial, alternate, response_voice]
    config = {
        "backend": backend,
        "model": f"test/{backend}",
        "voice": initial,
        "language": "en",
        "supported_languages": ["en", "fr"],
    }
    worker = SimpleNamespace(
        get_tts_config=lambda: config,
        resolve_tts_voice=lambda requested, current: next(
            (v for v in voices if v.lower() == str(requested).lower()), current
        ),
        synthesize_chunks=lambda *args, **kwargs: iter([np.ones(512, dtype=np.int16)]),
    )
    monkeypatch.setattr(module, "runtime", worker)
    scope = CancelScope()
    handler = module.ZeroGPUTTSHandler(
        Event(),
        Queue(),
        Queue(),
        setup_kwargs={
            "should_listen": Event(),
            "cancel_scope": scope,
        },
    )
    yield handler, worker, scope, voices
    handler.cleanup()


def test_streams_before_completion_and_records_timing(adapter):
    handler, worker, _, voices = adapter
    first, second = np.arange(512, dtype=np.int16), np.full(512, -1234, dtype=np.int16)
    release = Event()

    def synthesize(text, language, voice, **kwargs):
        assert (text, language, voice) == ("Hello.", "en", voices[0])
        yield first
        assert release.wait(2)
        yield second

    worker.synthesize_chunks = synthesize
    tracker = TurnLatencyTracker()
    handler.turn_latency_store = SimpleNamespace(get_response=lambda key: tracker)
    output = handler.process(
        TTSInput(text="Hello.", language_code="en", response_key="response", speech_stopped_at_s=perf_counter() - 0.1)
    )
    try:
        assert next(output) is first
        assert tracker.tts_ttfa_s is not None and tracker.e2e_s >= 0.1
    finally:
        release.set()
    assert list(output)[0] is second


def test_interrupt_before_first_audio_returns_without_waiting_for_native_inference(adapter):
    handler, worker, scope, _ = adapter
    entered, release, closed, completed = Event(), Event(), Event(), Event()
    output = []

    def synthesize(*args, **kwargs):
        try:
            entered.set()
            while not release.wait(0.005):
                if kwargs["cancelled"]():
                    return
            yield np.ones(512, dtype=np.int16)
        finally:
            closed.set()

    worker.synthesize_chunks = synthesize

    def consume():
        output.extend(handler.process(TTSInput(text="Slow first audio.")))
        completed.set()

    thread = Thread(target=consume)
    thread.start()
    try:
        assert entered.wait(1)
        scope.cancel()
        assert completed.wait(0.5)
        assert output == []
        assert closed.is_set(), "Cancellation closes the host stream on its own thread"
    finally:
        release.set()
        thread.join(2)
    assert closed.wait(1)
    assert handler.queue_out.empty()


def test_interrupt_discards_late_audio_and_closes_source(adapter):
    handler, worker, scope, _ = adapter
    closed = Event()

    def synthesize(*args, **kwargs):
        try:
            while True:
                yield np.ones(512, dtype=np.int16)
        finally:
            closed.set()

    worker.synthesize_chunks = synthesize
    output = handler.process(TTSInput(text="Continue."))
    assert next(output).dtype == np.int16
    scope.cancel()
    assert list(output) == []
    assert closed.wait(1)


def test_voice_precedence_validation_and_session_reset(adapter):
    handler, worker, _, voices = adapter
    seen = []
    worker.synthesize_chunks = lambda text, language, voice, **kwargs: seen.append(voice) or iter([b"pcm"])
    config = RuntimeConfig()
    config.session.audio.output.voice = voices[1].lower()
    assert list(handler.process(TTSInput(text="Hello.", runtime_config=config)))
    assert seen[-1] == voices[1]
    response = RealtimeResponseCreateParams.model_validate({"audio": {"output": {"voice": voices[2]}}})
    assert list(handler.process(TTSInput(text="Response.", runtime_config=config, response=response)))
    assert seen[-1] == voices[2]
    config.session.audio.output.voice = "not-a-provider-voice"
    assert list(handler.process(TTSInput(text="Hello.", runtime_config=config)))
    assert seen[-1] == voices[2]
    handler.on_session_end()
    assert handler.voice == voices[0]


def test_language_detection_falls_back_to_selected_supported_language(adapter):
    handler, worker, _, _ = adapter
    seen = []
    worker.synthesize_chunks = lambda text, language, voice, **kwargs: seen.append(language) or iter([b"pcm"])
    for detected, selected, expected in [("fr", "en", "fr"), ("xx", "fr", "fr"), ("xx", "auto", "en")]:
        assert list(handler.process(TTSInput(text="Hello", tts_language_code=detected, selected_language=selected)))
        assert seen[-1] == expected


@pytest.mark.parametrize("empty", [False, True])
def test_provider_failure_reported_once_and_remaining_response_suppressed(adapter, empty):
    handler, worker, scope, _ = adapter
    calls = []

    def synthesize(*args, **kwargs):
        calls.append(1)
        if not empty:
            raise RuntimeError("Provider crashed")
        return iter([])

    worker.synthesize_chunks = synthesize
    item = TTSInput(text="Hello.", response_key="response", cancel_generation=scope.generation)
    assert list(handler.process(item)) == []
    assert isinstance(handler.queue_out.get_nowait(), ResponseFailedEvent)
    assert list(handler.process(item)) == []
    assert calls == [1]
    assert handler.queue_out.empty()
    assert list(handler.process(EndOfResponse(response_key="response", cancel_generation=scope.generation))) == [
        AUDIO_RESPONSE_DONE
    ]
    assert list(handler.process(item)) == []
    assert calls == [1, 1]


def test_stale_speculation_drops_input_but_preserves_response_cleanup(adapter):
    handler, worker, _, _ = adapter
    handler.speculative_turns = SimpleNamespace(
        is_latest=lambda *args, **kwargs: True, wait_for_gate=lambda *args, **kwargs: False
    )
    worker.synthesize_chunks = lambda *args, **kwargs: pytest.fail("Stale request must not reach the worker")
    assert list(handler.process(TTSInput(text="Stale"))) == []
    end = EndOfResponse(response_key="stale-response")
    assert list(handler.process(end)) == [AUDIO_RESPONSE_DONE]
    assert end.cleanup_only
    assert list(handler.process(EndOfResponse())) == []


def test_closing_parent_iterator_closes_worker_stream(adapter):
    handler, worker, _, _ = adapter
    closed = Event()

    def synthesize(*args, **kwargs):
        try:
            while True:
                yield b"pcm"
        finally:
            closed.set()

    worker.synthesize_chunks = synthesize
    output = handler.process(TTSInput(text="Hello"))
    assert next(output) == b"pcm"
    output.close()
    assert closed.wait(1)


def test_session_end_cancels_pending_stream_without_reporting_failure(adapter):
    handler, worker, _, voices = adapter
    entered, release, completed, closed = Event(), Event(), Event(), Event()
    output = []

    def synthesize(*args, **kwargs):
        try:
            entered.set()
            while not release.wait(0.005):
                if kwargs["cancelled"]():
                    return
            yield b"late"
        finally:
            closed.set()

    worker.synthesize_chunks = synthesize

    def consume():
        output.extend(handler.process(TTSInput(text="Pending")))
        completed.set()

    thread = Thread(target=consume)
    thread.start()
    try:
        assert entered.wait(1)
        handler.voice = voices[1]
        handler.on_session_end()
        assert completed.wait(0.5)
        assert output == []
        assert handler.queue_out.empty()
        assert handler.voice == voices[0]
    finally:
        release.set()
        thread.join(2)
    assert closed.wait(1)
