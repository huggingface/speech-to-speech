"""Exercise native Qwen process semantics without loading a model or GPU."""

from queue import Queue
from threading import Event
from time import perf_counter
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.pipeline.turn_latency import TurnLatencyTracker
from speech_to_speech.TTS import qwen3_tts_handler


@pytest.fixture
def adapter(monkeypatch):
    from speech_to_speech.TTS import zerogpu_handler as module

    worker = ModuleType("fake_worker")
    worker.get_tts_speakers = lambda: ["Aiden", "Vivian"]
    worker.get_model_ids = lambda: {"tts": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"}
    monkeypatch.setattr(module, "runtime", worker)
    monkeypatch.setattr(qwen3_tts_handler, "platform", "linux")
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
    return handler, worker, scope


def test_native_process_yields_first_chunk_before_completion_and_records_timing(adapter):
    handler, worker, _ = adapter
    first = np.arange(512, dtype=np.int16)
    second = np.full(512, -1234, dtype=np.int16)
    reached_second = []

    def synthesize(text, language, speaker):
        assert (text, language, speaker) == ("Hello.", "english", "Aiden")
        yield first
        reached_second.append(True)
        yield second

    worker.synthesize_chunks = synthesize
    tracker = TurnLatencyTracker()
    handler.turn_latency_store = SimpleNamespace(get_response=lambda key: tracker)
    output = handler.process(
        TTSInput(text="Hello.", language_code="en", response_key="response", speech_stopped_at_s=perf_counter() - 0.1)
    )
    assert next(output) is first
    assert not reached_second, "The adapter must not buffer a complete sentence"
    assert tracker.tts_ttfa_s is not None and tracker.e2e_s >= 0.1
    assert list(output)[0] is second
    assert first.dtype == second.dtype == np.int16


def test_native_process_discards_chunks_after_interrupt_and_closes_source(adapter):
    handler, worker, scope = adapter
    closed = []

    def synthesize(*args):
        try:
            yield np.ones(512, dtype=np.int16)
            yield np.full(512, 2, dtype=np.int16)
        finally:
            closed.append(True)

    worker.synthesize_chunks = synthesize
    output = handler.process(TTSInput(text="Continue.", language_code="en"))
    assert next(output).dtype == np.int16
    scope.cancel()
    assert list(output) == []
    assert closed == [True]


def test_native_session_voice_validation_and_reset_use_checkpoint_metadata(adapter):
    handler, worker, _ = adapter
    seen = []

    def synthesize(text, language, speaker):
        seen.append(speaker)
        yield np.ones(512, dtype=np.int16)

    worker.synthesize_chunks = synthesize
    runtime = RuntimeConfig()
    runtime.session.audio.output.voice = "vivian"
    assert list(handler.process(TTSInput(text="Hello.", runtime_config=runtime)))
    assert seen[-1] == "Vivian"
    runtime.session.audio.output.voice = "not-a-checkpoint-speaker"
    assert list(handler.process(TTSInput(text="Hello.", runtime_config=runtime)))
    assert seen[-1] == "Vivian"
    handler.on_session_end()
    assert handler.speaker == "Aiden"
