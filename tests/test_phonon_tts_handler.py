from __future__ import annotations

import os
import sys
from queue import Queue
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import soxr
from openai.types.realtime import RealtimeSessionCreateRequest
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import ResponseFailedEvent
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, AudioOutput, EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.TTS.phonon_tts_handler import PhononTTSHandler
from tests.turns import reopen


class FakeStream:
    def __init__(self, samples, sample_rate=24000):
        self.samples = samples
        self.sample_rate = sample_rate
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True

    def __iter__(self):
        return iter(self.samples)


def make_handler(samples, *, sample_rate=24000, blocksize=512):
    stream = FakeStream(samples, sample_rate)
    requests = []

    def generate(text, *, voice):
        requests.append((text, voice))
        return stream

    handler = PhononTTSHandler.__new__(PhononTTSHandler)
    handler.stop_event = Event()
    handler.queue_out = Queue()
    handler._failed_responses = set()
    handler.cancel_scope = CancelScope()
    handler.speculative_turns = None
    handler.voice = None
    handler.language = "en"
    handler.blocksize = blocksize
    handler.model = SimpleNamespace(stream=generate, voices=["voice-a", "voice-b"])
    return handler, stream, requests


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({}, "requires a checkpoint"),
        ({"model_name": " "}, "requires a checkpoint"),
        ({"model_name": "/models/test", "blocksize": 0}, "blocksize must be positive"),
        ({"model_name": "/models/test", "num_threads": 0}, "num_threads must be positive"),
        ({"model_name": "/models/test", "num_threads": -1}, "num_threads must be positive"),
        ({"model_name": "/models/test", "language": "xx"}, "Unsupported Phonon normalization language"),
    ],
)
def test_invalid_setup_fails_before_importing_runtime(monkeypatch, kwargs, error):
    monkeypatch.setitem(sys.modules, "ptts", None)
    handler = PhononTTSHandler.__new__(PhononTTSHandler)
    with pytest.raises(ValueError, match=error):
        handler.setup(Event(), **kwargs)


@pytest.mark.parametrize("num_threads", [None, 4])
def test_setup_loads_once_and_forwards_checkpoint_options(monkeypatch, num_threads):
    loaded = []
    calls = []
    stream = FakeStream([np.zeros(1000, dtype=np.float32)])
    model = SimpleNamespace(sample_rate=24000, voices=["voice-a"], stream=lambda *a, **kw: stream)

    def load(**kwargs):
        calls.append("load")
        loaded.append(kwargs)
        return model

    monkeypatch.setitem(
        sys.modules, "ptts", SimpleNamespace(TTS=load, set_num_threads=lambda count: calls.append(("threads", count)))
    )
    handler = PhononTTSHandler(
        Event(),
        Queue(),
        Queue(),
        setup_args=(Event(),),
        setup_kwargs={
            "model_name": "/models/test",
            "revision": "revision",
            "language": " FR ",
            "voice": "voice-a",
            "num_threads": num_threads,
        },
    )
    list(handler.process(TTSInput(text="Bonjour")))
    list(handler.process(TTSInput(text="Salut")))
    assert loaded == [{"config": "/models/test", "revision": "revision", "lang": "fr", "quant": "q8", "device": "cpu"}]
    assert calls == (["load"] if num_threads is None else [("threads", num_threads), "load"])


def test_resampling_is_independent_of_native_chunk_boundaries():
    samples = (0.4 * np.sin(np.arange(4813) * 0.071)).astype(np.float32)
    chunks = [samples[:1], samples[1:714], samples[714:1257], samples[1257:]]
    handler, stream, _ = make_handler(chunks)
    output = list(handler.process(TTSInput(text="Hello")))
    expected = np.clip(soxr.resample(samples, 24000, 16000, quality="HQ") * 32768, -32768, 32767).astype(np.int16)
    actual = np.concatenate(output)
    assert all(block.shape == (512,) and block.dtype == np.int16 for block in output)
    np.testing.assert_array_equal(actual[: expected.size], expected)
    np.testing.assert_array_equal(actual[expected.size :], 0)
    assert actual.size - expected.size < 512
    assert stream.closed


def test_clipping_and_padding_without_resampling():
    handler, _, _ = make_handler([np.array([2, -2, 0.5], dtype=np.float32)], sample_rate=16000, blocksize=4)
    output = list(handler.process(TTSInput(text="Hello")))
    np.testing.assert_array_equal(output[0], [32767, -32768, 16384, 0])


@pytest.mark.parametrize("cancel", ["generation", "stop"])
def test_interrupt_drops_buffered_audio_and_closes_native_stream(cancel):
    handler, stream, _ = make_handler([np.ones(10, dtype=np.float32)], sample_rate=16000, blocksize=4)
    tracker = SpeculativeTurnTracker()
    turn_id, revision = tracker.start_turn()
    handler.speculative_turns = tracker
    output = handler.process(TTSInput(text="Hello", turn_id=turn_id, turn_revision=revision))
    next(output)
    if cancel == "generation":
        handler.cancel_scope.cancel()
    else:
        handler.stop_event.set()
    assert list(output) == []
    assert stream.closed


def test_generator_close_closes_native_stream():
    handler, stream, _ = make_handler([np.ones(10, dtype=np.float32)], sample_rate=16000, blocksize=4)
    output = handler.process(TTSInput(text="Hello"))
    next(output)
    output.close()
    assert stream.closed


def test_stale_generation_and_empty_text_do_not_start_native_stream():
    handler, _, requests = make_handler([])
    handler.cancel_scope.cancel()
    assert list(handler.process(TTSInput(text="Hello", cancel_generation=0))) == []
    assert list(handler.process(TTSInput(text=" "))) == []
    assert requests == []


def test_stale_speculative_input_does_not_start_native_stream():
    handler, _, requests = make_handler([])
    tracker = SpeculativeTurnTracker()
    turn_id, revision = tracker.start_turn()
    reopen(tracker, turn_id, revision)
    handler.speculative_turns = tracker
    assert list(handler.process(TTSInput(text="Hello", turn_id=turn_id, turn_revision=revision))) == []
    assert requests == []


def test_native_failure_closes_stream_and_marks_response_failed():
    def samples():
        yield np.ones(4, dtype=np.float32)
        raise RuntimeError("native generation failed")

    handler, stream, _ = make_handler(samples(), sample_rate=16000, blocksize=4)
    input = TTSInput(text="Hello", response_key="response", cancel_generation=0)
    list(handler.process(input))
    assert stream.closed
    failed = handler.queue_out.get_nowait()
    assert isinstance(failed, ResponseFailedEvent)
    assert failed.response_key == "response" and failed.cancel_generation == 0
    assert list(handler.process(input)) == []
    assert list(handler.process(EndOfResponse(response_key="response", cancel_generation=0))) == [AUDIO_RESPONSE_DONE]
    assert handler._failed_responses == set()


@pytest.mark.parametrize(
    ("configured", "session", "response", "expected"),
    [
        (None, None, None, None),
        ("voice-a", None, None, "voice-a"),
        ("voice-a", "voice-b", None, "voice-b"),
        ("voice-a", "voice-b", "voice-a", "voice-a"),
    ],
)
def test_voice_override_precedence(configured, session, response, expected):
    handler, _, requests = make_handler([np.zeros(4, dtype=np.float32)], sample_rate=16000, blocksize=4)
    handler.voice = configured
    runtime = RuntimeConfig(session=RealtimeSessionCreateRequest(type="realtime", audio={"output": {"voice": session}}))
    override = RealtimeResponseCreateParams(audio={"output": {"voice": response}}) if response else None
    list(handler.process(TTSInput(text="Hello", runtime_config=runtime, response=override)))
    assert requests == [("Hello", expected)]


def test_unknown_voice_is_rejected_before_generation():
    handler, _, requests = make_handler([])
    override = RealtimeResponseCreateParams(audio={"output": {"voice": "missing"}})
    assert list(handler.process(TTSInput(text="Hello", response=override))) == []
    assert isinstance(handler.queue_out.get_nowait(), ResponseFailedEvent)
    assert requests == []


def test_empty_native_stream_reports_failure():
    handler, stream, _ = make_handler([])
    assert list(handler.process(TTSInput(text="Hello"))) == []
    assert stream.closed
    assert isinstance(handler.queue_out.get_nowait(), ResponseFailedEvent)


def test_session_reset_clears_previous_failure():
    handler, _, _ = make_handler([])
    list(handler.process(TTSInput(text="Hello")))
    assert handler._failed_responses
    handler.on_session_end()
    assert handler._failed_responses == set()


def test_stale_keyed_terminal_keeps_response_cleanup():
    handler, _, _ = make_handler([])
    tracker = SpeculativeTurnTracker()
    turn_id, revision = tracker.start_turn()
    handler.speculative_turns = tracker
    terminal = EndOfResponse(response_key="response", turn_id=turn_id, turn_revision=revision, cancel_generation=7)
    reopen(tracker, turn_id, revision)
    outputs = list(handler.process(terminal))
    queued = handler.output_for_queue(outputs[0], terminal)
    assert outputs == [AUDIO_RESPONSE_DONE]
    assert isinstance(queued, AudioOutput)
    assert queued.cleanup_only and queued.response_key == "response" and queued.cancel_generation == 7


@pytest.mark.skipif(
    not os.environ.get("PHONON_TEST_CHECKPOINT"), reason="Set PHONON_TEST_CHECKPOINT to a local checkpoint"
)
def test_supplied_checkpoint_generates_pipeline_audio():
    handler = PhononTTSHandler(
        Event(),
        Queue(),
        Queue(),
        setup_args=(Event(),),
        setup_kwargs={"model_name": os.environ["PHONON_TEST_CHECKPOINT"], "num_threads": 4},
    )
    output = list(handler.process(TTSInput(text="Hello. This is a local speech test.")))
    assert output and all(block.shape == (512,) and block.dtype == np.int16 for block in output)
    assert np.max(np.abs(np.concatenate(output).astype(np.int32))) > 0
