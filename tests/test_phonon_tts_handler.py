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
from speech_to_speech.pipeline.console import conversation_output
from speech_to_speech.pipeline.events import ResponseFailedEvent
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, AudioOutput, EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.TTS.phonon_tts_handler import PhononTTSHandler, _TrailingSilenceTrimmer
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
    handler.voices = ["voice-a", "voice-b"]
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
        ({"model_name": "/models/test", "max_trailing_silence_ms": 39}, "must be at least 40"),
    ],
)
def test_invalid_setup_fails_before_importing_runtime(monkeypatch, kwargs, error):
    monkeypatch.setitem(sys.modules, "ptts", None)
    handler = PhononTTSHandler.__new__(PhononTTSHandler)
    with pytest.raises(ValueError, match=error):
        handler.setup(Event(), **kwargs)


def test_setup_offers_catalog_voices_in_checkpoint_languages(monkeypatch):
    model = SimpleNamespace(sample_rate=24000, voices=["default"], languages=["en"])
    catalog = [
        {"name": "Zoey", "language": "en"},
        {"name": "Lucie", "language": "fr"},
        {"name": "default", "language": "en"},
    ]
    monkeypatch.setitem(sys.modules, "ptts", SimpleNamespace(TTS=lambda **_kw: model, catalog_voices=lambda: catalog))
    handler = PhononTTSHandler.__new__(PhononTTSHandler)
    handler.setup(Event(), model_name="/models/test", voice="Zoey")
    assert handler.voices == ["default", "Zoey"]


def test_unavailable_catalog_keeps_checkpoint_voices(monkeypatch):
    def unavailable():
        raise OSError("offline")

    model = SimpleNamespace(sample_rate=24000, voices=["default"], languages=["en"])
    monkeypatch.setitem(sys.modules, "ptts", SimpleNamespace(TTS=lambda **_kw: model, catalog_voices=unavailable))
    handler = PhononTTSHandler.__new__(PhononTTSHandler)
    with pytest.raises(ValueError, match="Unknown Phonon voice 'Zoey'"):
        handler.setup(Event(), model_name="/models/test", voice="Zoey")


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


def test_first_audio_and_speech_e2e_are_recorded(monkeypatch):
    import speech_to_speech.TTS.phonon_tts_handler as phonon_module
    from speech_to_speech.pipeline.turn_latency import TurnLatencyStore

    clock = [100.0]
    monkeypatch.setattr(phonon_module, "perf_counter", lambda: clock[0])

    def samples():
        clock[0] = 100.05
        yield np.ones(4, dtype=np.float32)
        clock[0] = 100.50
        yield np.ones(4, dtype=np.float32)

    handler, _, _ = make_handler(samples(), sample_rate=16000, blocksize=4)
    store = handler.turn_latency_store = TurnLatencyStore()
    tracker = store.get_or_create_response("resp-1", turn_id="turn_1", turn_revision=0)
    tts_input = TTSInput(
        text="Hello", response_key="resp-1", turn_id="turn_1", turn_revision=0, speech_stopped_at_s=99.0
    )
    assert len(list(handler.process(tts_input))) == 2
    assert tracker.tts_ttfa_s == pytest.approx(0.05)
    assert tracker.e2e_s == pytest.approx(1.05)


def test_clipping_and_padding_without_resampling():
    handler, _, _ = make_handler([np.array([2, -2, 0.5], dtype=np.float32)], sample_rate=16000, blocksize=4)
    output = list(handler.process(TTSInput(text="Hello")))
    np.testing.assert_array_equal(output[0], [32767, -32768, 16384, 0])


@pytest.mark.parametrize("quiet", [False, True])
def test_backend_transcript_follows_conversation_display_policy(capsys, quiet):
    handler, _, requests = make_handler([np.ones(4, dtype=np.float32)], sample_rate=16000, blocksize=4)
    with conversation_output(quiet):
        output = list(handler.process(TTSInput(text="Hello")))
    assert len(output) == 1
    assert requests == [("Hello", None)]
    assert capsys.readouterr().out == ("" if quiet else "ASSISTANT: Hello\n")


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
        (None, "alloy", None, None),
        ("voice-a", "alloy", None, "voice-a"),
        (None, "voice-b", "alloy", None),
        ("voice-a", "alloy", "voice-b", "voice-b"),
        ("voice-a", "voice-b", "alloy", "voice-a"),
    ],
)
def test_voice_override_precedence(configured, session, response, expected):
    handler, _, requests = make_handler([np.zeros(4, dtype=np.float32)], sample_rate=16000, blocksize=4)
    handler.voice = configured
    runtime = RuntimeConfig(session=RealtimeSessionCreateRequest(type="realtime", audio={"output": {"voice": session}}))
    override = RealtimeResponseCreateParams(audio={"output": {"voice": response}}) if response else None
    output = list(handler.process(TTSInput(text="Hello", runtime_config=runtime, response=override)))
    assert len(output) == 1
    assert handler.queue_out.empty()
    assert requests == [("Hello", expected)]


@pytest.mark.parametrize(
    "voice", ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar"]
)
def test_realtime_voices_use_configured_voice(voice):
    handler, _, requests = make_handler([np.ones(4, dtype=np.float32)], sample_rate=16000, blocksize=4)
    handler.voice = "voice-a"
    override = RealtimeResponseCreateParams(audio={"output": {"voice": voice}})
    output = list(handler.process(TTSInput(text="Hello", response=override)))
    assert len(output) == 1 and np.any(output[0])
    assert handler.queue_out.empty()
    assert requests == [("Hello", "voice-a")]


def test_native_voice_named_like_realtime_voice_takes_precedence():
    handler, _, requests = make_handler([np.ones(4, dtype=np.float32)], sample_rate=16000, blocksize=4)
    handler.voice = "voice-a"
    handler.voices.append("alloy")
    override = RealtimeResponseCreateParams(audio={"output": {"voice": "alloy"}})
    assert len(list(handler.process(TTSInput(text="Hello", response=override)))) == 1
    assert requests == [("Hello", "alloy")]


@pytest.mark.parametrize("voice", ["missing", {"id": "voice_123"}])
@pytest.mark.parametrize("source", ["session", "response"])
def test_unknown_voice_is_rejected_before_generation(voice, source):
    handler, _, requests = make_handler([])
    runtime = RuntimeConfig()
    override = None
    if source == "session":
        runtime.apply_session_update(RealtimeSessionCreateRequest(type="realtime", audio={"output": {"voice": voice}}))
    else:
        override = RealtimeResponseCreateParams(audio={"output": {"voice": voice}})
    assert list(handler.process(TTSInput(text="Hello", runtime_config=runtime, response=override))) == []
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


@pytest.mark.parametrize("split", [False, True])
def test_opt_in_tail_trim_keeps_speech_and_a_short_pause(split):
    speech = np.full(1600, 0.1, dtype=np.float32)
    quiet = np.full(8000, 0.0001, dtype=np.float32)
    samples = np.concatenate((speech, quiet))
    chunks = [samples[:77], samples[77:1703], samples[1703:]] if split else [samples]
    handler, stream, _ = make_handler(chunks, sample_rate=16000, blocksize=160)
    handler.max_trailing_silence_ms = 120

    audio = np.concatenate(list(handler.process(TTSInput(text="A short sentence."))))

    assert audio.size == 1600 + 1920
    expected = np.clip(samples[: audio.size] * 32768, -32768, 32767).astype(np.int16)
    np.testing.assert_array_equal(audio, expected)
    assert stream.closed


def test_tail_trim_preserves_leading_audio_internal_pauses_and_quiet_speech():
    leading = np.zeros(3200, dtype=np.float32)
    speech = np.full(1600, 0.002, dtype=np.float32)
    internal = np.zeros(6400, dtype=np.float32)
    tail = np.zeros(8000, dtype=np.float32)
    samples = np.concatenate((leading, speech, internal, speech, tail))
    handler, _, _ = make_handler([samples], sample_rate=16000, blocksize=160)
    handler.max_trailing_silence_ms = 120

    audio = np.concatenate(list(handler.process(TTSInput(text="Quiet speech with an internal pause."))))

    expected = np.clip(samples[: len(samples) - len(tail) + 1920] * 32768, -32768, 32767).astype(np.int16)
    np.testing.assert_array_equal(audio, expected)


@pytest.mark.parametrize("sample_rate", [16000, 24000])
def test_tail_trim_is_independent_of_native_chunks_and_resampling(sample_rate):
    samples = np.concatenate(
        (np.sin(np.arange(sample_rate // 5) * 0.1).astype(np.float32) * 0.2, np.zeros(sample_rate // 2, np.float32))
    )
    outputs = []
    for chunks in ([samples], [samples[:1], samples[1:501], samples[501:-3], samples[-3:]]):
        handler, _, _ = make_handler(chunks, sample_rate=sample_rate)
        handler.max_trailing_silence_ms = 120
        outputs.append(np.concatenate(list(handler.process(TTSInput(text="A sentence.")))))
    np.testing.assert_array_equal(outputs[0], outputs[1])


def test_tail_trim_does_not_delay_the_first_voiced_block():
    visited = []

    def samples():
        visited.append("first chunk")
        yield np.ones(1024, dtype=np.float32)
        visited.append("second chunk")
        yield np.zeros(8000, dtype=np.float32)

    handler, stream, _ = make_handler(samples(), sample_rate=16000)
    handler.max_trailing_silence_ms = 120
    output = handler.process(TTSInput(text="A sentence."))

    assert next(output).size == 512
    assert visited == ["first chunk"]
    output.close()
    assert stream.closed


def test_tail_trim_cancellation_drops_held_quiet_audio_and_closes_stream():
    handler, stream, _ = make_handler(
        [np.ones(1600, np.float32), np.zeros(8000, np.float32)], sample_rate=16000, blocksize=160
    )
    handler.max_trailing_silence_ms = 120
    output = handler.process(TTSInput(text="A sentence."))
    next(output)
    handler.cancel_scope.cancel()

    assert list(output) == []
    assert stream.closed


def test_tail_trim_bounds_pending_memory_and_preserves_all_quiet_output():
    trimmer = _TrailingSilenceTrimmer(16000, 120)
    quiet = np.zeros(160000, dtype=np.float32)
    np.testing.assert_array_equal(np.concatenate(list(trimmer.push(quiet))), quiet)
    list(trimmer.push(np.ones(160, np.float32)))
    list(trimmer.push(quiet))
    assert trimmer.pending.size <= 16000
    list(trimmer.finish())
    assert trimmer.pending.size == 0
