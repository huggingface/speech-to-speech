from __future__ import annotations

import logging
import sys
from queue import Queue
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from openai.types.realtime import RealtimeSessionCreateRequest
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, PIPELINE_END, AudioOutput, EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.pipeline.transcript_logging import set_log_transcripts
from speech_to_speech.TTS import kitten_tts_handler as kitten_module
from speech_to_speech.TTS.kitten_tts_handler import KittenTTSHandler
from tests.turns import reopen


class FakeKitten:
    available_voices = ["expr-voice-2-f", "expr-voice-3-m"]
    voice_aliases = {"Bella": "expr-voice-2-f", "Bruno": "expr-voice-3-m"}

    def __init__(self):
        self.calls = []

    def generate(self, *, text, voice):
        self.calls.append((text, voice))
        return np.zeros((1, 12), dtype=np.float32)


def make_handler(*, cancel_scope=None, speculative_turns=None, blocksize=4):
    handler = KittenTTSHandler.__new__(KittenTTSHandler)
    handler.model = FakeKitten()
    handler.voice = "Bruno"
    handler.blocksize = blocksize
    handler.cancel_scope = cancel_scope
    handler.speculative_turns = speculative_turns
    return handler


def install_fake_runtime(monkeypatch, model):
    loads = []

    def load(model_name, *, backend):
        loads.append((model_name, backend))
        return model

    monkeypatch.setitem(sys.modules, "kittenml.kittentts_legacy", SimpleNamespace(download_from_huggingface=load))
    return loads


def test_setup_uses_supported_cpu_loader_and_warms_up(monkeypatch):
    model = FakeKitten()
    loads = install_fake_runtime(monkeypatch, model)
    handler = KittenTTSHandler.__new__(KittenTTSHandler)

    handler.setup(Event(), voice="Bella")

    assert loads == [("KittenML/kitten-tts-mini-0.8", "cpu")]
    assert model.calls == [("Hello", "Bella")]


@pytest.mark.parametrize(
    ("kwargs", "message"), [({"device": "cuda"}, "only device='cpu'"), ({"blocksize": 0}, "positive")]
)
def test_invalid_configuration_fails_before_model_loading(kwargs, message):
    with pytest.raises(ValueError, match=message):
        KittenTTSHandler.__new__(KittenTTSHandler).setup(Event(), **kwargs)


def test_invalid_startup_voice_fails_before_warmup(monkeypatch):
    model = FakeKitten()
    install_fake_runtime(monkeypatch, model)

    with pytest.raises(ValueError, match="Unsupported KittenTTS voice"):
        KittenTTSHandler.__new__(KittenTTSHandler).setup(Event(), voice="unknown")
    assert model.calls == []


def test_failed_warmup_does_not_report_success(monkeypatch, caplog):
    model = FakeKitten()

    def fail(**_kwargs):
        raise RuntimeError("model incompatible")

    model.generate = fail
    install_fake_runtime(monkeypatch, model)
    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError, match="model incompatible"):
        KittenTTSHandler.__new__(KittenTTSHandler).setup(Event())
    assert "warmed up" not in caplog.text


def test_current_turn_synthesizes_and_commits_using_current_tracker_api():
    tracker = SpeculativeTurnTracker()
    turn_id, revision = tracker.start_turn()
    handler = make_handler(speculative_turns=tracker)

    chunks = list(handler.process(TTSInput(text="Hello", turn_id=turn_id, turn_revision=revision)))

    assert chunks
    assert handler.model.calls == [("Hello", "Bruno")]
    assert tracker.begin_reopen_candidate(turn_id, revision) is None
    assert list(handler.process(EndOfResponse(turn_id=turn_id, turn_revision=revision))) == [AUDIO_RESPONSE_DONE]


def test_stale_text_does_not_synthesize():
    tracker = SpeculativeTurnTracker()
    turn_id, revision = tracker.start_turn()
    reopen(tracker)
    handler = make_handler(speculative_turns=tracker)

    assert list(handler.process(TTSInput(text="Old", turn_id=turn_id, turn_revision=revision))) == []
    assert handler.model.calls == []


def test_stale_keyed_terminal_preserves_cleanup_identity():
    tracker = SpeculativeTurnTracker()
    turn_id, revision = tracker.start_turn()
    reopen(tracker)
    handler = make_handler(speculative_turns=tracker)
    terminal = EndOfResponse(response_key="response_1", turn_id=turn_id, turn_revision=revision, cancel_generation=7)

    outputs = list(handler.process(terminal))
    assert outputs == [AUDIO_RESPONSE_DONE]
    queued = handler.output_for_queue(outputs[0], terminal)
    assert isinstance(queued, AudioOutput)
    assert queued.audio == AUDIO_RESPONSE_DONE
    assert queued.response_key == "response_1"
    assert queued.cancel_generation == 7
    assert queued.cleanup_only is True


def test_stale_unkeyed_terminal_is_dropped():
    tracker = SpeculativeTurnTracker()
    turn_id, revision = tracker.start_turn()
    reopen(tracker)
    handler = make_handler(speculative_turns=tracker)

    assert list(handler.process(EndOfResponse(turn_id=turn_id, turn_revision=revision))) == []


@pytest.mark.parametrize(
    ("session_voice", "response_voice", "expected"),
    [
        (None, None, "Bruno"),
        ("Bella", None, "Bella"),
        ("Bruno", "Bella", "Bella"),
        (None, "expr-voice-2-f", "expr-voice-2-f"),
        ("Bella", "unknown", "Bruno"),
    ],
)
def test_voice_precedence_and_invalid_override_fallback(session_voice, response_voice, expected):
    handler = make_handler()
    config = RuntimeConfig(
        session=RealtimeSessionCreateRequest(type="realtime", audio={"output": {"voice": session_voice}})
    )
    response = RealtimeResponseCreateParams(audio={"output": {"voice": response_voice}})

    list(handler.process(TTSInput(text="Hello", runtime_config=config, response=response)))

    assert handler.model.calls == [("Hello", expected)]


def test_response_and_session_voices_do_not_leak_to_later_inputs():
    handler = make_handler()
    config = RuntimeConfig(session=RealtimeSessionCreateRequest(type="realtime", audio={"output": {"voice": "Bella"}}))
    response = RealtimeResponseCreateParams(audio={"output": {"voice": "Bruno"}})
    list(handler.process(TTSInput(text="Response override", runtime_config=config, response=response)))
    list(handler.process(TTSInput(text="Session default", runtime_config=config)))
    handler.on_session_end()
    list(handler.process(TTSInput(text="Next session", runtime_config=RuntimeConfig())))

    assert [voice for _, voice in handler.model.calls] == ["Bruno", "Bella", "Bruno"]


def test_resamples_24khz_to_pipeline_rate_and_pads_blocks():
    handler = make_handler(blocksize=3)
    chunks = list(handler.process(TTSInput(text="Hello")))

    # 12 native samples become 8 at 16kHz, then one padded delivery block.
    assert [len(chunk) for chunk in chunks] == [3, 3, 3]
    assert all(chunk.dtype == np.int16 for chunk in chunks)
    assert chunks[-1][-1] == 0


def test_clips_pcm_instead_of_wrapping(monkeypatch):
    handler = make_handler()
    monkeypatch.setattr(
        kitten_module, "resample_poly", lambda *_args, **_kwargs: np.array([2, -2, 0.5], dtype=np.float32)
    )

    chunks = list(handler.process(TTSInput(text="Hello")))

    np.testing.assert_array_equal(chunks[0], np.array([32767, -32768, 16384, 0], dtype=np.int16))


def test_cancelled_blocking_synthesis_emits_no_audio():
    scope = CancelScope()
    handler = make_handler(cancel_scope=scope)

    def generate(**_kwargs):
        scope.cancel()
        return np.zeros(12, dtype=np.float32)

    handler.model.generate = generate
    assert list(handler.process(TTSInput(text="Hello"))) == []


@pytest.mark.parametrize("audio_length", [9, 12])
def test_cancellation_stops_full_blocks_and_padded_tail(monkeypatch, audio_length):
    scope = CancelScope()
    handler = make_handler(cancel_scope=scope)
    monkeypatch.setattr(kitten_module, "resample_poly", lambda *_args, **_kwargs: np.zeros(audio_length))
    chunks = handler.process(TTSInput(text="Hello"))
    next(chunks)
    next(chunks)
    scope.cancel()

    with pytest.raises(StopIteration):
        next(chunks)


@pytest.mark.parametrize("enabled", [False, True])
def test_transcript_logging_requires_opt_in(caplog, enabled):
    handler = make_handler()
    secret = "private conversation content"
    set_log_transcripts(enabled)
    try:
        with caplog.at_level(logging.DEBUG):
            list(handler.process(TTSInput(text=secret)))
        assert (secret in caplog.text) is enabled
        if not enabled:
            assert f"chars={len(secret)}" in caplog.text
    finally:
        set_log_transcripts(False)


@pytest.mark.parametrize("enabled", [False, True])
def test_generation_errors_use_base_handler_privacy_gate(caplog, enabled):
    handler = make_handler()
    secret = "private model error transcript"

    def fail(**_kwargs):
        raise RuntimeError(secret)

    handler.model.generate = fail
    handler.stop_event = Event()
    handler.pipeline_index = None
    handler.queue_in = Queue()
    handler.queue_out = Queue()
    handler._times = []
    handler.queue_in.put(TTSInput(text="Hello"))
    handler.queue_in.put(PIPELINE_END)
    set_log_transcripts(enabled)
    try:
        with caplog.at_level(logging.ERROR):
            handler.run()
        assert "RuntimeError" in caplog.text
        assert (secret in caplog.text) is enabled
    finally:
        set_log_transcripts(False)


def test_supported_runtime_splits_long_text_and_selects_length_dependent_styles(monkeypatch, tmp_path):
    """Exercise the published runtime when the optional Kitten extra is installed."""
    runtime = pytest.importorskip("kittenml.kittentts_legacy.onnx_model")
    loader = pytest.importorskip("kittenml.kittentts_legacy.model")
    voices_path = tmp_path / "voices.npz"
    table = np.repeat(np.arange(400, dtype=np.float32)[:, None], 256, axis=1)
    np.savez(voices_path, **{"expr-voice-3-m": table})
    config = {
        "type": "ONNX2",
        "model_file": "mini.onnx",
        "voices": "voices.npz",
        "voice_aliases": {"Bruno": "expr-voice-3-m"},
    }
    monkeypatch.setattr(loader, "resolve_repo", lambda *_args: (str(tmp_path), None, config))
    calls = []

    class Session:
        def __init__(self, _path, *, providers):
            assert providers == ["CPUExecutionProvider"]

        def run(self, _names, feeds):
            # The original unbounded inference fails beyond the model's context.
            assert feeds["input_ids"].shape[-1] <= 512
            assert feeds["style"].shape == (1, 256)
            calls.append(feeds["style"][0, 0])
            return [np.zeros((1, 5012), dtype=np.float32)]

    monkeypatch.setattr(runtime.ort, "InferenceSession", Session)
    monkeypatch.setattr(
        runtime.phonemizer.backend, "EspeakBackend", lambda **_kwargs: SimpleNamespace(phonemize=lambda text: text)
    )
    handler = KittenTTSHandler.__new__(KittenTTSHandler)
    handler.setup(Event(), model_name="KittenML/kitten-tts-mini-0.8", blocksize=4)
    calls.clear()
    text = " ".join(["The quick brown fox jumps over the lazy dog"] * 16) + "."

    chunks = list(handler.process(TTSInput(text=text)))

    assert len(text) == 704
    assert len(calls) == 2
    assert calls == [399, 304]
    assert chunks
