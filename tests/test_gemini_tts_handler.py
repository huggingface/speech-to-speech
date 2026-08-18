import base64
from queue import Queue
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

import speech_to_speech.TTS.gemini_tts_handler as gemini_tts
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import ResponseFailedEvent
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput


class _FakeStream:
    def __init__(self, events, error=None, on_second_event=None):
        self.events = events
        self.error = error
        self.on_second_event = on_second_event
        self.closed = False

    def __iter__(self):
        for index, event in enumerate(self.events):
            if index == 1 and self.on_second_event is not None:
                self.on_second_event()
            yield event
        if self.error is not None:
            raise self.error

    def close(self):
        self.closed = True


class _FakeInteractions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _StatusError(RuntimeError):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def _audio_event(raw):
    return SimpleNamespace(
        event_type="step.delta",
        delta=SimpleNamespace(type="audio", data=base64.b64encode(raw).decode("ascii")),
    )


def _handler(monkeypatch, outcomes, **kwargs):
    interactions = _FakeInteractions(outcomes)
    captured = {}

    class FakeClient:
        def __init__(self, **client_kwargs):
            captured.update(client_kwargs)
            self.interactions = interactions

    monkeypatch.setattr(gemini_tts, "genai", SimpleNamespace(Client=FakeClient))
    handler = gemini_tts.GeminiTTSHandler(
        Event(),
        Queue(),
        Queue(),
        setup_args=(Event(),),
        setup_kwargs={"api_key": "secret", **kwargs},
    )
    return handler, interactions, captured


def _one_second_pcm():
    time = np.arange(24000) / 24000
    return (np.sin(2 * np.pi * 440 * time) * 12000).astype("<i2").tobytes()


def test_setup_requires_key_and_uses_environment(monkeypatch):
    monkeypatch.setattr(gemini_tts, "genai", SimpleNamespace(Client=lambda **kwargs: kwargs))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    handler = object.__new__(gemini_tts.GeminiTTSHandler)

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        handler.setup(Event())

    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    handler.setup(Event())
    assert handler.client["api_key"] == "from-env"
    assert handler.client["http_options"] == {"timeout": 20000}


def test_streams_base64_pcm_with_continuous_resampling_and_chunking(monkeypatch):
    pcm = _one_second_pcm()
    stream = _FakeStream([_audio_event(pcm[:12001]), _audio_event(pcm[12001:])])
    handler, interactions, _captured = _handler(monkeypatch, [stream])

    outputs = list(handler.process(TTSInput(text="Dzień dobry.")))

    assert outputs
    assert all(isinstance(output, np.ndarray) for output in outputs)
    assert all(output.dtype == np.int16 and output.shape == (512,) for output in outputs)
    assert 16000 <= sum(len(output) for output in outputs) < 16512
    request = interactions.calls[0]
    assert request["model"] == "gemini-3.1-flash-tts-preview"
    assert request["response_format"] == {"type": "audio"}
    assert request["generation_config"] == {"speech_config": [{"voice": "Kore"}]}
    assert request["stream"] is True
    assert "Dzień dobry." in request["input"]
    assert "pl-PL" not in str(request)
    assert stream.closed


def test_voice_catalog_and_session_override(monkeypatch, caplog):
    assert len(gemini_tts.GEMINI_TTS_VOICES) == 30
    assert len(set(gemini_tts.GEMINI_TTS_VOICES)) == 30
    handler, _interactions, _captured = _handler(monkeypatch, [])
    runtime_config = SimpleNamespace(
        session=SimpleNamespace(audio=SimpleNamespace(output=SimpleNamespace(voice="puck")))
    )

    assert handler._session_voice(runtime_config, None) == "Puck"

    runtime_config.session.audio.output.voice = "not-a-voice"
    with caplog.at_level("ERROR"):
        assert handler._session_voice(runtime_config, None) == "Kore"
    assert "Rejecting unsupported Gemini TTS session voice" in caplog.text


def test_end_of_response_always_emits_audio_done(monkeypatch):
    handler, _interactions, _captured = _handler(monkeypatch, [])
    assert list(handler.process(EndOfResponse())) == [AUDIO_RESPONSE_DONE]


def test_retries_one_transient_error_only_before_audio(monkeypatch):
    stream = _FakeStream([_audio_event(_one_second_pcm())])
    handler, interactions, _captured = _handler(monkeypatch, [TimeoutError("timeout"), stream])

    outputs = list(handler.process(TTSInput(text="Retry me")))

    assert len(interactions.calls) == 2
    assert outputs and not any(isinstance(output, ResponseFailedEvent) for output in outputs)


@pytest.mark.parametrize("status_code", [400, 401, 403])
def test_does_not_retry_client_or_auth_errors(monkeypatch, status_code):
    handler, interactions, _captured = _handler(monkeypatch, [_StatusError(status_code)])

    outputs = list(handler.process(TTSInput(text="Fail")))

    assert len(interactions.calls) == 1
    assert len(outputs) == 1
    assert isinstance(outputs[0], ResponseFailedEvent)


def test_does_not_retry_after_first_audio(monkeypatch):
    stream = _FakeStream([_audio_event(_one_second_pcm())], error=_StatusError(500))
    handler, interactions, _captured = _handler(monkeypatch, [stream])

    outputs = list(handler.process(TTSInput(text="Partial")))

    assert len(interactions.calls) == 1
    assert any(isinstance(output, np.ndarray) for output in outputs)
    assert isinstance(outputs[-1], ResponseFailedEvent)


def test_interruption_closes_stream_without_late_audio(monkeypatch):
    cancel_scope = CancelScope()
    stream = _FakeStream(
        [_audio_event(_one_second_pcm()), _audio_event(_one_second_pcm())],
        on_second_event=cancel_scope.cancel,
    )
    handler, _interactions, _captured = _handler(monkeypatch, [stream], cancel_scope=cancel_scope)

    outputs = list(handler.process(TTSInput(text="Interrupt", cancel_generation=0)))

    assert outputs
    assert all(isinstance(output, np.ndarray) for output in outputs)
    assert stream.closed
