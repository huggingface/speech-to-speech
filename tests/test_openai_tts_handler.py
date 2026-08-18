from __future__ import annotations

from queue import Queue
from threading import Event, Thread

import numpy as np

from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import ResponseFailedEvent
from speech_to_speech.pipeline.messages import (
    AUDIO_RESPONSE_DONE,
    AudioOutput,
    EndOfResponse,
    TTSInput,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.TTS import openai_compatible_handler as tts_module
from speech_to_speech.TTS.openai_compatible_handler import (
    HttpSpeechOperation,
    OpenAICompatibleTTSHandler,
    SpeechRequestCancelled,
)


class _FakeSpeechOperation:
    instances: list["_FakeSpeechOperation"] = []
    startup_action = None
    startup_error: Exception | None = None
    failure_after_chunks: int | None = None

    def __init__(self, **kwargs) -> None:
        self.payload = kwargs["payload"]
        self.cancelled = False
        type(self).instances.append(self)

    def iter_bytes(self, cancel_check):
        if type(self).startup_action is not None:
            type(self).startup_action()
        if type(self).startup_error is not None:
            raise type(self).startup_error
        samples = np.arange(2400, dtype="<i2")
        encoded = samples.tobytes()
        for index, offset in enumerate(range(0, len(encoded), 301)):
            if type(self).failure_after_chunks == index:
                raise tts_module.SpeechRequestError("speech stream failed")
            if cancel_check():
                self.cancel()
                return
            yield encoded[offset : offset + 301]

    def cancel(self) -> None:
        self.cancelled = True


class _CountingSpeculativeTurnTracker(SpeculativeTurnTracker):
    def __init__(self) -> None:
        super().__init__()
        self.commit_calls = 0

    def commit_if_latest_after_reopen_grace(self, turn_id: str | None, revision: int | None) -> bool:
        self.commit_calls += 1
        return super().commit_if_latest_after_reopen_grace(turn_id, revision)


def _reset_fake_speech_operation() -> None:
    _FakeSpeechOperation.instances.clear()
    _FakeSpeechOperation.startup_action = None
    _FakeSpeechOperation.startup_error = None
    _FakeSpeechOperation.failure_after_chunks = None


def _openai_tts_handler(
    monkeypatch,
    *,
    cancel_scope: CancelScope | None = None,
    speculative_turns: SpeculativeTurnTracker | None = None,
    text_output_queue: Queue | None = None,
) -> OpenAICompatibleTTSHandler:
    _reset_fake_speech_operation()
    monkeypatch.setattr(tts_module, "HttpSpeechOperation", _FakeSpeechOperation)
    return OpenAICompatibleTTSHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_args=(Event(),),
        setup_kwargs={
            "sample_rate": 24000,
            "blocksize": 512,
            "cancel_scope": cancel_scope,
            "speculative_turns": speculative_turns,
            "text_output_queue": text_output_queue,
        },
    )


def test_openai_tts_streams_resampled_fixed_size_pcm(monkeypatch):
    handler = _openai_tts_handler(monkeypatch, cancel_scope=CancelScope())
    handler.model = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    handler.voice = "aiden"
    handler.language = "Auto"

    chunks = list(handler.process(TTSInput(text="Hello", language_code="en")))

    assert chunks
    assert all(isinstance(chunk, np.ndarray) for chunk in chunks)
    assert all(chunk.dtype == np.int16 and chunk.shape == (512,) for chunk in chunks)
    assert sum(chunk.size for chunk in chunks) == 2048
    payload = _FakeSpeechOperation.instances[0].payload
    assert payload["input"] == "Hello"
    assert payload["voice"] == "aiden"
    assert payload["language"] == "English"
    assert payload["stream"] is True
    assert payload["stream_format"] == "audio"


def test_openai_tts_http_failure_before_audio_does_not_commit(monkeypatch):
    tracker = _CountingSpeculativeTurnTracker()
    tracker.observe("turn-1", 0)
    text_output_queue = Queue()
    handler = _openai_tts_handler(
        monkeypatch,
        speculative_turns=tracker,
        text_output_queue=text_output_queue,
    )
    _FakeSpeechOperation.startup_error = tts_module.SpeechRequestError("speech server returned HTTP 500")

    chunks = list(
        handler.process(
            TTSInput(
                text="Hello",
                turn_id="turn-1",
                turn_revision=0,
                response_key="response-1",
            )
        )
    )

    assert chunks == []
    assert tracker.commit_calls == 0
    assert not tracker.is_committed("turn-1", 0)
    failure = text_output_queue.get_nowait()
    assert isinstance(failure, ResponseFailedEvent)
    assert failure.message == "speech server returned HTTP 500"
    assert failure.turn_id == "turn-1"
    assert failure.turn_revision == 0
    assert failure.response_key == "response-1"


def test_openai_tts_stale_keyed_terminal_becomes_cleanup(monkeypatch):
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn-1", 0)
    handler = _openai_tts_handler(monkeypatch, speculative_turns=tracker)
    terminal = EndOfResponse(
        response_key="response-1",
        turn_id="turn-1",
        turn_revision=0,
        cancel_generation=7,
    )
    tracker.observe("turn-1", 1)

    outputs = list(handler.process(terminal))
    queued = handler.output_for_queue(outputs[0], terminal)

    assert outputs == [AUDIO_RESPONSE_DONE]
    assert terminal.cleanup_only is True
    assert isinstance(queued, AudioOutput)
    assert queued.response_key == "response-1"
    assert queued.cancel_generation == 7
    assert queued.cleanup_only is True


def test_openai_tts_failure_after_audio_emits_response_failure(monkeypatch):
    text_output_queue = Queue()
    handler = _openai_tts_handler(monkeypatch, text_output_queue=text_output_queue)
    _FakeSpeechOperation.failure_after_chunks = 8
    generation = handler.process(TTSInput(text="Hello", turn_id="turn-1", turn_revision=0))

    first_audio = next(generation)
    remaining = list(generation)

    assert isinstance(first_audio, np.ndarray)
    assert remaining == []
    failure = text_output_queue.get_nowait()
    assert isinstance(failure, ResponseFailedEvent)
    assert failure.message == "speech stream failed"
    assert failure.turn_id == "turn-1"
    assert failure.turn_revision == 0


def test_openai_tts_suppresses_remaining_chunks_after_failure_until_response_end(monkeypatch):
    handler = _openai_tts_handler(monkeypatch)
    _FakeSpeechOperation.startup_error = tts_module.SpeechRequestError("speech server returned HTTP 500")
    tts_input = TTSInput(
        text="Hello",
        turn_id="turn-1",
        turn_revision=0,
        cancel_generation=3,
    )

    assert list(handler.process(tts_input)) == []
    assert len(_FakeSpeechOperation.instances) == 1
    assert list(handler.process(tts_input)) == []
    assert len(_FakeSpeechOperation.instances) == 1

    assert list(
        handler.process(
            EndOfResponse(
                turn_id="turn-1",
                turn_revision=0,
                cancel_generation=3,
            )
        )
    ) == [AUDIO_RESPONSE_DONE]
    _FakeSpeechOperation.startup_error = None
    assert list(handler.process(tts_input))
    assert len(_FakeSpeechOperation.instances) == 2


def test_openai_tts_cancellation_before_audio_does_not_commit(monkeypatch):
    tracker = _CountingSpeculativeTurnTracker()
    tracker.observe("turn-1", 0)
    cancel_scope = CancelScope()
    handler = _openai_tts_handler(
        monkeypatch,
        cancel_scope=cancel_scope,
        speculative_turns=tracker,
    )
    _FakeSpeechOperation.startup_action = cancel_scope.cancel

    chunks = list(
        handler.process(
            TTSInput(
                text="Hello",
                turn_id="turn-1",
                turn_revision=0,
                cancel_generation=cancel_scope.generation,
            )
        )
    )

    assert chunks == []
    assert tracker.commit_calls == 0
    assert not tracker.is_committed("turn-1", 0)
    assert _FakeSpeechOperation.instances[0].cancelled is True


def test_openai_tts_first_emitted_audio_commits_exactly_once(monkeypatch):
    tracker = _CountingSpeculativeTurnTracker()
    tracker.observe("turn-1", 0)
    handler = _openai_tts_handler(monkeypatch, speculative_turns=tracker)

    chunks = list(
        handler.process(
            TTSInput(
                text="Hello",
                turn_id="turn-1",
                turn_revision=0,
            )
        )
    )

    assert chunks
    assert tracker.commit_calls == 1
    assert tracker.is_committed("turn-1", 0)


def test_openai_tts_reopened_during_startup_suppresses_old_revision(monkeypatch):
    tracker = _CountingSpeculativeTurnTracker()
    tracker.observe("turn-1", 0)
    handler = _openai_tts_handler(monkeypatch, speculative_turns=tracker)
    _FakeSpeechOperation.startup_action = lambda: tracker.observe("turn-1", 1)

    chunks = list(
        handler.process(
            TTSInput(
                text="Hello",
                turn_id="turn-1",
                turn_revision=0,
            )
        )
    )

    assert chunks == []
    assert tracker.commit_calls == 0
    assert not tracker.is_committed("turn-1", 0)
    assert _FakeSpeechOperation.instances[0].cancelled is True


def test_openai_tts_cancellation_stops_publication_and_closes_operation(monkeypatch):
    cancel_scope = CancelScope()
    handler = _openai_tts_handler(monkeypatch, cancel_scope=cancel_scope)

    generation = handler.process(TTSInput(text="Hello", cancel_generation=cancel_scope.generation))
    first = next(generation)
    cancel_scope.cancel()

    assert isinstance(first, np.ndarray)
    assert list(generation) == []
    assert _FakeSpeechOperation.instances[0].cancelled is True


def test_http_speech_cancellation_during_client_construction_prevents_dispatch(monkeypatch):
    construction_started = Event()
    resume_construction = Event()
    stream_calls = 0

    class BlockingClient:
        def __init__(self, **kwargs) -> None:
            del kwargs
            construction_started.set()
            assert resume_construction.wait(1)

        def stream(self, *args, **kwargs):
            nonlocal stream_calls
            del args, kwargs
            stream_calls += 1
            raise AssertionError("cancelled operation must not enter client.stream()")

        def close(self) -> None:
            pass

    monkeypatch.setattr(tts_module.httpx, "Client", BlockingClient)
    operation = HttpSpeechOperation(
        endpoint_url="http://localhost:8000/v1/audio/speech",
        api_key=None,
        payload={"input": "hello"},
        timeout_s=2,
    )
    errors: list[BaseException] = []

    def consume() -> None:
        try:
            list(operation.iter_bytes(lambda: False))
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=consume)
    thread.start()
    assert construction_started.wait(1)
    operation.cancel()
    resume_construction.set()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], SpeechRequestCancelled)
    assert stream_calls == 0
