from __future__ import annotations

import asyncio
import io
import json
import socket
from queue import Queue
from threading import Event, Thread, current_thread
from time import perf_counter

import numpy as np
import pytest
from openai.types.realtime import RealtimeSessionCreateRequest
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from scipy.io import wavfile

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.control import SESSION_END
from speech_to_speech.pipeline.events import ResponseFailedEvent
from speech_to_speech.pipeline.messages import (
    AUDIO_RESPONSE_DONE,
    PIPELINE_END,
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
    response_bytes: bytes | None = None

    def __init__(self, **kwargs) -> None:
        self.payload = kwargs["payload"]
        self.cancelled = False
        type(self).instances.append(self)

    def iter_bytes(self, cancel_check):
        if type(self).startup_action is not None:
            type(self).startup_action()
        if type(self).startup_error is not None:
            raise type(self).startup_error
        encoded = type(self).response_bytes
        if encoded is None:
            samples = np.arange(2400, dtype="<i2")
            encoded = samples.tobytes()
        for index, offset in enumerate(range(0, len(encoded), 301)):
            if self.cancelled:
                raise SpeechRequestCancelled
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
    _FakeSpeechOperation.response_bytes = None


def _openai_tts_handler(
    monkeypatch,
    *,
    cancel_scope: CancelScope | None = None,
    speculative_turns: SpeculativeTurnTracker | None = None,
) -> OpenAICompatibleTTSHandler:
    _reset_fake_speech_operation()
    monkeypatch.setattr(tts_module, "HttpSpeechOperation", _FakeSpeechOperation)
    handler = OpenAICompatibleTTSHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_args=(Event(),),
        setup_kwargs={
            "sample_rate": 24000,
            "blocksize": 512,
            "cancel_scope": cancel_scope,
            "speculative_turns": speculative_turns,
        },
    )
    _reset_fake_speech_operation()
    return handler


def test_openai_tts_warmup_uses_configured_request(monkeypatch):
    _reset_fake_speech_operation()
    monkeypatch.setattr(tts_module, "HttpSpeechOperation", _FakeSpeechOperation)

    OpenAICompatibleTTSHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_args=(Event(),),
        setup_kwargs={
            "base_url": "http://speech.example/v1",
            "api_key": "test-key",
            "model": "test-model",
            "voice": "test-voice",
            "language": "English",
            "sample_rate": 24000,
            "stream": True,
            "timeout": 12,
            "blocksize": 512,
        },
    )

    assert len(_FakeSpeechOperation.instances) == 1
    operation = _FakeSpeechOperation.instances[0]
    assert operation.payload == {
        "model": "test-model",
        "input": "Warmup",
        "voice": "test-voice",
        "response_format": "pcm",
        "stream_format": "audio",
        "stream": True,
        "language": "English",
    }
    assert operation.cancelled is False


def test_openai_tts_warmup_http_failure_aborts_construction(monkeypatch):
    transport = tts_module.httpx.MockTransport(lambda request: tts_module.httpx.Response(404, request=request))
    client = tts_module.httpx.AsyncClient(transport=transport)
    monkeypatch.setattr(tts_module.httpx, "AsyncClient", lambda **kwargs: client)

    with pytest.raises(tts_module.SpeechRequestError, match="speech server returned HTTP 404"):
        OpenAICompatibleTTSHandler(
            Event(),
            queue_in=Queue(),
            queue_out=Queue(),
            setup_args=(Event(),),
            setup_kwargs={"sample_rate": 24000, "blocksize": 512},
        )


def test_openai_tts_streams_resampled_fixed_size_pcm(monkeypatch):
    handler = _openai_tts_handler(monkeypatch, cancel_scope=CancelScope())
    handler.model = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    handler.voice = "aiden"
    handler.language = "English"

    chunks = list(handler.process(TTSInput(text="Hello", language_code="de")))

    assert chunks
    assert all(isinstance(chunk, np.ndarray) for chunk in chunks)
    assert all(chunk.dtype == np.int16 and chunk.shape == (512,) for chunk in chunks)
    assert sum(chunk.size for chunk in chunks) == 2048
    payload = _FakeSpeechOperation.instances[0].payload
    assert payload["input"] == "Hello"
    assert payload["voice"] == "aiden"
    assert payload["language"] == "English"
    assert "stream" not in payload
    assert payload["stream_format"] == "audio"


def test_openai_tts_does_not_infer_language_from_pipeline(monkeypatch):
    handler = _openai_tts_handler(monkeypatch)

    assert list(handler.process(TTSInput(text="Hello", language_code="en")))

    payload = _FakeSpeechOperation.instances[0].payload
    assert "language" not in payload


def test_openai_tts_streaming_resampler_is_chunk_invariant_and_anti_aliased():
    sample_rate = 24000
    duration_s = 0.2
    timeline = np.arange(int(sample_rate * duration_s)) / sample_rate
    passband = np.round(16000 * np.sin(2 * np.pi * 1000 * timeline)).astype(np.int16)
    stopband = np.round(16000 * np.sin(2 * np.pi * 10000 * timeline)).astype(np.int16)

    def resample(samples, chunk_sizes):
        resampler = tts_module._StreamingFIRResampler(sample_rate, 16000)
        output = []
        offset = 0
        for chunk_size in chunk_sizes:
            output.append(resampler.push(samples[offset : offset + chunk_size]))
            offset += chunk_size
        output.append(resampler.push(samples[offset:]))
        output.append(resampler.push(np.empty(0, dtype=np.int16), final=True))
        return np.concatenate(output)

    passband_streamed = resample(passband, [1, 301, 17, 1024, 3])
    passband_single = resample(passband, [])
    stopband_streamed = resample(stopband, [299, 2, 777, 31])

    np.testing.assert_array_equal(passband_streamed, passband_single)
    assert np.sqrt(np.mean(passband_streamed.astype(np.float64) ** 2)) > 10000
    assert np.sqrt(np.mean(stopband_streamed.astype(np.float64) ** 2)) < 500


def test_openai_tts_vllm_stream_extension_is_opt_in(monkeypatch):
    handler = _openai_tts_handler(monkeypatch)
    handler.stream = True

    assert list(handler.process(TTSInput(text="Hello")))

    payload = _FakeSpeechOperation.instances[0].payload
    assert payload["stream"] is True
    assert payload["stream_format"] == "audio"


def test_openai_tts_preserves_session_custom_voice_id(monkeypatch):
    handler = _openai_tts_handler(monkeypatch)
    runtime_config = RuntimeConfig(
        session=RealtimeSessionCreateRequest(
            type="realtime",
            audio={"output": {"voice": {"id": "voice_session"}}},
        )
    )

    assert list(handler.process(TTSInput(text="Hello", runtime_config=runtime_config)))

    assert _FakeSpeechOperation.instances[0].payload["voice"] == {"id": "voice_session"}


def test_openai_tts_preserves_per_response_custom_voice_id(monkeypatch):
    handler = _openai_tts_handler(monkeypatch)
    response = RealtimeResponseCreateParams(
        audio={"output": {"voice": {"id": "voice_response"}}},
    )

    assert list(handler.process(TTSInput(text="Hello", response=response)))

    assert _FakeSpeechOperation.instances[0].payload["voice"] == {"id": "voice_response"}


def test_openai_tts_standard_request_yields_audio_before_response_eof(monkeypatch):
    release_response = Event()
    response_finished = Event()
    received_payload: dict[str, object] = {}
    encoded = np.arange(2400, dtype="<i2").tobytes()

    class GatedAudioStream(tts_module.httpx.AsyncByteStream):
        async def __aiter__(self):
            yield encoded
            while not release_response.is_set():
                await asyncio.sleep(0.01)
            response_finished.set()

        async def aclose(self) -> None:
            pass

    def respond(request):
        received_payload.update(json.loads(request.content))
        return tts_module.httpx.Response(
            200,
            headers={"Content-Type": "audio/pcm"},
            stream=GatedAudioStream(),
            request=request,
        )

    transport = tts_module.httpx.MockTransport(respond)
    client = tts_module.httpx.AsyncClient(transport=transport)
    monkeypatch.setattr(tts_module.httpx, "AsyncClient", lambda **kwargs: client)
    monkeypatch.setattr(OpenAICompatibleTTSHandler, "warmup", lambda self: None)
    handler = OpenAICompatibleTTSHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_args=(Event(),),
        setup_kwargs={"sample_rate": 24000, "blocksize": 512},
    )

    generation = handler.process(TTSInput(text="Hello"))
    try:
        first_audio = next(generation)
        finished_before_first_audio = response_finished.is_set()
    finally:
        release_response.set()
    remaining_audio = list(generation)

    assert isinstance(first_audio, np.ndarray)
    assert remaining_audio
    assert finished_before_first_audio is False
    assert response_finished.is_set()
    assert received_payload == {
        "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "input": "Hello",
        "voice": "aiden",
        "response_format": "pcm",
        "stream_format": "audio",
    }


def test_openai_tts_http_failure_before_audio_does_not_commit(monkeypatch):
    tracker = _CountingSpeculativeTurnTracker()
    tracker.observe("turn-1", 0)
    handler = _openai_tts_handler(monkeypatch, speculative_turns=tracker)
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
    failure = handler.queue_out.get_nowait()
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
    handler = _openai_tts_handler(monkeypatch)
    _FakeSpeechOperation.failure_after_chunks = 8
    generation = handler.process(TTSInput(text="Hello", turn_id="turn-1", turn_revision=0))

    first_audio = next(generation)
    remaining = list(generation)

    assert isinstance(first_audio, np.ndarray)
    assert remaining == []
    failure = handler.queue_out.get_nowait()
    assert isinstance(failure, ResponseFailedEvent)
    assert failure.message == "speech stream failed"
    assert failure.turn_id == "turn-1"
    assert failure.turn_revision == 0


def test_openai_tts_serializes_failure_after_emitted_audio(monkeypatch):
    handler = _openai_tts_handler(monkeypatch)
    _FakeSpeechOperation.failure_after_chunks = 8
    handler.queue_in.put(
        TTSInput(
            text="Hello",
            turn_id="turn-1",
            turn_revision=0,
            cancel_generation=3,
            response_key="response-1",
        )
    )
    handler.queue_in.put(
        EndOfResponse(
            turn_id="turn-1",
            turn_revision=0,
            cancel_generation=3,
            response_key="response-1",
        )
    )
    handler.queue_in.put(PIPELINE_END)

    thread = Thread(target=handler.run)
    thread.start()
    thread.join(timeout=1)

    assert not thread.is_alive()
    outputs = []
    while not handler.queue_out.empty():
        outputs.append(handler.queue_out.get_nowait())
    audio_indexes = [
        index
        for index, output in enumerate(outputs)
        if isinstance(output, AudioOutput) and isinstance(output.audio, np.ndarray)
    ]
    failure_index = next(index for index, output in enumerate(outputs) if isinstance(output, ResponseFailedEvent))
    done_index = next(
        index
        for index, output in enumerate(outputs)
        if isinstance(output, AudioOutput) and isinstance(output.audio, bytes) and output.audio == AUDIO_RESPONSE_DONE
    )

    assert audio_indexes
    assert max(audio_indexes) < failure_index < done_index


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


def test_openai_tts_failure_does_not_suppress_another_response(monkeypatch):
    handler = _openai_tts_handler(monkeypatch)
    _FakeSpeechOperation.startup_error = tts_module.SpeechRequestError("speech server returned HTTP 500")

    assert (
        list(
            handler.process(
                TTSInput(
                    text="First",
                    turn_id="turn-1",
                    turn_revision=0,
                    cancel_generation=3,
                    response_key="response-1",
                )
            )
        )
        == []
    )
    _FakeSpeechOperation.startup_error = None

    chunks = list(
        handler.process(
            TTSInput(
                text="Second",
                turn_id="turn-1",
                turn_revision=0,
                cancel_generation=3,
                response_key="response-2",
            )
        )
    )

    assert chunks
    assert len(_FakeSpeechOperation.instances) == 2
    assert _FakeSpeechOperation.instances[1].payload["input"] == "Second"


def test_openai_tts_does_not_merge_queued_responses(monkeypatch):
    handler = _openai_tts_handler(monkeypatch)
    second = TTSInput(
        text="Second",
        turn_id="turn-1",
        turn_revision=0,
        cancel_generation=3,
        response_key="response-2",
    )
    handler.queue_in.put(second)

    first_chunks = list(
        handler.process(
            TTSInput(
                text="First",
                turn_id="turn-1",
                turn_revision=0,
                cancel_generation=3,
                response_key="response-1",
            )
        )
    )

    assert first_chunks
    assert _FakeSpeechOperation.instances[0].payload["input"] == "First"
    assert handler.queue_in.get_nowait() is second


def test_openai_tts_empty_success_response_emits_failure(monkeypatch):
    handler = _openai_tts_handler(monkeypatch)
    _FakeSpeechOperation.response_bytes = b""

    assert list(handler.process(TTSInput(text="Hello", response_key="response-1"))) == []

    failure = handler.queue_out.get_nowait()
    assert isinstance(failure, ResponseFailedEvent)
    assert failure.message == "speech endpoint returned no audio"
    assert failure.response_key == "response-1"


def test_openai_tts_decodes_wav_stream(monkeypatch):
    handler = _openai_tts_handler(monkeypatch)
    handler.response_format = "wav"
    handler.stream = False
    encoded = io.BytesIO()
    wavfile.write(encoded, 24000, np.arange(2400, dtype=np.int16))
    _FakeSpeechOperation.response_bytes = encoded.getvalue()

    chunks = list(handler.process(TTSInput(text="Hello")))

    assert chunks
    assert all(chunk.dtype == np.int16 and chunk.shape == (512,) for chunk in chunks)
    assert sum(chunk.size for chunk in chunks) == 2048
    payload = _FakeSpeechOperation.instances[0].payload
    assert payload["response_format"] == "wav"
    assert "stream" not in payload


def test_openai_tts_wav_yields_audio_before_response_eof(monkeypatch):
    release_response = Event()
    response_finished = Event()
    encoded = io.BytesIO()
    wavfile.write(encoded, 24000, np.arange(4800, dtype=np.int16))
    wav_bytes = bytearray(encoded.getvalue())
    wav_bytes[4:8] = b"\xff\xff\xff\xff"
    wav_bytes[40:44] = b"\xff\xff\xff\xff"
    first_part_size = 44 + 2400 * 2

    class GatedWavStream(tts_module.httpx.AsyncByteStream):
        async def __aiter__(self):
            yield bytes(wav_bytes[:first_part_size])
            while not release_response.is_set():
                await asyncio.sleep(0.01)
            yield bytes(wav_bytes[first_part_size:])
            response_finished.set()

        async def aclose(self) -> None:
            pass

    def respond(request):
        return tts_module.httpx.Response(
            200,
            headers={"Content-Type": "audio/wav"},
            stream=GatedWavStream(),
            request=request,
        )

    client = tts_module.httpx.AsyncClient(transport=tts_module.httpx.MockTransport(respond))
    monkeypatch.setattr(tts_module.httpx, "AsyncClient", lambda **kwargs: client)
    monkeypatch.setattr(OpenAICompatibleTTSHandler, "warmup", lambda self: None)
    handler = OpenAICompatibleTTSHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_args=(Event(),),
        setup_kwargs={"response_format": "wav", "blocksize": 512},
    )

    generation = handler.process(TTSInput(text="Hello"))
    try:
        first_audio = next(generation)
        finished_before_first_audio = response_finished.is_set()
    finally:
        release_response.set()
    remaining_audio = list(generation)

    assert isinstance(first_audio, np.ndarray)
    assert remaining_audio
    assert finished_before_first_audio is False
    assert response_finished.is_set()


def test_openai_tts_decodes_streaming_wav_with_unknown_data_length(monkeypatch):
    handler = _openai_tts_handler(monkeypatch)
    handler.response_format = "wav"
    encoded = io.BytesIO()
    wavfile.write(encoded, 24000, np.arange(2400, dtype=np.int16))
    wav_bytes = bytearray(encoded.getvalue())
    wav_bytes[4:8] = b"\xff\xff\xff\xff"
    wav_bytes[40:44] = b"\xff\xff\xff\xff"
    _FakeSpeechOperation.response_bytes = bytes(wav_bytes)

    chunks = list(handler.process(TTSInput(text="Hello")))

    assert chunks
    assert all(chunk.dtype == np.int16 and chunk.shape == (512,) for chunk in chunks)
    assert sum(chunk.size for chunk in chunks) == 2048


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


def test_openai_tts_session_teardown_stops_publication_and_closes_operation(monkeypatch):
    handler = _openai_tts_handler(monkeypatch)
    generation = handler.process(TTSInput(text="Hello"))
    first = next(generation)
    operation = _FakeSpeechOperation.instances[0]

    handler.on_session_end()

    assert isinstance(first, np.ndarray)
    assert list(generation) == []
    assert operation.cancelled is True
    assert handler._active_operation is None
    assert handler.queue_out.empty()


def test_http_speech_cancellation_after_completion_is_harmless(monkeypatch):
    transport = tts_module.httpx.MockTransport(
        lambda request: tts_module.httpx.Response(200, content=b"\x00\x00", request=request)
    )
    client = tts_module.httpx.AsyncClient(transport=transport)
    monkeypatch.setattr(tts_module.httpx, "AsyncClient", lambda **kwargs: client)
    operation = HttpSpeechOperation(
        endpoint_url="http://localhost:8000/v1/audio/speech",
        api_key=None,
        payload={"input": "hello"},
        timeout_s=2,
    )

    assert list(operation.iter_bytes(lambda: False)) == [b"\x00\x00"]
    assert client.is_closed
    operation.cancel()
    operation.cancel()

    assert client.is_closed


def test_http_speech_cancellation_during_client_construction_prevents_dispatch(monkeypatch):
    construction_started = Event()
    resume_construction = Event()
    cancelled = Event()
    worker_observed_cancellation = Event()
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

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(tts_module.httpx, "AsyncClient", BlockingClient)
    operation = HttpSpeechOperation(
        endpoint_url="http://localhost:8000/v1/audio/speech",
        api_key=None,
        payload={"input": "hello"},
        timeout_s=2,
    )
    errors: list[BaseException] = []

    def cancel_check() -> bool:
        if not cancelled.is_set():
            return False
        if current_thread().name == "tts-http-reader":
            worker_observed_cancellation.set()
        else:
            assert worker_observed_cancellation.wait(1)
        return True

    def consume() -> None:
        try:
            list(operation.iter_bytes(cancel_check))
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=consume)
    thread.start()
    assert construction_started.wait(1)
    cancelled.set()
    resume_construction.set()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], SpeechRequestCancelled)
    assert worker_observed_cancellation.is_set()
    assert stream_calls == 0


def test_http_speech_timeout_is_a_total_deadline(monkeypatch):
    closed = Event()

    class TricklingResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            del args
            await self.aclose()

        async def aclose(self) -> None:
            closed.set()

        def raise_for_status(self) -> None:
            pass

        async def aiter_bytes(self):
            while not closed.wait(0.01):
                yield b"\x00\x00"
                await asyncio.sleep(0)

    response = TricklingResponse()

    class TricklingClient:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def stream(self, *args, **kwargs):
            del args, kwargs
            return response

        async def aclose(self) -> None:
            await response.aclose()

    monkeypatch.setattr(tts_module.httpx, "AsyncClient", TricklingClient)
    operation = HttpSpeechOperation(
        endpoint_url="http://localhost:8000/v1/audio/speech",
        api_key=None,
        payload={"input": "hello"},
        timeout_s=0.05,
    )

    started_at_s = perf_counter()
    with pytest.raises(tts_module.SpeechRequestError, match="speech request timed out"):
        list(operation.iter_bytes(lambda: False))

    assert closed.is_set()
    assert perf_counter() - started_at_s < 0.5


@pytest.mark.parametrize(
    "content_type",
    ["application/json; charset=utf-8", "application/problem+json", "text/plain"],
)
def test_openai_tts_rejects_non_audio_success_responses(monkeypatch, content_type):
    transport = tts_module.httpx.MockTransport(
        lambda request: tts_module.httpx.Response(
            200,
            headers={"Content-Type": content_type},
            content=b'{"error":"upstream failure"}',
            request=request,
        )
    )
    client = tts_module.httpx.AsyncClient(transport=transport)
    monkeypatch.setattr(tts_module.httpx, "AsyncClient", lambda **kwargs: client)
    monkeypatch.setattr(OpenAICompatibleTTSHandler, "warmup", lambda self: None)
    handler = OpenAICompatibleTTSHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_args=(Event(),),
        setup_kwargs={"sample_rate": 24000, "blocksize": 512},
    )

    assert list(handler.process(TTSInput(text="Hello", response_key="response-1"))) == []

    failure = handler.queue_out.get_nowait()
    assert isinstance(failure, ResponseFailedEvent)
    assert failure.message == "speech endpoint returned a non-audio response"
    assert failure.response_key == "response-1"


def test_http_speech_cancellation_closes_transport_before_response_headers():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    host, port = listener.getsockname()
    request_received = Event()
    connection_closed = Event()
    cancelled = Event()

    def serve_stalled_response() -> None:
        connection, _ = listener.accept()
        with connection:
            request = b""
            while b"\r\n\r\n" not in request:
                data = connection.recv(4096)
                if not data:
                    return
                request += data
            request_received.set()
            while connection.recv(4096):
                pass
            connection_closed.set()

    server_thread = Thread(target=serve_stalled_response, daemon=True)
    server_thread.start()
    operation = HttpSpeechOperation(
        endpoint_url=f"http://{host}:{port}/v1/audio/speech",
        api_key=None,
        payload={"input": "hello"},
        timeout_s=30,
    )
    errors: list[BaseException] = []

    def consume() -> None:
        try:
            list(operation.iter_bytes(cancelled.is_set))
        except BaseException as exc:
            errors.append(exc)

    consumer = Thread(target=consume)
    consumer.start()

    try:
        assert request_received.wait(1)
        cancelled.set()
        consumer.join(timeout=1)

        assert not consumer.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], SpeechRequestCancelled)
        assert connection_closed.wait(1)
        assert operation._worker_task is None
        assert operation._worker_loop is None
    finally:
        listener.close()
        server_thread.join(timeout=1)


def test_openai_tts_cancellation_unblocks_a_stalled_socket_and_session_end(monkeypatch):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    host, port = listener.getsockname()
    headers_sent = Event()
    release_server = Event()

    def serve_stalled_response() -> None:
        connection, _ = listener.accept()
        with connection:
            request = b""
            while b"\r\n\r\n" not in request:
                data = connection.recv(4096)
                if not data:
                    return
                request += data
            connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\nTransfer-Encoding: chunked\r\n\r\n")
            headers_sent.set()
            release_server.wait(5)

    server_thread = Thread(target=serve_stalled_response, daemon=True)
    server_thread.start()
    cancel_scope = CancelScope()
    monkeypatch.setattr(OpenAICompatibleTTSHandler, "warmup", lambda self: None)
    handler = OpenAICompatibleTTSHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_args=(Event(),),
        setup_kwargs={
            "base_url": f"http://{host}:{port}/v1",
            "sample_rate": 24000,
            "blocksize": 512,
            "timeout": 30,
            "cancel_scope": cancel_scope,
        },
    )
    handler.queue_in.put(
        TTSInput(
            text="Hello",
            cancel_generation=cancel_scope.generation,
        )
    )
    handler_thread = Thread(target=handler.run, daemon=True)
    handler_thread.start()

    try:
        assert headers_sent.wait(1)
        cancel_scope.cancel()
        handler.queue_in.put(SESSION_END)
        handler.queue_in.put(PIPELINE_END)

        assert handler.queue_out.get(timeout=1) == SESSION_END
        handler_thread.join(timeout=1)
        assert not handler_thread.is_alive()
        assert handler.queue_out.get_nowait() == PIPELINE_END
    finally:
        release_server.set()
        listener.close()
        server_thread.join(timeout=1)
