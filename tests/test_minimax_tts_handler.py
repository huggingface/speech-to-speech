"""MiniMax provider, real HTTP streaming, and current pipeline regressions."""

from __future__ import annotations

import json
import os
import socket
from contextlib import contextmanager
from queue import Queue
from threading import Event, Thread

import httpx
import numpy as np
import pytest

from speech_to_speech.backend_registry import TTS_BACKENDS, HandlerContext, create_backend_handler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.control import SESSION_END
from speech_to_speech.pipeline.events import ResponseFailedEvent
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, PIPELINE_END, AudioOutput, EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.s2s_pipeline import parse_arguments, prepare_all_args
from speech_to_speech.TTS.minimax_tts_handler import MiniMaxTTSHandler
from tests.turns import reopen


def _handler(**kwargs):
    return MiniMaxTTSHandler(
        Event(), Queue(), Queue(), setup_args=(Event(),), setup_kwargs={"api_key": "test-key", **kwargs}
    )


def _event(audio=b"", *, status=1, status_code=0):
    return (
        "data: "
        + json.dumps({"data": {"status": status, "audio": audio.hex()}, "base_resp": {"status_code": status_code}})
        + "\r\n\r\n"
    ).encode()


def _transport(monkeypatch, body, *, content_type="text/event-stream", status_code=200):
    requests = []
    original = httpx.AsyncClient

    def respond(request):
        requests.append(request)
        return httpx.Response(status_code, headers={"Content-Type": content_type}, content=body, request=request)

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs)
    )
    return requests


@contextmanager
def _streaming_endpoint(initial):
    """Send one HTTP chunk, then hold the socket open until released or cancelled."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(2)
    sent = Event()
    closed = Event()
    release = Event()

    def serve():
        connection, _ = listener.accept()
        with connection:
            connection.settimeout(0.05)
            request = b""
            while b"\r\n\r\n" not in request:
                request += connection.recv(4096)
            headers, body = request.split(b"\r\n\r\n", 1)
            content_length = int(
                next(
                    line.split(b":", 1)[1]
                    for line in headers.split(b"\r\n")
                    if line.lower().startswith(b"content-length:")
                )
            )
            while len(body) < content_length:
                body += connection.recv(4096)
            connection.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
            )
            if initial:
                connection.sendall(f"{len(initial):x}\r\n".encode() + initial + b"\r\n")
            sent.set()
            while not release.is_set():
                try:
                    if not connection.recv(4096):
                        closed.set()
                        return
                except socket.timeout:
                    continue
            final = _event(status=2)
            try:
                connection.sendall(f"{len(final):x}\r\n".encode() + final + b"\r\n0\r\n\r\n")
            except OSError:
                pass

    server = Thread(target=serve, daemon=True)
    server.start()
    try:
        yield f"http://127.0.0.1:{listener.getsockname()[1]}", sent, closed, release
    finally:
        release.set()
        server.join(2)
        listener.close()
        assert not server.is_alive()


def test_requires_minimax_key_and_does_not_fall_back_to_openai(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-key")
    with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
        _handler(api_key=None)


def test_environment_key_and_explicit_override(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "env-key")
    assert _handler(api_key=None).api_key == "env-key"
    assert _handler(api_key="explicit-key").api_key == "explicit-key"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"speed": 0.1},
        {"vol": 0},
        {"pitch": 13},
        {"blocksize": 0},
        {"timeout": 0},
        {"timeout": float("inf")},
        {"timeout": float("nan")},
    ],
)
def test_invalid_settings_fail_before_network_dispatch(kwargs):
    with pytest.raises(ValueError):
        _handler(**kwargs)


@pytest.mark.parametrize("model", ["speech-2.8-hd", "speech-2.8-turbo"])
@pytest.mark.parametrize("content_type", ["text/event-stream", "application/json"])
def test_payload_and_credentials_use_minimax_protocol(monkeypatch, model, content_type):
    requests = _transport(
        monkeypatch, _event(np.arange(512, dtype="<i2").tobytes()) + _event(status=2), content_type=content_type
    )
    handler = _handler(base_url="https://speech.example/", model=model, speed=1.5, vol=2, pitch=-2)
    assert requests == []  # No paid warmup.
    assert list(handler.process(TTSInput(text="Hello")))
    request = requests[0]
    assert str(request.url) == "https://speech.example/v1/t2a_v2"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert json.loads(request.content) == {
        "model": model,
        "text": "Hello",
        "stream": True,
        "voice_setting": {"voice_id": "English_Graceful_Lady", "speed": 1.5, "vol": 2, "pitch": -2},
        "audio_setting": {"sample_rate": 16000, "format": "pcm", "channel": 1},
        "stream_options": {"exclude_aggregated_audio": True},
    }


@pytest.mark.parametrize("sample_count", [100, 512, 1024, 1500])
def test_fragmented_sse_preserves_samples_and_pads_only_tail(sample_count):
    handler = _handler()
    samples = np.arange(sample_count, dtype="<i2")
    pcm = samples.tobytes()
    # Split inside a PCM16 sample as well as inside JSON and CRLF framing.
    body = _event(pcm[:3]) + _event(pcm[3:]) + _event(pcm, status=2)
    fragments = iter(body[i : i + 37] for i in range(0, len(body), 37))
    chunks = list(handler._decode_pcm_stream(fragments))
    assert chunks
    assert all(chunk.dtype == np.int16 and chunk.shape == (512,) for chunk in chunks)
    output = np.concatenate(chunks)
    assert len(output) == ((sample_count + 511) // 512) * 512
    np.testing.assert_array_equal(output[:sample_count], samples)
    assert np.all(output[sample_count:] == 0)


def test_null_data_heartbeat_is_safe():
    body = b'data: {"data":null,"base_resp":{"status_code":0}}\n\n'
    body += _event(np.zeros(512, dtype="<i2").tobytes()) + _event(status=2)
    assert len(list(_handler()._decode_pcm_stream(iter([body])))) == 1


def test_small_complete_audio_event_is_delivered_before_http_completion():
    samples = np.arange(512, dtype="<i2")
    with _streaming_endpoint(_event(samples.tobytes())) as (base_url, sent, _closed, release):
        handler = _handler(base_url=base_url, timeout=2)
        generation = handler.process(TTSInput(text="Hello"))
        first = Queue()
        worker = Thread(target=lambda: first.put(next(generation)), daemon=True)
        worker.start()
        try:
            assert sent.wait(1)
            chunk = first.get(timeout=1)
            np.testing.assert_array_equal(chunk, samples)
            assert not release.is_set()
        finally:
            release.set()
            worker.join(2)
        assert list(generation) == []
        assert not worker.is_alive()


def test_terminal_status_closes_stream_without_waiting_for_http_eof():
    body = _event(np.zeros(512, dtype="<i2").tobytes()) + _event(status=2)
    with _streaming_endpoint(body) as (base_url, _sent, closed, _release):
        chunks = list(_handler(base_url=base_url, timeout=1).process(TTSInput(text="Hello")))
        assert len(chunks) == 1
        assert closed.wait(1)


@pytest.mark.parametrize("action", ["cancel", "stop", "session_end"])
def test_stalled_request_is_closed_and_worker_released(action):
    with _streaming_endpoint(b"") as (base_url, sent, closed, _release):
        scope = CancelScope()
        handler = _handler(base_url=base_url, timeout=30, cancel_scope=scope)
        outputs = []
        worker = Thread(
            target=lambda: outputs.extend(handler.process(TTSInput(text="Hello", cancel_generation=0))), daemon=True
        )
        worker.start()
        try:
            assert sent.wait(1)
            if action == "cancel":
                scope.cancel()
                scope.new_response()
            elif action == "stop":
                handler.stop_event.set()
            else:
                handler.on_session_end()
            worker.join(1)
            assert not worker.is_alive()
            assert closed.wait(1)
            assert outputs == []
            assert handler.queue_out.empty()
            assert handler._active_operation is None
        finally:
            handler.stop_event.set()
            worker.join(2)


def test_stalled_request_has_finite_timeout_and_reports_failure():
    with _streaming_endpoint(b"") as (base_url, _sent, closed, _release):
        handler = _handler(base_url=base_url, timeout=0.1)
        assert list(handler.process(TTSInput(text="Hello", response_key="response-1"))) == []
        failure = handler.queue_out.get_nowait()
        assert isinstance(failure, ResponseFailedEvent)
        assert failure.message == "speech request timed out"
        assert failure.response_key == "response-1"
        assert closed.wait(1)


def test_cancel_then_new_response_discards_buffered_old_audio(monkeypatch):
    _transport(monkeypatch, _event(np.arange(1500, dtype="<i2").tobytes()) + _event(status=2))
    scope = CancelScope()
    handler = _handler(cancel_scope=scope)
    generation = handler.process(TTSInput(text="Old response", cancel_generation=0, response_key="old"))
    assert isinstance(next(generation), np.ndarray)
    scope.cancel()
    scope.new_response()
    assert list(generation) == []  # Includes the partial padded tail.
    assert handler.queue_out.empty()
    assert list(handler.process(TTSInput(text="New response", cancel_generation=1, response_key="new")))


def test_reopened_turn_during_http_startup_never_commits_or_emits(monkeypatch):
    tracker = SpeculativeTurnTracker()
    tracker.start_turn()
    original = httpx.AsyncClient

    def respond(request):
        reopen(tracker)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=_event(np.zeros(512, dtype="<i2").tobytes()) + _event(status=2),
            request=request,
        )

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs)
    )
    handler = _handler(speculative_turns=tracker)
    assert list(handler.process(TTSInput(text="Hello", turn_id="turn_1", turn_revision=0))) == []
    assert not tracker.is_committed("turn_1", 0)


@pytest.mark.parametrize(
    "body,content_type,status_code",
    [
        (_event(status_code=1004), "text/event-stream", 200),
        (b"data: {invalid json}\n\n", "text/event-stream", 200),
        (b'data: {"data":{"status":1,"audio":"invalid hex"}}\n\n', "text/event-stream", 200),
        (_event(np.zeros(512, dtype="<i2").tobytes()), "text/event-stream", 200),
        (_event(status=2), "text/event-stream", 200),
        (b'{"error":"credentials rejected"}', "application/json", 200),
        (b"private upstream error", "text/plain", 401),
    ],
)
def test_failures_are_attributed_and_do_not_become_success(monkeypatch, body, content_type, status_code):
    requests = _transport(monkeypatch, body, content_type=content_type, status_code=status_code)
    handler = _handler()
    item = TTSInput(
        text="Private sentence", turn_id="turn_1", turn_revision=0, cancel_generation=0, response_key="response-1"
    )
    list(handler.process(item))
    failure = handler.queue_out.get_nowait()
    assert isinstance(failure, ResponseFailedEvent)
    assert (failure.turn_id, failure.turn_revision, failure.cancel_generation, failure.response_key) == (
        "turn_1",
        0,
        0,
        "response-1",
    )
    assert "Private sentence" not in failure.message and "private upstream error" not in failure.message
    assert list(handler.process(item)) == []
    assert len(requests) == 1  # Do not continue speaking later sentences of a failed response.
    assert list(
        handler.process(
            EndOfResponse(response_key="response-1", cancel_generation=0, turn_id="turn_1", turn_revision=0)
        )
    ) == [AUDIO_RESPONSE_DONE]
    list(handler.process(TTSInput(text="Next response", response_key="response-2")))
    assert len(requests) == 2


def test_worker_preserves_audio_and_completion_identity(monkeypatch):
    _transport(monkeypatch, _event(np.zeros(512, dtype="<i2").tobytes()) + _event(status=2))
    handler = _handler()
    handler.queue_in.put(TTSInput(text="Hello", response_key="response-1", cancel_generation=3))
    handler.queue_in.put(EndOfResponse(response_key="response-1", cancel_generation=3))
    handler.queue_in.put(SESSION_END)
    handler.queue_in.put(PIPELINE_END)
    handler.run()
    audio = handler.queue_out.get_nowait()
    done = handler.queue_out.get_nowait()
    assert isinstance(audio, AudioOutput) and isinstance(done, AudioOutput)
    assert (audio.response_key, audio.cancel_generation) == ("response-1", 3)
    assert (done.response_key, done.cancel_generation, done.audio) == ("response-1", 3, AUDIO_RESPONSE_DONE)
    assert handler.queue_out.get_nowait() == SESSION_END
    assert handler.queue_out.get_nowait() == PIPELINE_END


def test_current_cli_registry_constructs_provider_and_wires_cancellation(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "speech-to-speech",
            "--tts",
            "minimax",
            "--minimax_tts_api_key",
            "test-key",
            "--minimax_tts_model",
            "speech-2.8-turbo",
        ],
    )
    args = parse_arguments()
    prepare_all_args(args)
    context = HandlerContext(
        stop_event=Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        text_output_queue=Queue(),
        should_listen=Event(),
        cancel_scope=CancelScope(),
        speculative_turns=SpeculativeTurnTracker(),
        pipeline_index=0,
        sample_rate=16000,
        enable_live_transcription=False,
        live_transcription_update_interval=0.5,
    )
    assert args.tts_backend.spec is TTS_BACKENDS["minimax"]
    handler = create_backend_handler(args.tts_backend, context)
    assert isinstance(handler, MiniMaxTTSHandler)
    assert handler.model == "speech-2.8-turbo"
    assert handler.cancel_scope is context.cancel_scope
    assert handler.speculative_turns is context.speculative_turns


def test_unselected_minimax_needs_no_credentials(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["speech-to-speech", "--tts", "qwen3"])
    assert parse_arguments().tts_backend.spec is TTS_BACKENDS["qwen3"]


@pytest.mark.skipif(
    not os.getenv("MINIMAX_API_KEY") or os.getenv("MINIMAX_TTS_LIVE_TEST") != "1",
    reason="Live MiniMax calls require MINIMAX_API_KEY and explicit MINIMAX_TTS_LIVE_TEST=1",
)
@pytest.mark.parametrize("model", ["speech-2.8-hd", "speech-2.8-turbo"])
def test_live_minimax_synthesis(model):
    handler = _handler(api_key=None, model=model)
    chunks = list(handler.process(TTSInput(text="Hello from MiniMax.")))
    assert chunks
    assert all(chunk.dtype == np.int16 and chunk.shape == (512,) for chunk in chunks)
    assert np.any(np.concatenate(chunks))
    assert handler.queue_out.empty()
