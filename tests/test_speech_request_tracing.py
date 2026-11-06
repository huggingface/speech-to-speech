from __future__ import annotations

import asyncio
import gc
import json
import re
import weakref
from dataclasses import FrozenInstanceError
from queue import Queue
from threading import Event

import httpx
import numpy as np
import pytest
from openai import OpenAI
from openai.types.realtime import RealtimeSessionCreateRequest, SessionUpdateEvent

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.LLM import base_openai_compatible_language_model as llm_module
from speech_to_speech.LLM.chat import make_user_message
from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler
from speech_to_speech.LLM.responses_api_language_model import ResponsesApiModelHandler
from speech_to_speech.pipeline.messages import (
    GenerateResponseRequest,
    Transcription,
    TranscriptionFailure,
    TTSInput,
    VADAudio,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.STT import openai_compatible_handler as stt_module
from speech_to_speech.STT.openai_compatible_handler import OpenAICompatibleSTTHandler
from speech_to_speech.TTS import openai_compatible_handler as tts_module
from speech_to_speech.TTS.openai_compatible_handler import OpenAICompatibleTTSHandler
from speech_to_speech.utils.request_tracing import REQUEST_ID_HEADER, HttpRequestContext, HttpRequestTrace
from tests.turns import reopen

TRACE_LOGGER = "speech_to_speech.utils.request_tracing"


def records(caplog, event=None):
    entries = [json.loads(record.message) for record in caplog.records if record.name == TRACE_LOGGER]
    return entries if event is None else [entry for entry in entries if entry["event"] == event]


def owned_config(session="session-synthetic"):
    config = RuntimeConfig()
    config.bind_trace_session(session)
    return config


def test_trace_headers_are_per_call_and_preserve_authentication(caplog):
    caplog.set_level("INFO", logger=TRACE_LOGGER)
    context = HttpRequestContext("llm", session_id="session-one", response_key="response-one")
    original_headers = {
        "Authorization": "Bearer provider-secret-sentinel",
        "X-S2S-API-Key": "",
        "x-speech-request-id": "caller-supplied",
    }
    options = {"extra_headers": original_headers, "temperature": 0.2}
    first, second = HttpRequestTrace(context), HttpRequestTrace(context)
    copied = first.request_options(options)

    assert first.request_id != second.request_id
    assert re.fullmatch("[0-9a-f]{32}", first.request_id)
    assert options["extra_headers"] is original_headers
    assert original_headers["x-speech-request-id"] == "caller-supplied"
    assert copied["extra_headers"] == {
        "Authorization": original_headers["Authorization"],
        "X-S2S-API-Key": "",
        REQUEST_ID_HEADER: first.request_id,
    }
    first.dispatch()
    assert records(caplog)[0]["request_id"] == first.request_id
    assert "provider-secret-sentinel" not in caplog.text
    with pytest.raises(FrozenInstanceError):
        context.session_id = "session-two"


@pytest.mark.parametrize("invalid", ["bad\nforged", "user@example.test", "é", "a" * 129, ""])
def test_invalid_identity_values_do_not_enter_logs(invalid, caplog):
    caplog.set_level("INFO", logger=TRACE_LOGGER)
    trace = HttpRequestTrace(
        HttpRequestContext("stt", session_id=invalid, turn_id=invalid, response_key=invalid, turn_revision=True)
    )
    trace.dispatch()
    entry = records(caplog)[0]
    assert entry["session_id"] is entry["turn_id"] is entry["response_key"] is entry["turn_revision"] is None
    assert len(caplog.records[-1].message) < 512


def test_runtime_trace_owner_is_not_a_public_session_field():
    config = RuntimeConfig.model_validate(
        {"trace_session_id": "spoof", "_trace_session_id": "spoof", "session": {"type": "realtime", "id": "spoof"}}
    )
    assert config.trace_session_id is None
    config.bind_trace_session("server-session")
    update = SessionUpdateEvent.model_validate(
        {"type": "session.update", "session": {"type": "realtime", "id": "spoof", "trace_session_id": "spoof"}}
    )
    config.apply_session_update(update.session)
    assert config.trace_session_id == "server-session"
    assert "trace_session_id" not in RuntimeConfig.model_fields
    assert "_trace_session_id" not in RuntimeConfig.model_fields
    assert "trace_session_id" not in config.model_dump()
    assert "_trace_session_id" not in config.model_dump()
    assert "trace_session_id" not in RealtimeSessionCreateRequest.model_json_schema()["properties"]
    with pytest.raises((AttributeError, ValueError)):
        config.trace_session_id = "spoof"
    with pytest.raises(ValueError, match="invalid server trace session ID"):
        config.bind_trace_session("bad\nforged")


def test_each_registered_connection_labels_its_own_config(caplog):
    caplog.set_level("INFO", logger=TRACE_LOGGER)
    first, second = RealtimeService(), RealtimeService()
    first_id, second_id = first.register(), second.register()
    assert first._state(first_id).runtime_config.trace_session_id == first_id
    assert second._state(second_id).runtime_config.trace_session_id == second_id
    for service, session in ((first, first_id), (second, second_id)):
        service.bind_response_latency_tracker(session, f"response-{session}", turn_id="turn_1", turn_revision=0)
        service.bind_response_latency_tracker(session, f"response-{session}", turn_id="turn_1", turn_revision=0)
    bindings = records(caplog, "speech_response_binding")
    assert len(bindings) == 2
    assert {binding["session_id"] for binding in bindings} == {first_id, second_id}
    assert {binding["turn_id"] for binding in bindings} == {"turn_1"}
    first.unregister(first_id)
    second.unregister(second_id)


def _run_stt(handler, source):
    list(handler.process(source))
    thread = handler._progressive_thread if source.mode == "progressive" else handler._final_thread
    assert thread is not None
    thread.join(5)
    assert not thread.is_alive()
    outputs = []
    while not handler.queue_out.empty():
        outputs.append(handler.queue_out.get_nowait())
    return outputs


@pytest.mark.parametrize("profile", ["transcription", "nemotron"])
@pytest.mark.parametrize("status", [200, 503])
def test_stt_http_ids_preserve_auth_and_bind_only_accepted_results(monkeypatch, caplog, profile, status):
    caplog.set_level("INFO", logger=TRACE_LOGGER)
    sent = []

    async def respond(request):
        sent.append(request)
        if status != 200:
            return httpx.Response(status, json={"error": "synthetic failure"})
        if profile == "nemotron":
            return httpx.Response(
                200, json={"choices": [{"finish_reason": "stop", "message": {"content": "synthetic"}}]}
            )
        return httpx.Response(200, json={"text": "synthetic"})

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        stt_module.httpx, "AsyncClient", lambda **kw: original_client(transport=httpx.MockTransport(respond), **kw)
    )
    monkeypatch.setattr(OpenAICompatibleSTTHandler, "warmup", lambda self: None)
    handler = OpenAICompatibleSTTHandler(
        Event(), Queue(), Queue(), setup_kwargs={"profile": profile, "api_key": "auth-secret-sentinel"}
    )
    try:
        source = VADAudio(audio=np.zeros(160), runtime_config=owned_config(), turn_id="turn_1", turn_revision=0)
        (output,) = _run_stt(handler, source)
        if status != 200:
            assert isinstance(output, TranscriptionFailure)
            assert records(caplog, "speech_http_stt_result") == []
            dispatch = records(caplog, "speech_http_dispatch")
            assert len(dispatch) == 1
            assert dispatch[0]["session_id"] == "session-synthetic"
            assert sent[0].headers[REQUEST_ID_HEADER] == dispatch[0]["request_id"]
            return
        assert isinstance(output, Transcription)
        assert not hasattr(output, "request_ids")
        assert "request_ids" not in output.model_dump()
        dispatch = records(caplog, "speech_http_dispatch")
        bindings = records(caplog, "speech_http_stt_result")
        assert len(dispatch) == len(bindings) == len(sent) == 1
        assert sent[0].headers[REQUEST_ID_HEADER] == dispatch[0]["request_id"] == bindings[0]["request_id"]
        assert sent[0].headers["Authorization"] == "Bearer auth-secret-sentinel"
        assert bindings[0]["session_id"] == "session-synthetic"
        assert "auth-secret-sentinel" not in "".join(r.message for r in caplog.records if r.name == TRACE_LOGGER)
        _run_stt(handler, source)
        assert len(records(caplog, "speech_http_stt_result")) == 1
    finally:
        handler.cleanup()


def test_window_reuse_links_actual_ids_to_final_revision_and_releases_them(monkeypatch, caplog):
    caplog.set_level("INFO", logger=TRACE_LOGGER)
    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        stt_module.httpx,
        "AsyncClient",
        lambda **kw: original_client(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"text": "synthetic"})), **kw
        ),
    )
    monkeypatch.setattr(OpenAICompatibleSTTHandler, "warmup", lambda self: None)
    turns = SpeculativeTurnTracker()
    turns.start_turn()
    handler = OpenAICompatibleSTTHandler(
        Event(), Queue(), Queue(), setup_kwargs={"max_window_s": 1, "speculative_turns": turns}
    )
    handler.window_partials_enabled = False
    config = owned_config()
    try:
        progressive = VADAudio(
            audio=np.arange(32_000, dtype=np.float32),
            runtime_config=config,
            mode="progressive",
            turn_id="turn_1",
            turn_revision=0,
        )
        assert _run_stt(handler, progressive) == []
        old_ids = {r["request_id"] for r in records(caplog, "speech_http_dispatch")}
        assert old_ids
        assert records(caplog, "speech_http_stt_result") == []
        reopen(turns, revision=0)
        final = VADAudio(
            audio=np.arange(36_800, dtype=np.float32),
            runtime_config=config,
            mode="final",
            turn_id="turn_1",
            turn_revision=1,
        )
        assert len(_run_stt(handler, final)) == 1
        bindings = records(caplog, "speech_http_stt_result")
        assert old_ids < {r["request_id"] for r in bindings}
        assert all(r["turn_revision"] == 1 and r["session_id"] == config.trace_session_id for r in bindings)
        references = [weakref.ref(result) for result in handler._windowed_turns["turn_1"].results]
        assert references
        turns.start_turn()
        next_turn = VADAudio(
            audio=np.arange(40_000, dtype=np.float32), runtime_config=config, turn_id="turn_2", turn_revision=0
        )
        _run_stt(handler, next_turn)
        gc.collect()
        assert all(reference() is None for reference in references)
        references = [weakref.ref(result) for result in handler._windowed_turns["turn_2"].results]
        assert references
        handler.on_session_end()
        gc.collect()
        assert handler._windowed_turns == {}
        assert all(reference() is None for reference in references)
    finally:
        handler.cleanup()


def test_cancelled_stt_dispatch_keeps_old_session_without_accepted_result(monkeypatch, caplog):
    caplog.set_level("INFO", logger=TRACE_LOGGER)
    entered = Event()

    async def blocked(request):
        entered.set()
        await asyncio.Event().wait()

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        stt_module.httpx, "AsyncClient", lambda **kw: original_client(transport=httpx.MockTransport(blocked), **kw)
    )
    monkeypatch.setattr(OpenAICompatibleSTTHandler, "warmup", lambda self: None)
    handler = OpenAICompatibleSTTHandler(Event(), Queue(), Queue())
    first = owned_config("old-session")
    try:
        list(handler.process(VADAudio(audio=np.zeros(160), runtime_config=first, turn_id="turn_1", turn_revision=0)))
        assert entered.wait(2)
        handler.on_session_end()
        handler._final_thread.join(2)
        assert not handler._final_thread.is_alive()
        assert records(caplog, "speech_http_stt_result") == []
        assert records(caplog, "speech_http_dispatch")[0]["session_id"] == "old-session"
        assert handler.queue_out.empty()
        assert first.trace_session_id == "old-session"
    finally:
        handler.cleanup()


@pytest.mark.parametrize("backend", ["chat", "responses", "responses-audio"])
def test_llm_calls_attach_independent_ids_without_changing_shared_client(monkeypatch, caplog, backend):
    caplog.set_level("INFO", logger=TRACE_LOGGER)
    sent = []

    def respond(request):
        sent.append(request)
        if request.url.path.endswith("/responses"):
            return httpx.Response(
                200,
                json={
                    "id": "provider-response",
                    "object": "response",
                    "created_at": 0,
                    "status": "completed",
                    "model": "synthetic",
                    "output": [
                        {
                            "id": "message-one",
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "Synthetic.", "annotations": []}],
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "provider-response",
                "object": "chat.completion",
                "created": 0,
                "model": "synthetic",
                "choices": [
                    {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Synthetic."}}
                ],
            },
        )

    client = OpenAI(
        api_key="provider-secret-sentinel",
        base_url="https://provider.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    monkeypatch.setattr(llm_module, "OpenAI", lambda **kwargs: client)
    handler_class = ChatCompletionsApiModelHandler if backend == "chat" else ResponsesApiModelHandler
    monkeypatch.setattr(handler_class, "warmup", lambda self: None)
    handler = handler_class(Event(), Queue(), Queue(), setup_kwargs={"model_name": "synthetic", "stream": False})
    try:
        headers_before = {key: value for key, value in client.default_headers.items() if isinstance(value, str)}
        for session in ("session-one", "session-two"):
            config = owned_config(session)
            config.chat.add_item(make_user_message("synthetic input"))
            request = GenerateResponseRequest(runtime_config=config, turn_id="turn_1", turn_revision=0)
            if backend == "responses-audio":
                request.audio = np.zeros(160, dtype=np.float32)
                request.audio_sample_rate = 16_000
            list(handler.process(request))
        dispatches = records(caplog, "speech_http_dispatch")
        assert len(sent) == len(dispatches) == 2
        assert len({r["request_id"] for r in dispatches}) == 2
        assert {r["session_id"] for r in dispatches} == {"session-one", "session-two"}
        assert all(req.headers[REQUEST_ID_HEADER] == record["request_id"] for req, record in zip(sent, dispatches))
        assert all(req.headers["Authorization"] == "Bearer provider-secret-sentinel" for req in sent)
        assert {key: value for key, value in client.default_headers.items() if isinstance(value, str)} == headers_before
        assert {req.url.path for req in sent} == {"/v1/responses" if backend == "responses" else "/v1/chat/completions"}
    finally:
        handler.cleanup()


def test_tts_batches_have_unique_ids_and_preserve_session_snapshot(monkeypatch, caplog):
    caplog.set_level("INFO", logger=TRACE_LOGGER)
    sent = []

    def respond(request):
        sent.append(request)
        return httpx.Response(200, content=np.zeros(512, dtype="<i2").tobytes(), headers={"content-type": "audio/pcm"})

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        tts_module.httpx, "AsyncClient", lambda **kw: original_client(transport=httpx.MockTransport(respond), **kw)
    )
    monkeypatch.setattr(OpenAICompatibleTTSHandler, "warmup", lambda self: None)
    handler = OpenAICompatibleTTSHandler(
        Event(),
        Queue(),
        Queue(),
        setup_args=(Event(),),
        setup_kwargs={"api_key": "tts-secret-sentinel", "sample_rate": 16_000},
    )
    config = owned_config()
    try:
        for text in ("Synthetic first.", "Synthetic second."):
            assert list(
                handler.process(
                    TTSInput(
                        text=text, runtime_config=config, response_key="response-one", turn_id="turn_1", turn_revision=0
                    )
                )
            )
        assert list(handler.process(TTSInput(text=" ", runtime_config=config))) == []
        dispatches = records(caplog, "speech_http_dispatch")
        assert len(sent) == len(dispatches) == 2
        assert len({r["request_id"] for r in dispatches}) == 2
        assert all(
            r["response_key"] == "response-one" and r["session_id"] == config.trace_session_id for r in dispatches
        )
        assert all(req.headers[REQUEST_ID_HEADER] == record["request_id"] for req, record in zip(sent, dispatches))
        assert all(req.headers["Authorization"] == "Bearer tts-secret-sentinel" for req in sent)
    finally:
        handler.cleanup()


def test_tts_worker_uses_the_captured_trace_even_if_operation_is_relabelled(monkeypatch, caplog):
    caplog.set_level("INFO", logger=TRACE_LOGGER)
    sent = []

    def respond(request):
        sent.append(request)
        return httpx.Response(200, content=b"\0\0", headers={"content-type": "audio/pcm"})

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        tts_module.httpx, "AsyncClient", lambda **kw: original_client(transport=httpx.MockTransport(respond), **kw)
    )
    operation = tts_module.HttpSpeechOperation(
        endpoint_url="https://provider.test/v1/audio/speech",
        api_key=None,
        payload={"response_format": "pcm"},
        timeout_s=2,
    )
    original = operation.trace = HttpRequestTrace(HttpRequestContext("tts", session_id="old-session"))
    read = operation._read_stream

    def replace_at_worker_entry(headers, results, cancel_check, trace):
        operation.trace = HttpRequestTrace(HttpRequestContext("tts", session_id="new-session"))
        read(headers, results, cancel_check, trace)

    monkeypatch.setattr(operation, "_read_stream", replace_at_worker_entry)
    assert list(operation.iter_bytes(lambda: False)) == [b"\0\0"]
    assert sent[0].headers[REQUEST_ID_HEADER] == original.request_id
    assert records(caplog, "speech_http_dispatch")[0]["request_id"] == original.request_id
    assert records(caplog, "speech_http_dispatch")[0]["session_id"] == "old-session"


def test_cancelled_before_dispatch_has_no_worker_claim(caplog):
    caplog.set_level("INFO", logger=TRACE_LOGGER)
    operation = tts_module.HttpSpeechOperation(
        endpoint_url="https://provider.test/v1/audio/speech",
        api_key=None,
        payload={"response_format": "pcm"},
        timeout_s=2,
    )
    operation.trace = HttpRequestTrace(HttpRequestContext("tts", session_id="old-session"))
    operation.cancel()
    with pytest.raises(tts_module.SpeechRequestCancelled):
        list(operation.iter_bytes(lambda: False))
    assert records(caplog, "speech_http_dispatch") == []
