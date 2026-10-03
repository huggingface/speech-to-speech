"""Exercise provider terminal signals through the SDK, pipeline, and Realtime service."""

import json
from queue import Queue
from threading import Event

import httpx
import pytest
from openai import OpenAI
from openai.types.realtime import RealtimeSessionCreateRequest
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
from speech_to_speech.pipeline.events import PipelineEvent, ResponseGenerationDoneEvent
from speech_to_speech.pipeline.messages import EndOfResponse, LLMResponseChunk, TTSInput
from tests.openai_realtime.realtime_contract import assert_openai_schema, assert_response_lifecycle_contract
from tests.test_chat_completions_backend import _make_handler as _chat_handler
from tests.test_responses_api_language_model import _make_handler as _responses_handler
from tests.test_responses_api_language_model import _make_request

TEXT = "Water boils at one"
ERROR = "The provider could not finish the response."


def _message():
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": TEXT, "annotations": []}],
    }


def _response(status, reason=None, output=None):
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "test-model",
        "status": status,
        "output": [_message()] if output is None else output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "incomplete_details": {"reason": reason} if reason else None,
        "error": {"code": "server_error", "message": ERROR} if status == "failed" else None,
        "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
    }


def _chat_chunk(content=None, finish_reason=None, usage=None, tool_calls=None):
    return {
        "id": "chat_1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": [
            {"index": 0, "delta": {"content": content, "tool_calls": tool_calls}, "finish_reason": finish_reason}
        ]
        if usage is None
        else [],
        "usage": usage,
    }


def _provider_payload(backend, stream, signal):
    if backend == "chat":
        usage = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
        if stream:
            events = [_chat_chunk(TEXT)]
            if signal != "missing":
                events.append(_chat_chunk(finish_reason=signal))
            events.append(_chat_chunk(usage=usage))
            return events
        return {
            "id": "chat_1",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": TEXT}, "finish_reason": signal}],
            "usage": usage,
        }
    reason = signal if signal in ("max_output_tokens", "content_filter") else None
    status = "incomplete" if reason else "failed" if signal == "failed" else "completed"
    response = _response(status, reason)
    if not stream:
        return response
    events = [
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": TEXT,
        },
        {"type": "response.output_item.done", "sequence_number": 2, "output_index": 0, "item": _message()},
    ]
    if signal == "error":
        events.append({"type": "error", "sequence_number": 3, "code": "server_error", "message": ERROR, "param": None})
    elif signal != "missing":
        events.append({"type": f"response.{status}", "sequence_number": 3, "response": response})
    return events


def _client(payload, stream):
    def respond(request):
        if stream:
            body = "".join(f"data: {json.dumps(event)}\n\n" for event in payload) + "data: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
        return httpx.Response(200, json=payload)

    return OpenAI(
        api_key="test",
        base_url="http://provider.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )


def _run(backend, stream, signal, modality, payload=None):
    handler = (_chat_handler if backend == "chat" else _responses_handler)(stream=stream)
    request = _make_request()
    request.runtime_config.session = RealtimeSessionCreateRequest(type="realtime", output_modalities=[modality])
    request.response = RealtimeResponseCreateParams(output_modalities=[modality])
    side_channel = Queue()
    processor = LMOutputProcessor.__new__(LMOutputProcessor)
    processor.setup(text_output_queue=side_channel)
    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    conn = service.register()
    state = service._state(conn)
    state.runtime_config = request.runtime_config
    state.current_response_params = request.response
    state.response_pending = True
    outputs = []
    events = []
    with _client(payload if payload is not None else _provider_payload(backend, stream, signal), stream) as client:
        handler.client = client
        for output in handler.process(request):
            outputs.append(output)
            for processed in processor.process(output):
                if isinstance(processed, PipelineEvent):
                    events.extend(service.dispatch_pipeline_event(conn, processed))
                elif isinstance(processed, TTSInput):
                    # Stand in for TTS with real PCM so the status tests also
                    # cover the audio resampler and its terminal ordering.
                    events.extend(service.encode_audio_chunk(conn, b"\x00" * 512, processed.response_key))
                elif isinstance(processed, EndOfResponse):
                    events.extend(service.finish_response(conn, response_key=processed.response_key))
    logical_done = next(event for event in list(side_channel.queue) if isinstance(event, ResponseGenerationDoneEvent))
    assert not state.in_response
    assert not state.response_pending
    service.unregister(conn)
    done = [event for event in events if event.type == "response.done"]
    assert len(done) == 1
    assert_openai_schema(events)
    if done[0].response.output:
        assert_response_lifecycle_contract(events, wants_audio=modality == "audio")
    else:
        # The shared lifecycle checker requires at least one announced item;
        # limits and filters can also end a response before any output exists.
        assert not any(event.type == "response.output_item.added" for event in events)
    return done[0].response, outputs, events, request, logical_done


CASES = [
    ("chat", "length", "incomplete", "max_output_tokens"),
    ("chat", "content_filter", "incomplete", "content_filter"),
    ("chat", "stop", "completed", None),
    ("responses", "max_output_tokens", "incomplete", "max_output_tokens"),
    ("responses", "content_filter", "incomplete", "content_filter"),
    ("responses", "failed", "failed", None),
    ("responses", "completed", "completed", None),
]


@pytest.mark.parametrize("backend,signal,status,reason", CASES)
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("modality", ["text", "audio"])
def test_provider_ending_reaches_realtime(backend, signal, status, reason, stream, modality):
    response, outputs, events, request, logical_done = _run(backend, stream, signal, modality)
    assert response.status == status
    assert response.status_details.reason == reason
    assert response.usage.input_tokens == 7
    assert response.usage.output_tokens == 3
    assert logical_done.succeeded == (status == "completed")
    assert bool(response.status_details.error) == (status == "failed")
    assert bool([event for event in events if event.type == "error"]) == (status == "failed")
    if status != "failed":
        assert "".join(output.text for output in outputs if isinstance(output, LLMResponseChunk)) == TEXT
    history = request.runtime_config.chat.to_responses_api_chat()
    assert any(item.get("role") == "assistant" for item in history) == (status == "completed")
    assert any(item.get("role") == "user" for item in history)
    assert all(item.status == ("completed" if status == "completed" else "incomplete") for item in response.output)


@pytest.mark.parametrize("modality", ["text", "audio"])
def test_responses_error_event_fails(modality):
    response, outputs, events, _, logical_done = _run("responses", True, "error", modality)
    assert response.status == "failed"
    assert not logical_done.succeeded
    assert ERROR in next(output.error for output in outputs if isinstance(output, EndOfResponse))
    assert ERROR in next(event.error.message for event in events if event.type == "error")


@pytest.mark.parametrize("backend", ["chat", "responses"])
def test_missing_terminal_stays_compatible_and_warns(backend, caplog):
    response, _, _, _, _ = _run(backend, True, "missing", "text")
    assert response.status == "completed"
    assert "ended without" in caplog.text


@pytest.mark.parametrize("backend", ["chat", "responses"])
@pytest.mark.parametrize("stream", [True, False])
def test_incomplete_tool_arguments_are_not_exposed(backend, stream):
    tool = {
        "id": "fc_1",
        "type": "function_call",
        "call_id": "call_1",
        "name": "weather",
        "arguments": '{"city":',
        "status": "incomplete",
    }
    if backend == "chat":
        signal = "length"
        delta = {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "weather", "arguments": '{"city":'},
        }
        if stream:
            payload = [_chat_chunk(tool_calls=[delta]), _chat_chunk(finish_reason=signal)]
        else:
            payload = _provider_payload(backend, False, signal)
            payload["choices"][0]["message"] = {"role": "assistant", "content": None, "tool_calls": [delta]}
    else:
        signal = "max_output_tokens"
        response = _response("incomplete", signal, output=[tool])
        payload = (
            [
                {"type": "response.output_item.done", "sequence_number": 1, "output_index": 0, "item": tool},
                {"type": "response.incomplete", "sequence_number": 2, "response": response},
            ]
            if stream
            else response
        )
    response, outputs, events, request, logical_done = _run(backend, stream, signal, "text", payload)
    assert response.status == "incomplete"
    assert response.output == []
    assert not any(output.tools for output in outputs if isinstance(output, LLMResponseChunk))
    assert not any("function_call" in event.type for event in events)
    assert not request.runtime_config.chat.has_pending_tool_calls()
    assert not logical_done.succeeded


@pytest.mark.parametrize("signal", ["max_output_tokens", "failed", "error"])
def test_terminal_after_completed_tool_rolls_back_history(signal):
    tool = {
        "id": "fc_1",
        "type": "function_call",
        "call_id": "call_1",
        "name": "weather",
        "arguments": '{"city":"Paris"}',
        "status": "completed",
    }
    payload = _provider_payload("responses", True, signal)
    payload.insert(2, {"type": "response.output_item.done", "sequence_number": 2, "output_index": 1, "item": tool})
    response, _, _, request, logical_done = _run("responses", True, signal, "text", payload)
    assert response.status == ("incomplete" if signal == "max_output_tokens" else "failed")
    assert not request.runtime_config.chat.has_pending_tool_calls()
    assert not any(item.get("role") == "assistant" for item in request.runtime_config.chat.to_responses_api_chat())
    assert not logical_done.succeeded


@pytest.mark.parametrize("reason", [None, "provider_extension"])
@pytest.mark.parametrize("stream", [True, False])
def test_incomplete_without_known_reason_still_incomplete(reason, stream):
    response = _response("incomplete", reason, output=[])
    payload = [{"type": "response.incomplete", "sequence_number": 1, "response": response}] if stream else response
    done, outputs, _, _, logical_done = _run("responses", stream, "unused", "text", payload)
    assert done.status == "incomplete"
    assert done.status_details.reason is None
    assert not any(isinstance(output, LLMResponseChunk) for output in outputs)
    assert not logical_done.succeeded


@pytest.mark.parametrize("backend", ["chat", "responses"])
@pytest.mark.parametrize("stream", [True, False])
def test_direct_audio_uses_chat_completion_status(backend, stream):
    import numpy as np

    handler = (_chat_handler if backend == "chat" else _responses_handler)(stream=stream)
    request = _make_request()
    request.audio = np.zeros(1600, dtype=np.float32)
    with _client(_provider_payload("chat", stream, "length"), stream) as client:
        handler.client = client
        outputs = list(handler.process(request))
    end = next(output for output in outputs if isinstance(output, EndOfResponse))
    assert end.status == "incomplete"
    assert end.reason == "max_output_tokens"
    assert end.error is None
    assert not any(item.get("role") == "assistant" for item in request.runtime_config.chat.to_responses_api_chat())


@pytest.mark.parametrize(
    "backend,signal",
    [("chat", "length"), ("responses", "max_output_tokens"), ("responses", "failed"), ("responses", "error")],
)
def test_cancelled_generation_ignores_provider_terminal(backend, signal):
    from speech_to_speech.pipeline.cancel_scope import CancelScope

    scope = CancelScope()
    handler = (_chat_handler if backend == "chat" else _responses_handler)(stream=True)
    handler.cancel_scope = scope
    request = _make_request()
    request.response = RealtimeResponseCreateParams(output_modalities=["text"])

    class CancelBeforeTerminal(httpx.SyncByteStream):
        def __iter__(self):
            payload = _provider_payload(backend, True, signal)
            for index, event in enumerate(payload):
                if index == 1:
                    scope.cancel()
                yield f"data: {json.dumps(event)}\n\n".encode()
            yield b"data: [DONE]\n\n"

    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=CancelBeforeTerminal())
    )
    with OpenAI(
        api_key="test", base_url="http://provider.invalid/v1", http_client=httpx.Client(transport=transport)
    ) as client:
        handler.client = client
        outputs = list(handler.process(request))
    end = next(output for output in outputs if isinstance(output, EndOfResponse))
    assert end.error is None
    assert end.reason is None
    assert end.status == "completed"  # The router owns cancellation; no provider override.
    assert not any(item.get("role") == "assistant" for item in request.runtime_config.chat.to_responses_api_chat())


@pytest.mark.parametrize(
    "final_status,reason", [("cancelled", "turn_detected"), ("cancelled", "client_cancelled"), ("failed", None)]
)
def test_cancellation_and_tts_failure_take_precedence(final_status, reason):
    from speech_to_speech.pipeline.events import AssistantResponseDoneEvent, ResponseFailedEvent

    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    conn = service.register()
    service.dispatch_pipeline_event(
        conn, AssistantResponseDoneEvent(response_key="first", status="incomplete", reason="max_output_tokens")
    )
    if final_status == "failed":
        service.dispatch_pipeline_event(conn, ResponseFailedEvent(response_key="first", message="TTS failed"))
        events = service.finish_response(conn, response_key="first")
    else:
        events = service.finish_response(conn, status=final_status, reason=reason, response_key="first")
    done = next(event.response for event in events if event.type == "response.done")
    assert done.status == final_status
    assert done.status_details.reason == reason
    # Neither failure nor incomplete status may leak into the next response.
    service.dispatch_pipeline_event(conn, AssistantResponseDoneEvent(response_key="second"))
    next_done = next(
        event.response
        for event in service.finish_response(conn, response_key="second")
        if event.type == "response.done"
    )
    assert next_done.status == "completed"
    assert next_done.status_details.reason is None
    service.unregister(conn)


@pytest.mark.parametrize("status", ["failed", "incomplete"])
def test_responses_terminal_event_does_not_require_nested_status(status):
    response = _response(status, "max_output_tokens" if status == "incomplete" else None, output=[])
    response.pop("status")
    payload = [{"type": f"response.{status}", "sequence_number": 1, "response": response}]
    done, _, _, _, _ = _run("responses", True, "unused", "text", payload)
    assert done.status == status


@pytest.mark.parametrize("confirm_reopen", [True, False])
def test_incomplete_terminal_waits_for_reopen_decision(confirm_reopen):
    from speech_to_speech.pipeline.events import AssistantResponseDoneEvent
    from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

    tracker = SpeculativeTurnTracker()
    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event(), speculative_turns=tracker)
    conn = service.register()
    turn_id, revision = tracker.start_turn()
    candidate = tracker.begin_reopen_candidate(turn_id, revision)
    terminal = AssistantResponseDoneEvent(
        response_key="first",
        turn_id=turn_id,
        turn_revision=revision,
        status="incomplete",
        reason="max_output_tokens",
    )
    assert service.should_defer_pipeline_event(terminal)
    assert service.try_dispatch_pipeline_event(conn, terminal) is None
    assert not service._state(conn).in_response
    if confirm_reopen:
        assert tracker.confirm_reopen_candidate(turn_id, revision, candidate)
        assert service.try_dispatch_pipeline_event(conn, terminal) == []
        assert not service._state(conn).response_incomplete
    else:
        tracker.cancel_reopen_candidate(turn_id, candidate)
        events = service.try_dispatch_pipeline_event(conn, terminal)
        assert [event.type for event in events] == ["response.created"]
        done = next(event.response for event in service.finish_response(conn) if event.type == "response.done")
        assert done.status == "incomplete"
        assert done.status_details.reason == "max_output_tokens"
    service.unregister(conn)


@pytest.mark.parametrize("backend,signal", [("chat", "length"), ("responses", "max_output_tokens")])
def test_incomplete_prefetch_is_discarded_before_terminal(backend, signal):
    from speech_to_speech.pipeline.messages import ResponsePrefetchTransaction

    handler = (_chat_handler if backend == "chat" else _responses_handler)(stream=True)
    request = _make_request()
    request.prefetch_transaction = ResponsePrefetchTransaction()
    with _client(_provider_payload(backend, True, signal), True) as client:
        handler.client = client
        for output in handler.process(request):
            if isinstance(output, EndOfResponse):
                assert output.status == "incomplete"
                assert request.prefetch_transaction.discarded
    assert not any(item.get("role") == "assistant" for item in request.runtime_config.chat.to_responses_api_chat())


@pytest.mark.parametrize("modality", ["text", "audio"])
@pytest.mark.parametrize(
    "backend,stream,kind",
    [
        ("chat", True, "empty_stop"),
        ("chat", True, "usage_only"),
        ("chat", True, "no_events"),
        ("chat", False, "empty_stop"),
        ("chat", False, "no_choices"),
        ("responses", True, "empty_completed"),
        ("responses", True, "no_events"),
        ("responses", True, "created_only"),
        ("responses", False, "empty_completed"),
        ("responses", False, "no_status"),
    ],
)
def test_successful_empty_provider_output_completes(backend, stream, kind, modality, caplog):
    if backend == "chat":
        if kind == "no_events":
            payload = []
        elif kind == "usage_only":
            payload = [_chat_chunk(usage={"prompt_tokens": 7, "completion_tokens": 0, "total_tokens": 7})]
        elif stream:
            payload = [_chat_chunk(finish_reason="stop")]
        else:
            payload = _provider_payload("chat", False, "stop")
            payload["choices"][0]["message"]["content"] = None
            if kind == "no_choices":
                payload["choices"] = []
    else:
        empty_response = _response("completed", output=[])
        if kind == "no_status":
            empty_response.pop("status")
        if kind == "no_events":
            payload = []
        elif kind == "created_only":
            empty_response["status"] = "in_progress"
            payload = [{"type": "response.created", "sequence_number": 0, "response": empty_response}]
        elif stream:
            payload = [{"type": "response.completed", "sequence_number": 1, "response": empty_response}]
        else:
            payload = empty_response

    response, outputs, events, request, logical_done = _run(backend, stream, "unused", modality, payload)
    assert response.status == "completed"
    assert response.output == []
    assert response.status_details.error is None
    assert response.status_details.reason is None
    assert logical_done.succeeded
    assert not any(isinstance(output, LLMResponseChunk) for output in outputs)
    assert next(output for output in outputs if isinstance(output, EndOfResponse)).error is None
    assert not any(event.type == "error" or event.type.startswith("response.output_") for event in events)
    assert [event.type for event in events].count("response.created") == 1
    assert any(item.get("role") == "user" for item in request.runtime_config.chat.to_responses_api_chat())
    if stream and kind in ("no_events", "usage_only", "created_only"):
        assert "ended without" in caplog.text
    else:
        assert "ended without" not in caplog.text
