"""An early response.cancel followed by an immediate retry (#724).

Runs the real router, Chat Completions handler and output processor. Only the
provider client and the TTS backend are fake, and the provider's cancelled
request is held open on an event so the ordering does not depend on timing.
"""

import logging
import threading
from queue import Empty, Queue
from threading import Event
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

import speech_to_speech.LLM.base_openai_compatible_language_model as base_mod
import speech_to_speech.LLM.chat_completions_language_model as ccm
from speech_to_speech.api.openai_realtime.websocket_router import create_app
from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler
from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
from speech_to_speech.pipeline.control import SESSION_END, is_control_message
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse
from speech_to_speech.TTS.base_tts_handler import BaseTTSHandler
from tests.openai_realtime.conftest import make_pipeline_unit

EMPTY_COMPLETION_LOG = "LLM completed without output"
CANCELLED_INSTRUCTIONS = "Count out loud from one to one hundred."
RETRY_INSTRUCTIONS = "Say ready briefly in English."
ECHO_TOOL = {
    "type": "function",
    "name": "canary_echo",
    "description": "Echo a test value.",
    "parameters": {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
}


def _chunk(*, content=None, tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)], usage=None)


class _Stream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __iter__(self):
        return iter(self._chunks)

    def close(self):
        pass


class _Provider:
    """Scripted stand-in for the OpenAI client."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.requests = []
        self.cancelled_request_started = Event()
        self.release_cancelled_request = Event()
        self.retry_chunks = [_chunk(content="Ready."), _chunk(finish_reason="stop")]

    def with_options(self, **_kwargs):
        return self

    def close(self):
        pass

    def _cancelled_stream(self):
        self.cancelled_request_started.set()
        assert self.release_cancelled_request.wait(5)
        yield _chunk(content="One, two, three.")
        yield _chunk(finish_reason="stop")

    def _create(self, **kwargs):
        if not kwargs.get("stream"):
            return SimpleNamespace(choices=[], usage=None)
        self.requests.append(kwargs)
        system_prompt = kwargs["messages"][0]["content"]
        if CANCELLED_INSTRUCTIONS in system_prompt:
            return _Stream(self._cancelled_stream())
        if RETRY_INSTRUCTIONS in system_prompt:
            return _Stream(self.retry_chunks)
        if kwargs.get("tool_choice") == "required":
            function = SimpleNamespace(name="canary_echo", arguments='{"value":"seven"}')
            call = SimpleNamespace(index=0, id="call_1", function=function)
            return _Stream([_chunk(tool_calls=[call]), _chunk(finish_reason="tool_calls")])
        return _Stream([_chunk(content="Seven came back."), _chunk(finish_reason="stop")])


class _FakeTTS(BaseTTSHandler):
    def setup(self, cancel_scope):
        self.cancel_scope = cancel_scope

    def process(self, tts_input):
        yield AUDIO_RESPONSE_DONE if isinstance(tts_input, EndOfResponse) else b"\x01\x00" * 512


def _forward_session_end(unit, stop):
    while not stop.is_set():
        try:
            item = unit.input_queue.get(timeout=0.01)
        except Empty:
            continue
        if is_control_message(item, SESSION_END.kind):
            unit.output_queue.put(item)


@pytest.fixture
def chain(monkeypatch):
    provider = _Provider()
    monkeypatch.setattr(base_mod, "OpenAI", lambda **_kwargs: provider)
    monkeypatch.setattr(ccm, "Stream", _Stream)
    stop = Event()
    unit = make_pipeline_unit()
    llm_output, tts_input = Queue(), Queue()
    llm = ChatCompletionsApiModelHandler(
        stop,
        unit.text_prompt_queue,
        llm_output,
        setup_kwargs={"base_url": "http://fake/v1", "api_key": "k", "cancel_scope": unit.cancel_scope},
    )
    processor = LMOutputProcessor(
        stop,
        queue_in=llm_output,
        queue_out=tts_input,
        setup_kwargs={"text_output_queue": unit.text_output_queue},
    )
    tts = _FakeTTS(stop, tts_input, unit.output_queue, setup_kwargs={"cancel_scope": unit.cancel_scope})
    workers = [threading.Thread(target=handler.run, daemon=True) for handler in (llm, processor, tts)] + [
        threading.Thread(target=_forward_session_end, args=(unit, stop), daemon=True)
    ]
    for worker in workers:
        worker.start()
    yield create_app(pool=[unit], stop_event=stop), provider
    stop.set()
    provider.release_cancelled_request.set()
    for worker in workers:
        worker.join(2)


def _until(ws, *event_types):
    events = []
    while not events or events[-1]["type"] not in event_types:
        events.append(ws.receive_json())
    return events


def _add_item(ws, item):
    ws.send_json({"type": "conversation.item.create", "item": item})
    _until(ws, "conversation.item.added", "conversation.item.created")


def _create_response(ws, **response):
    ws.send_json({"type": "response.create", "response": response})
    return _until(ws, "response.done")


def _complete_tool_exchange(ws):
    ws.send_json(
        {"type": "session.update", "session": {"type": "realtime", "tools": [ECHO_TOOL], "tool_choice": "auto"}}
    )
    _until(ws, "session.updated")
    _add_item(
        ws,
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Use canary_echo with the value seven."}],
        },
    )
    call = _create_response(ws, tool_choice="required")[-1]["response"]["output"][0]
    _add_item(ws, {"type": "function_call_output", "call_id": call["call_id"], "output": '{"value":"seven"}'})
    _create_response(ws)


def _cancel_early_and_retry(ws, provider):
    ws.send_json({"type": "response.create", "response": {"instructions": CANCELLED_INSTRUCTIONS}})
    _until(ws, "response.created")
    assert provider.cancelled_request_started.wait(5)
    ws.send_json({"type": "response.cancel"})
    cancelled = _until(ws, "response.done")[-1]["response"]
    ws.send_json({"type": "response.create", "response": {"instructions": RETRY_INSTRUCTIONS}})
    created = _until(ws, "response.created")
    provider.release_cancelled_request.set()
    return cancelled, created[-1]["response"]["id"], _until(ws, "response.done")


def test_retry_after_early_cancel_speaks_and_ignores_the_cancelled_request(chain, caplog):
    app, provider = chain
    with caplog.at_level(logging.WARNING), TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()
            _complete_tool_exchange(ws)
            cancelled, retry_id, retry_events = _cancel_early_and_retry(ws, provider)

    assert cancelled["status"] == "cancelled"
    retry = retry_events[-1]["response"]
    assert retry["id"] == retry_id
    assert retry["status"] == "completed"
    assert any(event["type"] == "response.output_audio.delta" for event in retry_events)
    assert [part["transcript"] for part in retry["output"][0]["content"]] == ["Ready."]

    cancelled_request, retry_request = provider.requests[-2:]
    assert CANCELLED_INSTRUCTIONS in cancelled_request["messages"][0]["content"]
    assert CANCELLED_INSTRUCTIONS not in retry_request["messages"][0]["content"]
    assert RETRY_INSTRUCTIONS in retry_request["messages"][0]["content"]
    assert retry_request["tool_choice"] == "auto"
    # The cancelled attempt must leave the retry's conversation untouched.
    assert retry_request["messages"][1:] == cancelled_request["messages"][1:]
    assert EMPTY_COMPLETION_LOG not in caplog.text


def test_provider_with_nothing_to_say_is_completed_and_logged(chain, caplog):
    app, provider = chain
    provider.retry_chunks = [_chunk(finish_reason="stop")]
    with caplog.at_level(logging.WARNING), TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()
            _complete_tool_exchange(ws)
            _, _, retry_events = _cancel_early_and_retry(ws, provider)

    retry = retry_events[-1]["response"]
    assert (retry["status"], retry["output"]) == ("completed", [])
    assert [event["type"] for event in retry_events if "output" in event["type"]] == []
    messages = [record.getMessage() for record in caplog.records if EMPTY_COMPLETION_LOG in record.getMessage()]
    assert len(messages) == 1
    assert "finish_reason=stop" in messages[0]
    assert "last_item=assistant" in messages[0]
    assert "tool_choice=auto" in messages[0]
    assert "tools=1" in messages[0]
    assert RETRY_INSTRUCTIONS not in messages[0]
