"""Tests for server-side (Keenable) tool execution in the OpenAI-compatible LLM handler.

The OpenAI client is faked with a scripted sequence of responses, and the
Keenable executor is stubbed, so the agentic loop in ``_generate`` is exercised
fully in-process: advertise tools -> intercept server-owned calls -> execute ->
write function_call/function_call_output to history -> re-request -> speak.

Run with pytest, or standalone:  python tests/test_keenable_tool_loop.py
"""

from __future__ import annotations

import json
import queue
import threading
from types import SimpleNamespace

from openai.types.realtime.conversation_item import (
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemFunctionCallOutput,
)
from openai.types.realtime.realtime_session_create_request import RealtimeSessionCreateRequest

import speech_to_speech.LLM.base_openai_compatible_language_model as base_mod
import speech_to_speech.LLM.chat_completions_language_model as ccm
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler
from speech_to_speech.pipeline.messages import (
    EndOfResponse,
    GenerateResponseRequest,
    LLMResponseChunk,
    TokenUsage,
)

# ── Fakes ────────────────────────────────────────────────────────────────────


class _FakeStream:
    """Iterable stand-in for openai.Stream; yields preset chunks."""

    def __init__(self, chunks):
        self._chunks = chunks

    def __iter__(self):
        return iter(self._chunks)

    def close(self):
        pass


# Make the handler's ``isinstance(resp, Stream)`` check recognise our fake.
ccm.Stream = _FakeStream


class _StubKeenable:
    """Stands in for KeenableWebTools: records calls, returns canned output."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.result = json.dumps({"results": [{"title": "Sunny", "url": "https://w.example"}]})

    @property
    def tool_definitions(self):
        return [
            {"type": "function", "name": "web_search", "description": "d", "parameters": {"type": "object"}},
            {"type": "function", "name": "fetch_page", "description": "d", "parameters": {"type": "object"}},
        ]

    def voice_guidance(self):
        return "GUIDANCE"

    def owns(self, name):
        return name in ("web_search", "fetch_page")

    def execute(self, name, arguments_json):
        self.calls.append((name, arguments_json))
        return self.result


class _SeqCompletions:
    """Fake chat.completions returning a scripted sequence of responses."""

    def __init__(self, results):
        self.results = list(results)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.results:
            raise AssertionError("fake client ran out of scripted responses")
        return self.results.pop(0)


class _FakeClient:
    def __init__(self, *a, **k):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kw: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]))],
                    usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
                )
            )
        )


def _text_response(text, in_tokens=10, out_tokens=5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=[]))],
        usage=SimpleNamespace(prompt_tokens=in_tokens, completion_tokens=out_tokens),
    )


def _tool_response(name, arguments, in_tokens=10, out_tokens=5):
    tc = SimpleNamespace(id="x", function=SimpleNamespace(name=name, arguments=arguments))
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tc]))],
        usage=SimpleNamespace(prompt_tokens=in_tokens, completion_tokens=out_tokens),
    )


def _stream_chunk(content=None, tool_calls=None, usage=None):
    choices = []
    if content is not None or tool_calls is not None:
        choices = [SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=tool_calls), finish_reason=None)]
    return SimpleNamespace(choices=choices, usage=usage)


def _stream_tool_response(name, arguments):
    tc = SimpleNamespace(index=0, id="x", function=SimpleNamespace(name=name, arguments=arguments))
    return _FakeStream([_stream_chunk(content="On it. ", tool_calls=None), _stream_chunk(tool_calls=[tc])])


def _stream_text_response(text):
    return _FakeStream([_stream_chunk(content=text)])


def _make_handler(max_rounds=3, stream=False):
    orig_openai = base_mod.OpenAI
    base_mod.OpenAI = _FakeClient
    try:
        handler = ChatCompletionsApiModelHandler(
            threading.Event(),
            queue.Queue(),
            queue.Queue(),
            setup_kwargs=dict(
                model_name="test-model",
                base_url="http://fake/v1",
                api_key="k",
                stream=stream,
                disable_thinking=True,
                compact_history=False,
                keenable_web_search=False,  # a stub replaces the real executor below
                tool_call_max_rounds=max_rounds,
            ),
        )
    finally:
        base_mod.OpenAI = orig_openai
    handler.server_tools = _StubKeenable()
    return handler


def _drive(handler, responses, *, tools=None, user="What's the weather in Paris?"):
    chat = Chat(10)
    chat.add_item(make_user_message(user))
    session = RealtimeSessionCreateRequest(type="realtime", instructions="Be brief.")
    if tools is not None:
        session.tools = tools
    completions = _SeqCompletions(responses)
    handler.client.chat.completions = completions
    req = GenerateResponseRequest(
        runtime_config=RuntimeConfig(chat=chat, session=session),
        response=None,
        language_code="en",
        turn_id="t",
        turn_revision=0,
    )
    text, tools_out, usage, end = "", [], None, None
    for out in handler.process(req):
        if isinstance(out, LLMResponseChunk):
            text += out.text
            tools_out += list(out.tools)
        elif isinstance(out, TokenUsage):
            usage = (out.input_tokens, out.output_tokens)
        elif isinstance(out, EndOfResponse):
            end = out
    return SimpleNamespace(text=text, tools_out=tools_out, usage=usage, chat=chat, end=end, completions=completions)


# ── Tests ────────────────────────────────────────────────────────────────────


def test_server_tool_call_is_executed_and_answer_spoken():
    handler = _make_handler()
    r = _drive(
        handler,
        [
            _tool_response("web_search", '{"query": "weather Paris"}'),
            _text_response("It is sunny in Paris."),
        ],
    )
    assert r.text == "It is sunny in Paris."
    assert r.tools_out == []  # nothing forwarded to the client
    assert r.end.error is None
    assert handler.server_tools.calls == [("web_search", '{"query": "weather Paris"}')]
    assert len(r.completions.calls) == 2


def test_tools_advertised_with_guidance():
    handler = _make_handler()
    r = _drive(handler, [_text_response("hi")])
    request_tools = r.completions.calls[0]["tools"]
    assert {t["function"]["name"] for t in request_tools} == {"web_search", "fetch_page"}
    system = r.completions.calls[0]["messages"][0]
    assert system["role"] == "system"
    assert "GUIDANCE" in str(system["content"])


def test_history_contains_call_and_output_pair():
    handler = _make_handler()
    r = _drive(
        handler,
        [
            _tool_response("web_search", '{"query": "q"}'),
            _text_response("Done."),
        ],
    )
    items = r.chat.buffer
    fcs = [i for i in items if isinstance(i, RealtimeConversationItemFunctionCall)]
    fcos = [i for i in items if isinstance(i, RealtimeConversationItemFunctionCallOutput)]
    assert len(fcs) == 1 and len(fcos) == 1
    assert fcs[0].name == "web_search"
    assert fcs[0].call_id == fcos[0].call_id
    assert fcos[0].output == handler.server_tools.result
    # The follow-up request saw the pair too.
    followup_roles = [m["role"] for m in r.completions.calls[1]["messages"]]
    assert "tool" in followup_roles


def test_final_round_forces_tool_choice_none():
    handler = _make_handler(max_rounds=3)
    r = _drive(
        handler,
        [
            _tool_response("web_search", '{"query": "a"}'),
            _tool_response("web_search", '{"query": "b"}'),
            _tool_response("web_search", '{"query": "c"}'),  # still stubborn on the last round
        ],
    )
    assert len(r.completions.calls) == 3
    assert "tool_choice" not in r.completions.calls[0]
    assert "tool_choice" not in r.completions.calls[1]
    assert r.completions.calls[2]["tool_choice"] == "none"
    # Only the first two calls were executed; the loop ends after the final round.
    assert [c[1] for c in handler.server_tools.calls] == ['{"query": "a"}', '{"query": "b"}']
    assert r.end.error is None


def test_usage_accumulates_across_rounds():
    handler = _make_handler()
    r = _drive(
        handler,
        [
            _tool_response("web_search", '{"query": "q"}', in_tokens=10, out_tokens=5),
            _text_response("Done.", in_tokens=30, out_tokens=7),
        ],
    )
    assert r.usage == (40, 12)


def test_client_tool_call_still_forwarded():
    handler = _make_handler()
    client_tool = {"type": "function", "name": "camera_snapshot", "description": "d", "parameters": {}}
    r = _drive(
        handler,
        [_tool_response("camera_snapshot", "{}")],
        tools=[client_tool],
    )
    assert [t.name for t in r.tools_out] == ["camera_snapshot"]
    assert handler.server_tools.calls == []
    assert len(r.completions.calls) == 1  # no follow-up round for client tools


def test_client_registered_name_wins_over_server_tool():
    handler = _make_handler()
    client_ws = {"type": "function", "name": "web_search", "description": "client", "parameters": {}}
    r = _drive(
        handler,
        [_tool_response("web_search", '{"query": "q"}')],
        tools=[client_ws],
    )
    # web_search advertised once (the client's), fetch_page still added by the server.
    names = [t["function"]["name"] for t in r.completions.calls[0]["tools"]]
    assert names.count("web_search") == 1
    assert "fetch_page" in names
    # The call went to the client, not the server executor.
    assert [t.name for t in r.tools_out] == ["web_search"]
    assert handler.server_tools.calls == []


def test_tool_error_output_still_answers():
    handler = _make_handler()
    handler.server_tools.result = json.dumps({"error": "Rate limit exceeded"})
    r = _drive(
        handler,
        [
            _tool_response("web_search", '{"query": "q"}'),
            _text_response("Sorry, search is unavailable."),
        ],
    )
    assert r.text == "Sorry, search is unavailable."
    fcos = [i for i in r.chat.buffer if isinstance(i, RealtimeConversationItemFunctionCallOutput)]
    assert json.loads(fcos[0].output) == {"error": "Rate limit exceeded"}
    assert r.end.error is None


def test_streaming_server_tool_loop_speaks_interim_and_final_text():
    handler = _make_handler(stream=True)
    r = _drive(
        handler,
        [
            _stream_tool_response("web_search", '{"query": "q"}'),
            _stream_text_response("It is sunny."),
        ],
    )
    # The interim acknowledgement streamed before the tool call is spoken too.
    assert "On it." in r.text and "It is sunny." in r.text
    assert r.tools_out == []  # server-owned call not forwarded to the client
    assert handler.server_tools.calls == [("web_search", '{"query": "q"}')]
    fcos = [i for i in r.chat.buffer if isinstance(i, RealtimeConversationItemFunctionCallOutput)]
    assert len(fcos) == 1
    assert r.end.error is None


def test_no_server_tools_single_round():
    handler = _make_handler()
    handler.server_tools = None
    r = _drive(handler, [_text_response("plain")])
    assert r.text == "plain"
    assert len(r.completions.calls) == 1
    assert "tools" not in r.completions.calls[0]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
