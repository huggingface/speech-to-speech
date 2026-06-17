"""Unit tests for the chat-completions LLM backend.

These run without a GPU or a live server: the OpenAI client is faked at the
module level, so the streaming/non-streaming parse logic and the format
converters are exercised purely in-process.

Run with pytest, or standalone:  python tests/test_chat_completions_backend.py
"""

from __future__ import annotations

import json
import queue
import threading
from types import SimpleNamespace

import speech_to_speech.LLM.chat_completions_language_model as ccm
from speech_to_speech.LLM.chat_completions_language_model import (
    ChatCompletionsApiModelHandler,
    _to_chat_tools,
    _to_chat_tool_choice,
)
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.pipeline.messages import (
    GenerateResponseRequest,
    LLMResponseChunk,
    TokenUsage,
)
from openai.types.realtime.realtime_session_create_request import RealtimeSessionCreateRequest
from openai.types.realtime.conversation_item import (
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemFunctionCallOutput,
)
from openai.types.responses import ResponseFunctionToolCall


# ── Fakes ────────────────────────────────────────────────────────────────────


class _FakeStream:
    """Iterable stand-in for openai.Stream; yields preset chunks."""

    def __init__(self, chunks):
        self._chunks = chunks

    def __iter__(self):
        return iter(self._chunks)

    def close(self):
        pass


# Make the handler's ``isinstance(resp, Stream)`` check recognise our fake as a
# stream. Non-streaming fakes stay plain SimpleNamespace, so they still take the
# non-stream branch.
ccm.Stream = _FakeStream


class _FakeCompletions:
    def __init__(self):
        self.next_result = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self.next_result


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self, *a, **k):
        self.chat = _FakeChat()


def _make_handler(stream=True):
    """Build a handler whose warmup hits the fake client (no network)."""
    orig_openai = ccm.OpenAI
    ccm.OpenAI = _FakeClient
    try:
        h = ChatCompletionsApiModelHandler(
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
            ),
        )
    finally:
        ccm.OpenAI = orig_openai
    return h


def _chunk(content=None, tool_calls=None, usage=None):
    choices = []
    if content is not None or tool_calls is not None:
        choices = [SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=tool_calls), finish_reason=None)]
    return SimpleNamespace(choices=choices, usage=usage)


def _tc_delta(index, id=None, name=None, arguments=None):
    return SimpleNamespace(index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments))


def _drive(handler, *, tools=None, tool_choice=None, user="Hallo", chat=None):
    chat = chat or Chat(10)
    chat.add_item(make_user_message(user))
    session = RealtimeSessionCreateRequest(type="realtime", instructions="Du bist ein Roboter.")
    if tools is not None:
        session.tools = tools
    if tool_choice is not None:
        session.tool_choice = tool_choice
    rc = RuntimeConfig(chat=chat, session=session)
    req = GenerateResponseRequest(
        runtime_config=rc, response=None, language_code="de", turn_id="t", turn_revision=0
    )
    text, tools_out, usage = "", [], None
    for out in handler.process(req):
        if isinstance(out, LLMResponseChunk):
            text += out.text
            tools_out += list(out.tools)
        elif isinstance(out, TokenUsage):
            usage = (out.input_tokens, out.output_tokens)
    return text, tools_out, usage, chat


# ── Converter tests ──────────────────────────────────────────────────────────


def test_to_chat_tools_flat_to_nested():
    out = _to_chat_tools([{"type": "function", "name": "f", "description": "d", "parameters": {"type": "object"}}])
    assert out == [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}]


def test_to_chat_tools_passthrough_and_none():
    nested = [{"type": "function", "function": {"name": "f"}}]
    assert _to_chat_tools(nested) == nested
    assert _to_chat_tools(None) is None
    assert _to_chat_tools([]) is None


def test_to_chat_tool_choice():
    assert _to_chat_tool_choice("auto") == "auto"
    assert _to_chat_tool_choice("required") == "required"
    assert _to_chat_tool_choice({"type": "function", "name": "f"}) == {"type": "function", "function": {"name": "f"}}


def test_build_extra_body_variants():
    f = ChatCompletionsApiModelHandler._build_extra_body
    assert f("http://x/v1", True, None) == {"chat_template_kwargs": {"enable_thinking": False}}
    assert f("http://x/v1", True, "none") == {"reasoning_effort": "none"}  # explicit effort wins
    assert f("https://api.openai.com/v1", True, "none") is None  # official OpenAI: no extra_body
    assert f("http://x/v1", False, None) is None
    assert f(None, True, None) is None


def test_chat_messages_encodes_tool_arguments_as_string():
    """to_transformers_chat emits arguments as a dict; the chat API needs a string."""
    chat = Chat(10)
    chat.add_item(make_user_message("Kopf links"))
    chat.add_item(
        RealtimeConversationItemFunctionCall(
            type="function_call", name="move_head", arguments='{"direction": "left"}', call_id="call_1", id="fc_1"
        )
    )
    chat.add_item(
        RealtimeConversationItemFunctionCallOutput(type="function_call_output", call_id="call_1", output="ok")
    )
    messages = ChatCompletionsApiModelHandler._chat_messages(chat)
    tool_call_msgs = [m for m in messages if m.get("tool_calls")]
    assert tool_call_msgs, "expected an assistant message carrying tool_calls"
    args = tool_call_msgs[0]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, str), f"arguments must be a JSON string, got {type(args)}"
    assert json.loads(args) == {"direction": "left"}


# ── Streaming / non-streaming parse tests ─────────────────────────────────────


def test_streaming_text_and_usage():
    h = _make_handler(stream=True)
    h.client.chat.completions.create = lambda **k: _FakeStream(
        [
            _chunk(content="Hallo. "),
            _chunk(content="Wie geht es dir?"),
            _chunk(usage=SimpleNamespace(prompt_tokens=12, completion_tokens=5)),
        ]
    )
    text, tools, usage, chat = _drive(h)
    assert "Hallo" in text and "Wie geht es dir" in text
    assert usage == (12, 5)
    assert tools == []
    # assistant text was stored back into the conversation history
    assert any(getattr(i, "role", None) == "assistant" for i in chat.buffer)


def test_streaming_tool_call_accumulates_arguments():
    h = _make_handler(stream=True)
    # Arguments arrive split across deltas, as real servers stream them.
    h.client.chat.completions.create = lambda **k: _FakeStream(
        [
            _chunk(tool_calls=[_tc_delta(0, id="srv_1", name="move_head", arguments='{"direction"')]),
            _chunk(tool_calls=[_tc_delta(0, arguments=': "left"}')]),
            _chunk(usage=SimpleNamespace(prompt_tokens=20, completion_tokens=8)),
        ]
    )
    text, tools, usage, chat = _drive(
        h,
        tools=[{"type": "function", "name": "move_head", "parameters": {"type": "object"}}],
        tool_choice="required",
    )
    assert len(tools) == 1
    tc = tools[0]
    assert isinstance(tc, ResponseFunctionToolCall)
    assert tc.name == "move_head"
    assert json.loads(tc.arguments) == {"direction": "left"}  # reassembled from two deltas
    assert usage == (20, 8)
    # the function_call was stored in history with a freshly minted call_id
    assert chat._pending_tool_calls, "tool call should be recorded in chat history"


def test_non_streaming_tool_call():
    h = _make_handler(stream=False)
    h.client.chat.completions.create = lambda **k: SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            id="srv_9",
                            function=SimpleNamespace(name="move_head", arguments='{"direction": "right"}'),
                        )
                    ],
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
    )
    text, tools, usage, chat = _drive(
        h,
        tools=[{"type": "function", "name": "move_head", "parameters": {"type": "object"}}],
        tool_choice="required",
    )
    assert len(tools) == 1 and tools[0].name == "move_head"
    assert json.loads(tools[0].arguments) == {"direction": "right"}
    assert usage == (7, 3)


def test_tools_converted_to_chat_format_on_request():
    """The request sent to the server must carry Chat-Completions-shaped tools."""
    h = _make_handler(stream=True)
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeStream([_chunk(content="ok.")])

    h.client.chat.completions.create = fake_create
    _drive(h, tools=[{"type": "function", "name": "f", "parameters": {"type": "object"}}], tool_choice="auto")
    assert captured["tools"] == [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    assert captured["tool_choice"] == "auto"
    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}


# ── Standalone runner (no pytest required) ────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
