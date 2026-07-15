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

from openai.types.realtime.conversation_item import (
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemFunctionCallOutput,
    RealtimeConversationItemUserMessage,
)
from openai.types.realtime.realtime_conversation_item_user_message import Content as UserContent
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from openai.types.realtime.realtime_session_create_request import RealtimeSessionCreateRequest
from openai.types.responses import ResponseFunctionToolCall

import speech_to_speech.LLM.base_openai_compatible_language_model as base_mod
import speech_to_speech.LLM.chat_completions_language_model as ccm
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.chat_completions_language_model import (
    ChatCompletionsApiModelHandler,
    _to_chat_tool_choice,
    _to_chat_tools,
)
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
        self.last_options = None

    def with_options(self, **kwargs):
        self.last_options = kwargs
        return self


def _make_handler(stream=True):
    """Build a handler whose warmup hits the fake client (no network)."""
    orig_openai = base_mod.OpenAI
    base_mod.OpenAI = _FakeClient
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
        base_mod.OpenAI = orig_openai
    return h


def test_warmup_uses_request_scoped_sdk_retries():
    handler = _make_handler()

    assert handler.client.last_options == {"max_retries": base_mod.WARMUP_MAX_RETRIES}


def _chunk(content=None, tool_calls=None, usage=None):
    choices = []
    if content is not None or tool_calls is not None:
        choices = [SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=tool_calls), finish_reason=None)]
    return SimpleNamespace(choices=choices, usage=usage)


def _tc_delta(index, id=None, name=None, arguments=None):
    return SimpleNamespace(index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments))


def _drive(
    handler,
    *,
    tools=None,
    tool_choice=None,
    user="Hallo",
    chat=None,
    response=None,
    instructions="Du bist ein Roboter.",
):
    chat = chat or Chat(10)
    if user:
        chat.add_item(make_user_message(user))
    session = RealtimeSessionCreateRequest(type="realtime", instructions=instructions)
    if tools is not None:
        session.tools = tools
    if tool_choice is not None:
        session.tool_choice = tool_choice
    rc = RuntimeConfig(chat=chat, session=session)
    req = GenerateResponseRequest(
        runtime_config=rc, response=response, language_code="de", turn_id="t", turn_revision=0
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
    return text, tools_out, usage, chat, end


# ── Converter tests ──────────────────────────────────────────────────────────


def test_to_chat_tools_flat_to_nested():
    out = _to_chat_tools([{"type": "function", "name": "f", "description": "d", "parameters": {"type": "object"}}])
    assert out == [
        {"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}
    ]


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
    assert f("https://api.openai.com/v1/", True, "none") is None  # trailing slash still official
    assert f("http://x/v1", True, "") == {"chat_template_kwargs": {"enable_thinking": False}}  # empty effort ignored
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


def test_chat_messages_strips_tool_output_name():
    """to_transformers_chat adds a tool name for HF templates; Chat Completions
    tool messages only accept role/tool_call_id/content."""
    chat = Chat(10)
    chat.add_item(make_user_message("Search for x"))
    chat.add_item(
        RealtimeConversationItemFunctionCall(
            type="function_call",
            name="search",
            arguments='{"q": "x"}',
            call_id="call_1",
            id="fc_1",
            status="completed",
        )
    )
    chat.add_item(
        RealtimeConversationItemFunctionCallOutput(type="function_call_output", call_id="call_1", output="found")
    )

    messages = ChatCompletionsApiModelHandler._chat_messages(chat)
    tool_message = [m for m in messages if m.get("role") == "tool"][0]
    assert tool_message == {"role": "tool", "tool_call_id": "call_1", "content": "found"}


def test_chat_messages_converts_image_and_text_parts_to_chat_shape():
    """to_transformers_chat emits Realtime-shaped parts (input_text / input_image
    with a bare-string image_url); the Chat Completions API needs text / image_url
    with a nested object."""
    chat = Chat(10)
    chat.add_item(
        RealtimeConversationItemUserMessage(
            type="message",
            role="user",
            content=[
                UserContent(type="input_text", text="What is this?"),
                UserContent(type="input_image", image_url="https://example.com/img.png", detail="auto"),
            ],
        )
    )
    messages = ChatCompletionsApiModelHandler._chat_messages(chat)
    user = [m for m in messages if m.get("role") == "user"][0]
    assert isinstance(user["content"], list)
    parts = {p["type"]: p for p in user["content"]}
    assert parts["text"]["text"] == "What is this?"
    assert parts["image_url"]["image_url"] == {"url": "https://example.com/img.png", "detail": "auto"}
    # No Realtime-shaped parts leak through.
    assert all(p["type"] not in ("input_text", "input_image") for p in user["content"])


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
    text, tools, usage, chat, _end = _drive(h)
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
    text, tools, usage, chat, _end = _drive(
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


def test_tool_call_recorded_before_chunk_is_emitted():
    """Regression: a fast client can return function_call_output before the
    deferred end-of-turn write-back runs. The call must already be in history
    the instant its chunk is yielded, otherwise the output is rejected with
    'No function_call with call_id ... found' and the model re-issues the call."""
    h = _make_handler(stream=True)
    h.client.chat.completions.create = lambda **k: _FakeStream(
        [
            _chunk(content="Sure."),
            _chunk(tool_calls=[_tc_delta(0, id="srv_1", name="camera_snapshot", arguments="{}")]),
            _chunk(usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2)),
        ]
    )
    chat = Chat(10)
    chat.add_item(make_user_message("take a photo"))
    session = RealtimeSessionCreateRequest(type="realtime", instructions="Du bist ein Roboter.")
    session.tools = [{"type": "function", "name": "camera_snapshot", "parameters": {"type": "object"}}]
    rc = RuntimeConfig(chat=chat, session=session)
    req = GenerateResponseRequest(runtime_config=rc, language_code="de", turn_id="t", turn_revision=0)

    emitted_call_id = None
    for out in h.process(req):
        if isinstance(out, LLMResponseChunk) and out.tools:
            emitted_call_id = out.tools[0].call_id
            # At the moment the client receives the call, it must exist in history.
            assert emitted_call_id in chat._pending_tool_calls, (
                "function_call must be recorded BEFORE its chunk is forwarded to the client"
            )
            # A fast client returning the output here must pair cleanly (no raise).
            chat.add_item(
                RealtimeConversationItemFunctionCallOutput(
                    type="function_call_output", call_id=emitted_call_id, output="ok"
                )
            )
    assert emitted_call_id is not None, "a tool call should have been emitted"
    assert chat._has_call_id_in_buffer(emitted_call_id), "call+output should be paired in the buffer"


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
    text, tools, usage, chat, _end = _drive(
        h,
        tools=[{"type": "function", "name": "move_head", "parameters": {"type": "object"}}],
        tool_choice="required",
    )
    assert len(tools) == 1 and tools[0].name == "move_head"
    assert json.loads(tools[0].arguments) == {"direction": "right"}
    assert usage == (7, 3)


def test_streaming_refusal_is_spoken_and_stored():
    """A refusal streams as delta.refusal (content None); it must be surfaced as
    assistant text and written to history, not silently dropped."""
    h = _make_handler(stream=True)
    h.client.chat.completions.create = lambda **k: _FakeStream(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None, refusal="I cannot help with that.", tool_calls=None),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            _chunk(usage=SimpleNamespace(prompt_tokens=4, completion_tokens=6)),
        ]
    )
    text, tools, usage, chat, _end = _drive(h)
    assert "I cannot help with that." in text
    assert any(getattr(i, "role", None) == "assistant" for i in chat.buffer)


def test_non_streaming_refusal_is_spoken_and_stored():
    h = _make_handler(stream=False)
    h.client.chat.completions.create = lambda **k: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, refusal="No can do.", tool_calls=[]))],
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=2),
    )
    text, tools, usage, chat, _end = _drive(h)
    assert text == "No can do."
    assert any(getattr(i, "role", None) == "assistant" for i in chat.buffer)


def test_non_streaming_empty_choices_completes_cleanly():
    """A valid response with no choices (e.g. content filter) completes with no
    assistant text and no error, instead of raising IndexError."""
    h = _make_handler(stream=False)
    h.client.chat.completions.create = lambda **k: SimpleNamespace(
        choices=[], usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0)
    )
    text, tools, usage, chat, end = _drive(h)
    assert text == ""
    assert tools == []
    assert end is not None and end.error is None  # clean end, not a generation failure


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


# ── Text-only (output_modalities=["text"]) ────────────────────────────────────


def test_text_only_streaming_preserves_raw_deltas():
    """With output_modalities=["text"], deltas are forwarded verbatim: no
    remove_unspeechable (emoji/markdown survive) and no sentence batching."""
    h = _make_handler(stream=True)
    h.client.chat.completions.create = lambda **k: _FakeStream(
        [
            _chunk(content="# Title 🎉\n"),
            _chunk(content="- one\n- two 😀\n"),
            _chunk(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4)),
        ]
    )
    text, tools, usage, chat, end = _drive(h, response=RealtimeResponseCreateParams(output_modalities=["text"]))
    # Raw markdown layout and emoji preserved end-to-end.
    assert text == "# Title 🎉\n- one\n- two 😀\n"
    assert tools == []
    assert usage == (3, 4)
    # Raw assistant text is committed to history (not the filtered TTS string).
    assert any(getattr(i, "role", None) == "assistant" for i in chat.buffer), "assistant turn should be stored"


def test_text_only_tool_call_in_same_delta_not_dropped():
    """In text-only mode a delta can carry both content and a tool_call fragment;
    the tool_call must still be accumulated despite the verbatim-forward `continue`."""
    h = _make_handler(stream=True)
    h.client.chat.completions.create = lambda **k: _FakeStream(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="Looking it up. ",
                            tool_calls=[_tc_delta(0, id="srv_1", name="search", arguments='{"q":"x"}')],
                        ),
                        finish_reason=None,
                    )
                ],
                usage=None,
            ),
            _chunk(usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5)),
        ]
    )
    text, tools, usage, chat, _end = _drive(
        h,
        tools=[{"type": "function", "name": "search", "parameters": {"type": "object"}}],
        response=RealtimeResponseCreateParams(output_modalities=["text"]),
    )
    assert "Looking it up." in text
    assert len(tools) == 1 and tools[0].name == "search"  # not dropped by the text-only continue
    assert json.loads(tools[0].arguments) == {"q": "x"}


def test_non_streaming_text_only_preserves_symbols():
    h = _make_handler(stream=False)
    h.client.chat.completions.create = lambda **k: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="**bold** 🎉", tool_calls=[]))],
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=2),
    )
    text, tools, usage, chat, end = _drive(h, response=RealtimeResponseCreateParams(output_modalities=["text"]))
    assert text == "**bold** 🎉"  # symbols not stripped


# ── tool_choice decoupled from tools ──────────────────────────────────────────


def test_tool_choice_sent_without_tools():
    """A session-level tool_choice must reach the server even when no tools list
    is supplied (e.g. tool_choice="none" to suppress tool use)."""
    h = _make_handler(stream=True)
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeStream([_chunk(content="ok.")])

    h.client.chat.completions.create = fake_create
    _drive(h, tool_choice="none")
    assert "tools" not in captured
    assert captured["tool_choice"] == "none"


# ── Error propagation ─────────────────────────────────────────────────────────


def test_empty_input_emits_failed_end_of_response():
    """No instructions and no conversation input → terminating EndOfResponse with
    an error, instead of an opaque provider 400."""
    h = _make_handler(stream=True)
    called = {"n": 0}

    def fake_create(**kwargs):
        called["n"] += 1
        return _FakeStream([_chunk(content="should not happen")])

    h.client.chat.completions.create = fake_create
    # Empty chat + empty instructions => nothing to send.
    text, tools, usage, chat, end = _drive(h, user="", instructions="", chat=Chat(10))
    assert called["n"] == 0, "no API call should be made when there is nothing to send"
    assert end is not None and end.error is not None
    assert text == ""


def test_generation_error_emits_failed_end_of_response():
    """An exception during generation is caught and surfaced on EndOfResponse.error
    so the response is closed instead of leaving the pipeline stuck."""
    h = _make_handler(stream=True)

    def boom(**kwargs):
        raise RuntimeError("kaboom")

    h.client.chat.completions.create = boom
    text, tools, usage, chat, end = _drive(h)
    assert end is not None and end.error is not None
    assert "kaboom" in end.error


# ── Out-of-band (conversation="none") responses ───────────────────────────────


def test_out_of_band_does_not_commit_to_default_conversation():
    """Out-of-band output is emitted but never written back to the default chat."""
    h = _make_handler(stream=True)
    h.client.chat.completions.create = lambda **k: _FakeStream(
        [_chunk(content="Background note."), _chunk(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))]
    )
    chat = Chat(10)
    text, tools, usage, chat, end = _drive(
        h, chat=chat, response=RealtimeResponseCreateParams(conversation="none", output_modalities=["text"])
    )
    assert "Background note." in text
    # Default conversation keeps only the seeded user turn — no assistant commit.
    assert not any(getattr(i, "role", None) == "assistant" for i in chat.buffer)


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
