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

import httpx
import numpy as np
import pytest
from openai import APIConnectionError, InternalServerError, RateLimitError
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
from speech_to_speech.LLM.chat import Chat, make_user_audio_message, make_user_message
from speech_to_speech.LLM.chat_completions_language_model import (
    ChatCompletionsApiModelHandler,
    _to_chat_tool_choice,
    _to_chat_tools,
)
from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import AssistantOutputEvent, ResponseFailedEvent
from speech_to_speech.pipeline.messages import (
    EndOfResponse,
    GenerateResponseRequest,
    LLMResponseChunk,
    ResponsePrefetchTransaction,
    TokenUsage,
    TTSInput,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

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


def _make_handler(stream=True, *, base_url="http://fake/v1", reasoning_effort=None):
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
                base_url=base_url,
                api_key="k",
                stream=stream,
                disable_thinking=True,
                reasoning_effort=reasoning_effort,
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
    assert f("https://api.openai.com/v1", True, "none") == {"reasoning_effort": "none"}
    assert f("https://api.openai.com/v1/", True, "none") == {"reasoning_effort": "none"}
    assert f(None, True, "none") == {"reasoning_effort": "none"}
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


def test_chat_messages_converts_audio_to_llama_cpp_shape():
    chat = Chat(10)
    chat.add_item(make_user_audio_message("abc123"))

    messages = ChatCompletionsApiModelHandler._chat_messages(chat)

    assert messages == [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": "abc123",
                        "format": "wav",
                    },
                }
            ],
        }
    ]


def test_chat_messages_converts_audio_to_base64_data_url_shape():
    chat = Chat(10)
    chat.add_item(make_user_audio_message("abc123"))

    messages = ChatCompletionsApiModelHandler._chat_messages(chat, audio_content_type="audio_url")

    assert messages == [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": "data:audio/wav;base64,abc123",
                    },
                }
            ],
        }
    ]


# ── Streaming / non-streaming parse tests ─────────────────────────────────────


def test_chat_completions_backend_processes_audio_without_responses_api():
    handler = _make_handler(stream=False)
    handler.client.chat.completions.next_result = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="I heard you.", refusal=None, tool_calls=[]),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4),
    )
    cfg = RuntimeConfig(
        chat=Chat(5),
        session=RealtimeSessionCreateRequest(type="realtime", instructions="You are helpful."),
    )

    outputs = list(
        handler.process(
            GenerateResponseRequest(
                runtime_config=cfg,
                audio=np.zeros(1600, dtype=np.float32),
                audio_sample_rate=16000,
            )
        )
    )

    captured = handler.client.chat.completions.last_kwargs
    assert captured["messages"][-1]["content"][0]["type"] == "input_audio"
    assert captured["max_tokens"] == 256
    assert captured["temperature"] == 0.0
    assert any(isinstance(output, LLMResponseChunk) and output.text == "I heard you." for output in outputs)
    assert any(isinstance(output, TokenUsage) for output in outputs)
    assert cfg.chat.buffer[0].content[0].type == "input_audio"


def test_chat_completions_backend_uses_configured_audio_url_payload():
    handler = _make_handler(stream=False)
    handler.audio_content_type = "audio_url"
    cfg = RuntimeConfig(
        chat=Chat(5),
        session=RealtimeSessionCreateRequest(type="realtime", instructions="You are helpful."),
    )

    list(
        handler.process(
            GenerateResponseRequest(
                runtime_config=cfg,
                audio=np.zeros(1600, dtype=np.float32),
                audio_sample_rate=16000,
            )
        )
    )

    audio_part = handler.client.chat.completions.last_kwargs["messages"][-1]["content"][0]
    assert audio_part["type"] == "audio_url"
    assert audio_part["audio_url"]["url"].startswith("data:audio/wav;base64,UklGR")


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


def test_streaming_preserves_text_tool_text_order():
    h = _make_handler(stream=True)
    h.client.chat.completions.create = lambda **kwargs: _FakeStream(
        [
            _chunk(content="Before."),
            _chunk(tool_calls=[_tc_delta(0, id="srv_1", name="lookup", arguments='{"q":')]),
            _chunk(content="After.", tool_calls=[_tc_delta(0, arguments='"x"}')]),
        ]
    )
    chat = Chat(10)
    chat.add_item(make_user_message("go"))
    session = RealtimeSessionCreateRequest(type="realtime", instructions="Use tools.")
    session.tools = [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}]
    request = GenerateResponseRequest(runtime_config=RuntimeConfig(chat=chat, session=session))

    outputs = list(h.process(request))

    output_parts = [part.type for output in outputs if isinstance(output, LLMResponseChunk) for part in output.parts]
    assert output_parts == ["text", "tool_call", "text"]
    call = next(item for item in chat.buffer if isinstance(item, RealtimeConversationItemFunctionCall))
    assert json.loads(call.arguments) == {"q": "x"}
    chat.add_item(
        RealtimeConversationItemFunctionCallOutput(
            type="function_call_output",
            call_id=call.call_id,
            output="done",
        )
    )
    assert [item.type for item in chat.buffer] == [
        "message",
        "message",
        "function_call",
        "message",
        "function_call_output",
    ]
    assert [message["role"] for message in h._serialize(chat)] == [
        "user",
        "assistant",
        "assistant",
        "tool",
        "assistant",
    ]


def test_prefetch_defers_irreversible_chat_cleanup_until_claim():
    handler = _make_handler(stream=False)
    chat = Chat(1)
    old_message = chat.add_item(make_user_message("old turn"))
    image_message = chat.add_item(
        RealtimeConversationItemUserMessage(
            type="message",
            role="user",
            content=[UserContent(type="input_image", image_url="data:image/jpeg;base64,abc")],
        )
    )
    session = RealtimeSessionCreateRequest(type="realtime", instructions="Describe the image.")
    transaction = ResponsePrefetchTransaction()
    request = GenerateResponseRequest(
        runtime_config=RuntimeConfig(chat=chat, session=session),
        prefetch_transaction=transaction,
    )

    list(handler.process(request))

    assert any(item.id == old_message.id for item in chat.buffer)
    live_image = next(item for item in chat.buffer if item.id == image_message.id)
    assert live_image.content[0].type == "input_image"

    transaction.claim()

    assert not any(item.id == old_message.id for item in chat.buffer)
    live_image = next(item for item in chat.buffer if item.id == image_message.id)
    assert live_image.content == []


def test_prefetch_cleanup_failure_restores_consumed_image_and_history(monkeypatch):
    handler = _make_handler(stream=False)
    chat = Chat(1)
    old_message = chat.add_item(make_user_message("old turn"))
    image_message = chat.add_item(
        RealtimeConversationItemUserMessage(
            type="message",
            role="user",
            content=[UserContent(type="input_image", image_url="data:image/jpeg;base64,abc")],
        )
    )
    session = RealtimeSessionCreateRequest(type="realtime", instructions="Describe the image.")
    transaction = ResponsePrefetchTransaction()
    request = GenerateResponseRequest(
        runtime_config=RuntimeConfig(chat=chat, session=session),
        prefetch_transaction=transaction,
    )
    list(handler.process(request))

    def fail_after_image_strip(compactor=None):
        live_image = next(item for item in chat.buffer if item.id == image_message.id)
        assert live_image.content == []
        raise RuntimeError("trim failed")

    monkeypatch.setattr(chat, "trim_if_needed", fail_after_image_strip)

    with pytest.raises(RuntimeError, match="trim failed"):
        transaction.claim()

    assert any(item.id == old_message.id for item in chat.buffer)
    live_image = next(item for item in chat.buffer if item.id == image_message.id)
    assert live_image.content[0].type == "input_image"


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
    assert req.response_key in chat._provisional_generations
    chat.finalize_provisional_generation(req.response_key)
    assert chat._provisional_generations == {}


def test_cancelled_text_tool_turn_rolls_back_ordered_call():
    h = _make_handler(stream=True)
    scope = CancelScope()
    h.cancel_scope = scope
    h.client.chat.completions.create = lambda **kwargs: _FakeStream(
        [_chunk(tool_calls=[_tc_delta(0, id="srv_1", name="camera_snapshot", arguments="{}")])]
    )
    chat = Chat(10)
    user = chat.add_item(make_user_message("take a photo"))
    session = RealtimeSessionCreateRequest(type="realtime", instructions="Use tools.")
    session.tools = [{"type": "function", "name": "camera_snapshot", "parameters": {"type": "object"}}]
    request = GenerateResponseRequest(
        runtime_config=RuntimeConfig(chat=chat, session=session),
        turn_id="t",
        turn_revision=0,
    )
    generation = h.process(request)

    while True:
        output = next(generation)
        if isinstance(output, LLMResponseChunk) and output.tools:
            break
    assert chat.has_pending_tool_calls()
    assert request.response_key in chat._provisional_generations

    scope.cancel()
    remaining = list(generation)

    assert any(isinstance(output, EndOfResponse) for output in remaining)
    assert chat.buffer == [user]
    assert not chat.has_pending_tool_calls()
    assert chat._provisional_generations == {}


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


@pytest.mark.parametrize("base_url", [None, "https://api.openai.com/v1"])
def test_official_openai_request_includes_configured_reasoning_effort_with_tools(base_url):
    h = _make_handler(stream=True, base_url=base_url, reasoning_effort="none")
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeStream([_chunk(content="ok.")])

    h.client.chat.completions.create = fake_create
    _drive(h, tools=[{"type": "function", "name": "f", "parameters": {"type": "object"}}])

    assert captured["extra_body"] == {"reasoning_effort": "none"}
    assert captured["tools"] == [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]


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
    assert text == base_mod.PROVIDER_FAILURE_FALLBACK


def _api_error_response(status_code):
    request = httpx.Request("POST", "https://provider.example/v1/chat/completions")
    return httpx.Response(status_code, request=request)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (APIConnectionError(request=httpx.Request("POST", "https://provider.example")), "Connection error"),
        (RateLimitError("rate limited", response=_api_error_response(429), body=None), "rate limited"),
        (InternalServerError("server failed", response=_api_error_response(500), body=None), "server failed"),
    ],
)
def test_provider_failure_before_output_speaks_fallback_then_fails_without_history(error, message):
    h = _make_handler(stream=True)

    def fail(**kwargs):
        raise error

    h.client.chat.completions.create = fail
    chat = Chat(10)
    chat.add_item(make_user_message("Hallo"))
    runtime_config = RuntimeConfig(
        chat=chat,
        session=RealtimeSessionCreateRequest(type="realtime", instructions="Du bist ein Roboter."),
    )
    request = GenerateResponseRequest(
        runtime_config=runtime_config,
        language_code="de",
        turn_id="t",
        turn_revision=0,
    )

    outputs = list(h.process(request))

    assert [type(output) for output in outputs] == [LLMResponseChunk, EndOfResponse]
    assert outputs[0].text == base_mod.PROVIDER_FAILURE_FALLBACK
    assert outputs[0].language_code is None
    assert outputs[1].error is not None and message in outputs[1].error
    assert [getattr(item, "role", None) for item in chat.buffer] == ["user"]

    processor = LMOutputProcessor.__new__(LMOutputProcessor)
    processor.setup()
    pipeline_outputs = [processed for output in outputs for processed in processor.process(output)]
    assert [type(output) for output in pipeline_outputs] == [
        AssistantOutputEvent,
        TTSInput,
        ResponseFailedEvent,
        EndOfResponse,
    ]


def test_cancelled_provider_failure_does_not_emit_fallback():
    scope = CancelScope()
    h = _make_handler(stream=True)
    h.cancel_scope = scope

    def fail(**kwargs):
        scope.cancel()
        raise RuntimeError("cancelled request failed")

    h.client.chat.completions.create = fail
    text, tools, usage, chat, end = _drive(h)

    assert text == ""
    assert end is not None and "cancelled request failed" in end.error
    assert not any(getattr(item, "role", None) == "assistant" for item in chat.buffer)


def test_stale_provider_failure_does_not_emit_fallback():
    tracker = SpeculativeTurnTracker()
    tracker.observe("t", 0)
    h = _make_handler(stream=True)
    h.speculative_turns = tracker

    def fail(**kwargs):
        tracker.observe("t", 1)
        raise RuntimeError("stale request failed")

    h.client.chat.completions.create = fail
    text, tools, usage, chat, end = _drive(h)

    assert text == ""
    assert end is not None and "stale request failed" in end.error
    assert not any(getattr(item, "role", None) == "assistant" for item in chat.buffer)


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
