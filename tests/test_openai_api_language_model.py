from pathlib import Path
from types import SimpleNamespace
import sys

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cancel_scope import CancelScope
from LLM.chat import Chat
from LLM.openai_api_language_model import OpenApiModelHandler
from pipeline_messages import MessageTag


def _make_text_delta_event(text):
    return SimpleNamespace(type="response.output_text.delta", delta=text)


def _make_output_item_done_event(role="assistant", content="Hello.", item_type="message"):
    if item_type == "function_call":
        return SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(
                type="function_call",
                model_dump=lambda: {"type": "function_call", "name": "test_fn"},
            ),
        )
    return SimpleNamespace(
        type="response.output_item.done",
        item=SimpleNamespace(
            type="message",
            role=role,
            content=content,
        ),
    )


def _make_handler(*, disable_thinking=False, stream=True, cancel_scope=None):
    handler = object.__new__(OpenApiModelHandler)
    handler.model_name = "test-model"
    handler.stream = stream
    handler.gen_kwargs = {}
    handler.request_timeout_s = 20.0
    handler.request_timeout = 20.0
    handler.disable_thinking = disable_thinking
    handler._extra_body = (
        {"chat_template_kwargs": {"enable_thinking": False}}
        if disable_thinking
        else None
    )
    handler.user_role = "user"
    handler.chat = Chat(1)
    handler.runtime_config = None
    handler.cancel_scope = cancel_scope
    handler.tools = None
    handler.tools_choice = None
    return handler


def test_process_streams_text_from_response_events():
    handler = _make_handler()

    streamed_events = [
        _make_text_delta_event("Hello. "),
        _make_text_delta_event("How are you?"),
        _make_output_item_done_event(content="Hello. How are you?"),
    ]

    handler.client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kwargs: iter(streamed_events),
        )
    )

    outputs = list(handler.process("Hi"))

    assert outputs == [
        ("Hello.", None, []),
        ("How are you?", None, []),
        (MessageTag.END_OF_RESPONSE, None, None),
    ]


def test_process_handles_cancellation():
    scope = CancelScope()
    handler = _make_handler(cancel_scope=scope)

    def fake_create(**kwargs):
        scope.cancel()
        return iter([_make_text_delta_event("Hello")])

    handler.client = SimpleNamespace(
        responses=SimpleNamespace(create=fake_create)
    )

    outputs = list(handler.process("Hi"))

    assert outputs == [(MessageTag.END_OF_RESPONSE, None, None)]


def test_process_read_timeout_ends_response_cleanly():
    handler = _make_handler()

    class TimeoutStream:
        def __iter__(self):
            raise httpx.ReadTimeout("timed out")

        def close(self):
            return None

    handler.client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **kwargs: TimeoutStream())
    )

    outputs = list(handler.process("Hi"))

    assert outputs == [
        ("Wow I'm a bit slow today, could you repeat that?", None, None),
        (MessageTag.END_OF_RESPONSE, None, None),
    ]


def test_partial_transcription_updates_are_ignored():
    handler = _make_handler(stream=False)

    handler.client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("LLM should not be called for partial transcription updates")
            )
        )
    )

    outputs = list(handler.process((MessageTag.PARTIAL, "hey, who are")))

    assert outputs == []
    assert handler.chat.to_list() == []


def test_disable_thinking_passes_extra_body():
    handler = _make_handler(disable_thinking=True)
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return iter([
            _make_text_delta_event("Ok"),
            _make_output_item_done_event(content="Ok"),
        ])

    handler.client = SimpleNamespace(
        responses=SimpleNamespace(create=fake_create)
    )

    list(handler.process("Hi"))

    assert captured["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_no_disable_thinking_omits_extra_body():
    handler = _make_handler(disable_thinking=False)
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return iter([
            _make_text_delta_event("Ok"),
            _make_output_item_done_event(content="Ok"),
        ])

    handler.client = SimpleNamespace(
        responses=SimpleNamespace(create=fake_create)
    )

    list(handler.process("Hi"))

    assert captured.get("extra_body") is None


def test_second_turn_flattens_assistant_history_for_responses():
    handler = _make_handler(stream=False)
    captured = {}

    first_response = SimpleNamespace(
        usage=None,
        output=[
            SimpleNamespace(
                type="message",
                role="assistant",
                content=[SimpleNamespace(type="output_text", text="Hello.")],
            )
        ],
    )
    second_response = SimpleNamespace(usage=None, output=[])
    call_count = 0

    def fake_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return first_response
        captured.update(kwargs)
        return second_response

    handler.client = SimpleNamespace(
        responses=SimpleNamespace(create=fake_create)
    )

    list(handler.process("Hi"))
    list(handler.process("Again"))

    assistant_items = [
        item for item in captured["input"]
        if item.get("role") == "assistant"
    ]
    assert assistant_items == [
        {
            "type": "message",
            "role": "assistant",
            "content": [SimpleNamespace(type="output_text", text="Hello.")],
        }
    ]
