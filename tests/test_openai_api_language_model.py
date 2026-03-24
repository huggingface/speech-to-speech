from pathlib import Path
from types import SimpleNamespace
from threading import Event
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from LLM.chat import Chat
from LLM.openai_api_language_model import OpenApiModelHandler


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


def _make_handler(*, disable_thinking=False, stream=True, cancel_response=None):
    handler = object.__new__(OpenApiModelHandler)
    handler.model_name = "test-model"
    handler.stream = stream
    handler.gen_kwargs = {}
    handler.disable_thinking = disable_thinking
    handler._extra_body = (
        {"chat_template_kwargs": {"enable_thinking": False}}
        if disable_thinking
        else None
    )
    handler.user_role = "user"
    handler.chat = Chat(1)
    handler.runtime_config = None
    handler.cancel_response = cancel_response
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
        ("Hello. How are you?", None, []),
        ("__END_OF_RESPONSE__", None, None),
    ]


def test_process_handles_cancellation():
    cancel = Event()
    cancel.set()

    handler = _make_handler(cancel_response=cancel)

    handler.client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kwargs: iter([_make_text_delta_event("Hello")]),
        )
    )

    outputs = list(handler.process("Hi"))

    assert outputs == [("__END_OF_RESPONSE__", None, None)]


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
