from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cancel_scope import CancelScope
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


def _make_handler(*, disable_thinking=False, stream=True, cancel_scope=None):
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
        ("__END_OF_RESPONSE__", None, None),
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


def test_serialize_responses_input_uses_typed_content_blocks():
    handler = _make_handler()

    serialized = handler._serialize_responses_input(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hola."},
        ]
    )

    assert serialized == [
        {
            "type": "message",
            "role": "system",
            "content": [{"type": "input_text", "text": "You are helpful."}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello"}],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hola."}],
        },
    ]


def test_process_second_turn_uses_typed_responses_history(monkeypatch):
    monkeypatch.setattr(
        sys.modules["LLM.openai_api_language_model"],
        "sent_tokenize",
        lambda text: [text] if text else [],
    )

    handler = _make_handler()
    captured_inputs = []

    def fake_create(**kwargs):
        captured_inputs.append(kwargs["input"])
        if len(captured_inputs) == 1:
            return iter([
                _make_text_delta_event("Hola."),
                _make_output_item_done_event(content=[SimpleNamespace(type="output_text", text="Hola.")]),
            ])
        return iter([
            _make_text_delta_event("Adiós."),
            _make_output_item_done_event(content=[SimpleNamespace(type="output_text", text="Adiós.")]),
        ])

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

    list(handler.process("Say hello in Spanish."))
    list(handler.process("Now say goodbye in Spanish."))

    assert captured_inputs[1] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Say hello in Spanish."}],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hola."}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Now say goodbye in Spanish."}],
        },
    ]
