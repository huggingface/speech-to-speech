from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from LLM.chat import Chat
from LLM import openai_api_language_model
from LLM.openai_api_language_model import OpenApiModelHandler, extract_stream_chunk_text


def _chunk(*, content=None, delta=True, choices=True, finish_reason=None):
    if not choices:
        return SimpleNamespace(choices=[])

    if not delta:
        choice = SimpleNamespace(finish_reason=finish_reason)
        return SimpleNamespace(choices=[choice])

    choice = SimpleNamespace(
        delta=SimpleNamespace(content=content),
        finish_reason=finish_reason,
    )
    return SimpleNamespace(choices=[choice])


def test_extract_stream_chunk_text_handles_empty_non_content_chunks():
    assert extract_stream_chunk_text(_chunk(choices=False)) == ""
    assert extract_stream_chunk_text(_chunk(delta=False, finish_reason="stop")) == ""
    assert extract_stream_chunk_text(_chunk(content=None)) == ""
    assert extract_stream_chunk_text(_chunk(content="hello")) == "hello"


def test_process_skips_empty_stream_chunks_without_breaking_sentence_chunking(monkeypatch):
    monkeypatch.setattr(
        openai_api_language_model,
        "sent_tokenize",
        lambda text: [part for part in ("Hello.", "How are you?") if part in text],
    )

    streamed_chunks = [
        _chunk(choices=False),
        _chunk(content="Hello. "),
        _chunk(delta=False, finish_reason="stop"),
        _chunk(content=None),
        _chunk(content="How are you?"),
    ]

    handler = object.__new__(OpenApiModelHandler)
    handler.model_name = "test-model"
    handler.stream = True
    handler.user_role = "user"
    handler.chat = Chat(1)
    handler.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: iter(streamed_chunks),
            )
        )
    )

    outputs = list(handler.process("Hi"))

    assert outputs == [
        ("Hello.", None, []),
        ("How are you?", None, []),
    ]
    assert handler.chat.buffer[-1] == {
        "role": "assistant",
        "content": "Hello. How are you?",
    }
