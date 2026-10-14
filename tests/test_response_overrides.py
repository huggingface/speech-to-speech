from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, Optional

from openai.types.realtime import RealtimeSessionCreateRequest
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import Chat, make_system_message, make_user_message
from speech_to_speech.LLM.language_model import BaseLanguageModelHandler, StreamContext
from speech_to_speech.pipeline.messages import GenerateResponseRequest, LLMResponseChunk


class _RecordingLocalHandler(BaseLanguageModelHandler):
    def _load_model(
        self,
        model_name: str,
        device: str,
        torch_dtype: str,
        gen_kwargs: dict[str, Any],
    ) -> None:
        pass

    def _generate(
        self,
        chat: Chat,
        language_code: Optional[str],
        gen: int | None,
        ctx: StreamContext,
        runtime_config: RuntimeConfig | None = None,
        response: RealtimeResponseCreateParams | None = None,
    ) -> Iterator[LLMResponseChunk]:
        self.seen_chat = chat.copy(deep=True)
        self.seen_function_tools = list(ctx.function_tools)
        if getattr(self, "emit_text", False):
            yield LLMResponseChunk(text="A complete answer.", runtime_config=runtime_config)


def test_local_backend_preserves_explicitly_empty_response_overrides():
    handler = object.__new__(_RecordingLocalHandler)
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.enable_lang_prompt = False
    handler.compactor = None
    handler.tokenizer = SimpleNamespace(encode=lambda _text: [])
    chat = Chat(10)
    chat.add_item(make_user_message("Answer without tools."))
    session = RealtimeSessionCreateRequest(
        type="realtime",
        instructions="SESSION INSTRUCTIONS",
        tools=[{"type": "function", "name": "lookup", "parameters": {"type": "object"}}],
    )
    request = GenerateResponseRequest(
        runtime_config=RuntimeConfig(chat=chat, session=session),
        response=RealtimeResponseCreateParams(instructions="", tools=[]),
    )

    list(handler.process(request))

    assert [item.type for item in handler.seen_chat.buffer] == ["message"]
    assert handler.seen_function_tools == []


def test_local_backend_preserves_response_creation_language():
    handler = object.__new__(_RecordingLocalHandler)
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.enable_lang_prompt = False
    handler.compactor = None
    handler.tokenizer = SimpleNamespace(encode=lambda _text: [])
    handler.emit_text = True
    chat = Chat(10)
    chat.add_item(make_user_message("Please answer."))
    config = RuntimeConfig(
        chat=chat,
        session=RealtimeSessionCreateRequest(type="realtime", audio={"input": {"transcription": {"language": "es"}}}),
    )
    request = GenerateResponseRequest(runtime_config=config)
    config.session.audio.input.transcription.language = "de"

    outputs = list(handler.process(request))

    assert outputs[0].selected_language == "es"


def test_local_backend_keeps_injected_context_with_session_instructions():
    handler = object.__new__(_RecordingLocalHandler)
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.enable_lang_prompt = False
    handler.compactor = None
    handler.tokenizer = SimpleNamespace(encode=lambda _text: [])
    chat = Chat(10)
    snapshot = make_system_message("CURRENT CHAT SNAPSHOT")
    chat.add_item(snapshot)
    chat.add_item(make_user_message("What was my last question?"))
    config = RuntimeConfig(
        chat=chat,
        session=RealtimeSessionCreateRequest(type="realtime", instructions="SESSION INSTRUCTIONS"),
    )

    for _ in range(2):
        list(handler.process(GenerateResponseRequest(runtime_config=config)))
        prompt = handler.seen_chat.to_transformers_chat()[0]["content"]
        assert "SESSION INSTRUCTIONS" in prompt
        assert prompt.count("CURRENT CHAT SNAPSHOT") == 1
        assert chat.init_chat_message is snapshot
        assert snapshot.content[0].text == "CURRENT CHAT SNAPSHOT"
