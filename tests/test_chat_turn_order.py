"""Local language-model history ordering under overlapping speech.

Covers the non-interrupting overlap described in issue #454: a second
transcription can reach ``Chat`` while the first response is still being
generated, and the first response must still land in its own turn.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Optional

from openai.types.realtime import RealtimeSessionCreateRequest
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from openai.types.responses import ResponseFunctionToolCall

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.language_model import BaseLanguageModelHandler, StreamContext
from speech_to_speech.pipeline.messages import (
    AssistantTextPart,
    AssistantToolCallPart,
    GenerateResponseRequest,
    LLMResponseChunk,
)


class _OverlappingSpeechHandler(BaseLanguageModelHandler):
    """Local handler whose generation is overtaken by a newer user turn."""

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
        assert runtime_config is not None
        runtime_config.chat.add_item(make_user_message("B"))
        yield LLMResponseChunk(text="answer A", runtime_config=runtime_config, response=response)


def _make_handler() -> _OverlappingSpeechHandler:
    handler = object.__new__(_OverlappingSpeechHandler)
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.enable_lang_prompt = False
    handler.compactor = None
    handler.tokenizer = None
    return handler


def test_local_response_history_precedes_speech_that_arrived_during_generation():
    chat = Chat(10)
    chat.add_item(make_user_message("A"))
    request = GenerateResponseRequest(
        runtime_config=RuntimeConfig(
            chat=chat,
            session=RealtimeSessionCreateRequest(type="realtime", instructions="SESSION INSTRUCTIONS"),
        )
    )

    list(_make_handler().process(request))

    assert [part.text for item in chat.buffer for part in item.content if part.text] == ["A", "answer A", "B"]


def test_keyless_commit_places_text_and_tool_calls_in_their_own_turn():
    """Responses committed without a response key follow the same anchor."""
    chat = Chat(10)
    chat.add_item(make_user_message("A"))
    anchor = chat.history_anchor_id()
    chat.add_item(make_user_message("B"))

    committed = BaseLanguageModelHandler._commit_ordered_output(
        chat,
        [
            AssistantTextPart(text="answer A"),
            AssistantToolCallPart(
                tool=ResponseFunctionToolCall(
                    type="function_call",
                    call_id="call_a",
                    name="camera",
                    arguments="{}",
                )
            ),
        ],
        wants_audio=False,
        after_item_id=anchor,
    )

    assert committed
    assert [item.type for item in chat.buffer] == ["message", "message", "function_call", "message"]
