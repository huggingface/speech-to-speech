"""Local language-model history ordering under overlapping speech.

Covers the non-interrupting overlap described in issue #454: a second
transcription can reach ``Chat`` while the first response is still being
generated, and the first response must still land in its own turn.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Optional

import pytest
from openai.types.realtime import RealtimeSessionCreateRequest
from openai.types.realtime.realtime_conversation_item_user_message import Content as UserContent
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from openai.types.responses import ResponseFunctionToolCall

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.LLM.language_model import BaseLanguageModelHandler, StreamContext
from speech_to_speech.pipeline.events import AssistantOutputEvent
from speech_to_speech.pipeline.history import ResponseHistory
from speech_to_speech.pipeline.messages import (
    AssistantTextPart,
    AssistantToolCallPart,
    GenerateResponseRequest,
    LLMResponseChunk,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from tests.llm_history import drive_llm


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
        runtime_config.chat.strip_images()
        assert chat.image_message_ids(), "Live cleanup must not alter the model's input snapshot"
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
    user = make_user_message("A")
    user.content.append(UserContent(type="input_image", image_url="data:image/jpeg;base64,abc"))
    chat.add_item(user)
    request = GenerateResponseRequest(
        runtime_config=RuntimeConfig(
            chat=chat,
            session=RealtimeSessionCreateRequest(type="realtime", instructions="SESSION INSTRUCTIONS"),
        )
    )

    list(drive_llm(_make_handler(), request))

    assert [part.text for item in chat.buffer for part in item.content if part.text] == ["A", "answer A", "B"]


@pytest.mark.parametrize("outcome", ["superseded", "completed", "cancelled"])
def test_history_acceptance_uses_global_order_and_keeps_committed_work(outcome):
    """A proposal must pass the same turn gate as its output; accepted work survives."""
    tracker = SpeculativeTurnTracker()
    service = RealtimeService(speculative_turns=tracker)
    conn_id = service.register()
    chat = service._state(conn_id).runtime_config.chat
    chat.add_item(make_user_message("A"))
    turn_id, revision = tracker.start_turn()
    anchor = chat.history_anchor_id()
    parts = [
        AssistantTextPart(text="answer A"),
        AssistantToolCallPart(
            tool=ResponseFunctionToolCall(type="function_call", call_id="call_a", name="camera", arguments="{}")
        ),
    ]
    proposal = ResponseHistory.capture(
        chat,
        BaseLanguageModelHandler._ordered_output_items(parts, wants_audio=False),
        after_item_id=anchor,
        complete=True,
    )
    event = AssistantOutputEvent(
        parts=parts,
        history=proposal,
        response_key="response_a",
        turn_id=turn_id,
        turn_revision=revision,
    )
    # Merely producing a proposal leaves shared history untouched.
    assert [item.type for item in chat.buffer] == ["message"]
    if outcome != "superseded":
        assert service.dispatch_pipeline_event(conn_id, event)
    tracker.start_turn()
    chat.add_item(make_user_message("B"))
    if outcome == "superseded":
        assert service.dispatch_pipeline_event(conn_id, event) == []
    else:
        # Replayed side/ordered copies do not duplicate the history proposal.
        service.dispatch_pipeline_event(conn_id, event)
        service.finish_response(
            conn_id, "completed" if outcome == "completed" else "cancelled", response_key="response_a"
        )
        # A late copy of a closed response cannot write its history again.
        service.dispatch_pipeline_event(conn_id, event)
    if outcome == "completed":
        assert [item.type for item in chat.buffer] == ["message", "message", "function_call", "message"]
    else:
        assert [item.type for item in chat.buffer] == ["message", "message"]
    service.unregister(conn_id)
