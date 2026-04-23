from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from openai.types.realtime import (
    ConversationItem,
    ConversationItemCreatedEvent,
    ConversationItemCreateEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
)
from openai.types.realtime.conversation_item_input_audio_transcription_completed_event import (
    UsageTranscriptTextUsageDuration,
)

from speech_to_speech.api.openai_realtime.handlers.base import RealtimeBaseHandler
from speech_to_speech.pipeline.events import PartialTranscriptionEvent, TranscriptionCompletedEvent

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.service import ServerEvent

logger = logging.getLogger(__name__)


class ConversationHandler(RealtimeBaseHandler):
    """Owns conversation item injection and pipeline-to-protocol translation."""

    def handle_conversation_item_create(
        self,
        conn_id: str,
        event: ConversationItemCreateEvent,
    ) -> list[ServerEvent]:
        """Inject a text message or function-call output into the LLM context.

        Items are added to the LLM chat context but do NOT trigger response
        generation on their own.  A subsequent ``response.create`` event is
        required to trigger the model.
        """
        events: list[ServerEvent] = []
        item = event.item

        if not self._append_item(conn_id, item):
            return []

        if item:
            st = self._state(conn_id)
            events.append(
                ConversationItemCreatedEvent(
                    type="conversation.item.created",
                    event_id=self._next_event_id(),
                    previous_item_id=st.last_item_id,
                    item=item,
                )
            )
            st.last_item_id = item.id

        return events

    def _append_item(self, conn_id: str, item: ConversationItem) -> bool:
        """Add a conversation item directly to the connection's chat history.

        Returns ``True`` if the item was handled, ``False`` otherwise.
        """
        st = self._state(conn_id)

        if getattr(item, "type", None) == "message" and getattr(item, "content", None):
            role = getattr(item, "role", None)
            if not role or role not in ("user", "assistant"):
                logger.warning("Unsupported message role: %s", role)
                return False
            content_parts: list[dict] = []
            for part in item.content:  # type: ignore[union-attr]
                if (part.type == "input_text" and part.text) or (part.type == "input_image" and part.image_url):  # type: ignore[union-attr]
                    content_parts.append(part.model_dump(exclude_none=True))
                else:
                    logger.warning("Unsupported content part type: %s", part.type)
            if content_parts:
                st.runtime_config.chat.append({"role": role, "content": content_parts})
                logger.debug("Added message to chat (role=%s, %d parts)", role, len(content_parts))
                return True
            return False

        if getattr(item, "type", None) == "function_call_output" and getattr(item, "output", None):
            st.runtime_config.chat.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,  # type: ignore[union-attr]
                    "output": item.output,  # type: ignore[union-attr]
                }
            )
            logger.debug("Added function_call_output to chat (call_id=%s)", item.call_id)  # type: ignore[union-attr]
            return True

        logger.warning("Unsupported item type: %s", getattr(item, "type", None))
        return False

    # ── Pipeline event handlers ────────────────────

    def on_partial_transcription(self, conn_id: str, event: PartialTranscriptionEvent) -> list[ServerEvent]:
        """Handle partial_transcription: emit transcription delta event."""
        response = self._service.response
        return [
            ConversationItemInputAudioTranscriptionDeltaEvent(
                type="conversation.item.input_audio_transcription.delta",
                event_id=self._next_event_id(),
                content_index=response._next_content_index(conn_id),
                item_id=response._current_item_id(conn_id),
                delta=event.delta,
            )
        ]

    def on_transcription_completed(self, conn_id: str, event: TranscriptionCompletedEvent) -> list[ServerEvent]:
        """Handle transcription_completed: accumulate duration and emit completed event."""
        response = self._service.response
        st = self._state(conn_id)
        st.response_usage.audio_duration_s += st.input_audio_duration_s
        return [
            ConversationItemInputAudioTranscriptionCompletedEvent(
                type="conversation.item.input_audio_transcription.completed",
                event_id=self._next_event_id(),
                content_index=0,
                item_id=response._current_item_id(conn_id),
                transcript=event.transcript,
                usage=UsageTranscriptTextUsageDuration(
                    seconds=st.input_audio_duration_s,
                    type="duration",
                ),
            )
        ]
