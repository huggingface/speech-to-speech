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
from openai.types.realtime.realtime_conversation_item_function_call_output import (
    RealtimeConversationItemFunctionCallOutput,
)
from openai.types.realtime.realtime_conversation_item_user_message import (
    RealtimeConversationItemUserMessage,
)

from speech_to_speech.api.openai_realtime.handlers.base import RealtimeBaseHandler
from speech_to_speech.LLM.chat import ChatItemError, add_supported_item
from speech_to_speech.pipeline.events import PartialTranscriptionEvent, TranscriptionCompletedEvent

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.service import ServerEvent

logger = logging.getLogger(__name__)
TOOL_FOLLOW_UP_RESPONSE_ID_FIELD = "s2s_tool_follow_up_for_response_id"


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

        Most items remain deferred while a response is generating. Tool results
        and their image-only inputs are safe to apply immediately: function calls
        are recorded before their events reach the client, and the LLM strips
        only images present in its serialized input snapshot. This lets a queued
        tool follow-up start before the originating response finishes TTS.
        """
        st = self._state(conn_id)
        queued_origin_key = st.queued_tool_followup_origin_response_key
        is_image_only = (
            isinstance(event.item, RealtimeConversationItemUserMessage)
            and bool(event.item.content)
            and all(part.type == "input_image" for part in event.item.content)
        )
        tool_follow_up_for = getattr(event.item, TOOL_FOLLOW_UP_RESPONSE_ID_FIELD, None)
        if (
            is_image_only
            and queued_origin_key is not None
            and tool_follow_up_for == st.queued_tool_followup_origin_response_id
            and (not st.in_response or st.current_response_key == queued_origin_key)
        ):
            events = self._apply_item(conn_id, event.item)
            if event.item.id is not None:
                st.provisional_tool_followup_item_ids.setdefault(queued_origin_key, set()).add(event.item.id)
            return events
        if st.in_response:
            if isinstance(event.item, RealtimeConversationItemFunctionCallOutput):
                return self._apply_item(conn_id, event.item)
            st.deferred_items.append(event.item)
            logger.debug("Deferred conversation item until the active response completes")
            return []
        return self._apply_item(conn_id, event.item)

    def _apply_item(self, conn_id: str, item: ConversationItem) -> list[ServerEvent]:
        """Add one item to the chat and build its ``conversation.item.created``."""
        try:
            self._append_item(conn_id, item)
        except ChatItemError as exc:
            return [self.make_error(str(exc), "invalid_conversation_item")]

        if not item:
            return []
        st = self._state(conn_id)
        event = ConversationItemCreatedEvent(
            type="conversation.item.created",
            event_id=self._next_event_id(),
            previous_item_id=st.last_item_id,
            item=item,
        )
        st.last_item_id = item.id
        return [event]

    def flush_deferred_items(self, conn_id: str) -> list[ServerEvent]:
        """Apply items buffered during a response, in arrival order.

        Called at response completion (after the generation's own write-back),
        so a ``function_call_output`` pairs with its now-recorded ``function_call``
        and an image survives the just-finished response's ``strip_images``.
        """
        st = self._state(conn_id)
        if not st.deferred_items:
            return []
        items = st.deferred_items
        st.deferred_items = []
        events: list[ServerEvent] = []
        for item in items:
            events.extend(self._apply_item(conn_id, item))
        return events

    def _append_item(self, conn_id: str, item: ConversationItem) -> None:
        """Narrow ``ConversationItem`` to ``SupportedItem`` and delegate to ``Chat.add_item``.

        Raises :class:`ChatItemError` on validation failure or unsupported type.
        """
        add_supported_item(self._state(conn_id).runtime_config.chat, item)

    # ── Pipeline event handlers ────────────────────

    def on_partial_transcription(self, conn_id: str, event: PartialTranscriptionEvent) -> list[ServerEvent]:
        """Handle partial_transcription: emit transcription delta event."""
        return [
            ConversationItemInputAudioTranscriptionDeltaEvent(
                type="conversation.item.input_audio_transcription.delta",
                event_id=self._next_event_id(),
                content_index=self._next_input_content_index(conn_id),
                item_id=self._input_item_id(conn_id),
                delta=event.delta,
            )
        ]

    def on_transcription_completed(self, conn_id: str, event: TranscriptionCompletedEvent) -> list[ServerEvent]:
        """Handle transcription_completed: accumulate duration and emit completed event."""
        st = self._state(conn_id)
        st.response_usage.audio_duration_s += st.input_audio_duration_s
        return [
            ConversationItemInputAudioTranscriptionCompletedEvent(
                type="conversation.item.input_audio_transcription.completed",
                event_id=self._next_event_id(),
                content_index=0,
                item_id=self._input_item_id(conn_id),
                transcript=event.transcript,
                usage=UsageTranscriptTextUsageDuration(
                    seconds=st.input_audio_duration_s,
                    type="duration",
                ),
            )
        ]
