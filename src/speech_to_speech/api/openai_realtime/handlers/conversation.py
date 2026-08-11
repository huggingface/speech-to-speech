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


class ConversationHandler(RealtimeBaseHandler):
    """Owns conversation item injection and pipeline-to-protocol translation."""

    @staticmethod
    def _is_image_message(item: ConversationItem) -> bool:
        return (
            isinstance(item, RealtimeConversationItemUserMessage)
            and bool(item.content)
            and all(part.type == "input_image" for part in item.content)
        )

    def _linked_tool_followup_image_ids(
        self,
        conn_id: str,
        items: list[ConversationItem],
    ) -> set[str] | None:
        """Validate standard item ordering and return linked image sidecars.

        Function outputs need no extension to the Realtime protocol: the demo
        gives a camera image the client item ID ``msg_tool_image_<call_id>``
        and uses that ID as the output event's standard ``previous_item_id``.
        The ID convention supplies sidecar identity; ``previous_item_id`` only
        preserves standard conversation ordering. Other images remain ordinary
        user input and are never consumed or rolled back as tool sidecars.
        """
        st = self._state(conn_id)
        linked_image_ids: set[str] = set()
        for index, item in enumerate(items):
            if isinstance(item, RealtimeConversationItemFunctionCallOutput):
                continue
            if not self._is_image_message(item) or item.id is None or index + 1 >= len(items):
                return None
            following = items[index + 1]
            if not isinstance(following, RealtimeConversationItemFunctionCallOutput):
                return None
            if item.id != f"msg_tool_image_{following.call_id}":
                return None
            if st.deferred_function_output_previous_item_ids.get(following.call_id) != item.id:
                return None
            linked_image_ids.add(item.id)
        return linked_image_ids

    def handle_conversation_item_create(
        self,
        conn_id: str,
        event: ConversationItemCreateEvent,
    ) -> list[ServerEvent]:
        """Inject a text message or function-call output into the LLM context.

        Items are added to the LLM chat context but do NOT trigger response
        generation on their own.  A subsequent ``response.create`` event is
        required to trigger the model.

        While model generation is active, items remain deferred so a fast tool
        result cannot overtake later assistant items from the same response.
        Once logical generation completes, tool outputs can be applied to the
        internal chat immediately; their wire acknowledgements remain ordered
        behind the response's still-buffered output.
        """
        st = self._state(conn_id)
        if st.in_response:
            if isinstance(event.item, RealtimeConversationItemFunctionCallOutput):
                st.deferred_function_output_previous_item_ids[event.item.call_id] = event.previous_item_id
            st.deferred_items.append(event.item)
            if (
                st.current_response_key in st.generation_done_tool_calls
                and any(isinstance(item, RealtimeConversationItemFunctionCallOutput) for item in st.deferred_items)
                and self._linked_tool_followup_image_ids(conn_id, st.deferred_items) is not None
            ):
                return self.flush_deferred_items(
                    conn_id,
                    tool_followup_inputs_only=True,
                    defer_acknowledgements=True,
                )
            logger.debug("Deferred conversation item until the active response completes")
            return []
        return self._apply_item(conn_id, event.item)

    def _apply_item(
        self,
        conn_id: str,
        item: ConversationItem,
        *,
        defer_acknowledgement: bool = False,
    ) -> list[ServerEvent]:
        """Add one item to the chat and build its ``conversation.item.created``."""
        try:
            self._append_item(conn_id, item)
        except ChatItemError as exc:
            return [self.make_error(str(exc), "invalid_conversation_item")]

        if not item:
            return []
        st = self._state(conn_id)
        if defer_acknowledgement:
            # The prefetching LM strips consumed images from Chat in place.
            # Keep the protocol echo immutable until it can be acknowledged in
            # order behind the origin response.
            st.pending_item_acks.append(item.model_copy(deep=True))
            return []
        return [self._ack_item(conn_id, item)]

    def _ack_item(self, conn_id: str, item: ConversationItem) -> ConversationItemCreatedEvent:
        """Build one ordered acknowledgement for an item already in the chat."""
        st = self._state(conn_id)
        event = ConversationItemCreatedEvent(
            type="conversation.item.created",
            event_id=self._next_event_id(),
            previous_item_id=st.last_item_id,
            item=item,
        )
        st.last_item_id = item.id
        return event

    def flush_deferred_items(
        self,
        conn_id: str,
        *,
        tool_followup_inputs_only: bool = False,
        defer_acknowledgements: bool = False,
    ) -> list[ServerEvent]:
        """Apply items buffered during a response, in arrival order.

        Called as soon as model generation has committed its history, or at
        response completion as a fallback if the side-channel event is delayed.
        """
        st = self._state(conn_id)
        if not st.deferred_items:
            return []
        has_function_output = any(
            isinstance(item, RealtimeConversationItemFunctionCallOutput) for item in st.deferred_items
        )
        linked_image_ids = self._linked_tool_followup_image_ids(conn_id, st.deferred_items)
        if tool_followup_inputs_only and (not has_function_output or linked_image_ids is None):
            return []
        items = st.deferred_items
        st.deferred_items = []
        for item in items:
            if isinstance(item, RealtimeConversationItemFunctionCallOutput):
                st.deferred_function_output_previous_item_ids.pop(item.call_id, None)
        events: list[ServerEvent] = []
        for item in items:
            events.extend(
                self._apply_item(
                    conn_id,
                    item,
                    defer_acknowledgement=defer_acknowledgements,
                )
            )
        if tool_followup_inputs_only:
            assert linked_image_ids is not None
            st.tool_followup_image_item_ids.update(linked_image_ids)
        return events

    def flush_pending_item_acks(
        self,
        conn_id: str,
        *,
        revalidate_tool_outputs: bool = False,
    ) -> list[ServerEvent]:
        """Emit acknowledgements deferred behind an active response's output."""
        st = self._state(conn_id)
        items = st.pending_item_acks
        st.pending_item_acks = []
        events: list[ServerEvent] = []
        for item in items:
            if revalidate_tool_outputs and isinstance(item, RealtimeConversationItemFunctionCallOutput):
                events.extend(self._apply_item(conn_id, item))
            elif revalidate_tool_outputs and item.id in st.tool_followup_image_item_ids:
                if item.id is not None:
                    st.runtime_config.chat.remove_user_message(item.id)
            else:
                events.append(self._ack_item(conn_id, item))
        st.tool_followup_image_item_ids.clear()
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
