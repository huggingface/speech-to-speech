from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unicodedata import category

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


def _transcript_words(transcript: str) -> list[tuple[str, str]]:
    """Return display and comparison forms for whitespace-delimited words."""
    words: list[tuple[str, str]] = []
    for token in transcript.split():
        start = 0
        end = len(token)
        while start < end and category(token[start]).startswith("P"):
            start += 1
        while end > start and category(token[end - 1]).startswith("P"):
            end -= 1
        display = token[start:end]
        if display:
            words.append((display, display.casefold()))
    return words


def _stable_transcript_words(previous: str, current: str) -> list[tuple[str, str]]:
    """Find words confirmed by two hypotheses, excluding their unstable tail."""
    previous_words = _transcript_words(previous)
    current_words = _transcript_words(current)
    common_count = 0
    for previous_word, current_word in zip(previous_words, current_words):
        if previous_word[1] != current_word[1]:
            break
        common_count += 1

    # The last matching word was at the speculative edge of the previous
    # hypothesis. Hold it back until another update adds context after it.
    return current_words[: max(0, common_count - 1)]


class ConversationHandler(RealtimeBaseHandler):
    """Owns conversation item injection and pipeline-to-protocol translation."""

    @staticmethod
    def _is_image_message(item: ConversationItem) -> bool:
        return (
            isinstance(item, RealtimeConversationItemUserMessage)
            and bool(item.content)
            and all(part.type == "input_image" for part in item.content)
        )

    def _tool_followup_inputs_are_ordered(
        self,
        conn_id: str,
        items: list[ConversationItem],
    ) -> bool:
        """Return whether deferred items form a prefetch-safe tool batch.

        Image items are allowed immediately before a function output when
        ``previous_item_id`` confirms that insertion order. The field does not
        imply ownership, so every image remains an ordinary conversation item
        and must survive if the tool response is later rolled back.
        """
        st = self._state(conn_id)
        for index, item in enumerate(items):
            if isinstance(item, RealtimeConversationItemFunctionCallOutput):
                continue
            if not self._is_image_message(item) or item.id is None or index + 1 >= len(items):
                return False
            following = items[index + 1]
            if not isinstance(following, RealtimeConversationItemFunctionCallOutput):
                return False
            if st.deferred_function_output_previous_item_ids.get(following.call_id) != item.id:
                return False
        return True

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
                and self._tool_followup_inputs_are_ordered(conn_id, st.deferred_items)
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
        inputs_are_ordered = self._tool_followup_inputs_are_ordered(conn_id, st.deferred_items)
        if tool_followup_inputs_only and (not has_function_output or not inputs_are_ordered):
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
            else:
                events.append(self._ack_item(conn_id, item))
        return events

    def _append_item(self, conn_id: str, item: ConversationItem) -> None:
        """Narrow ``ConversationItem`` to ``SupportedItem`` and delegate to ``Chat.add_item``.

        Raises :class:`ChatItemError` on validation failure or unsupported type.
        """
        add_supported_item(self._state(conn_id).runtime_config.chat, item)

    # ── Pipeline event handlers ────────────────────

    def on_partial_transcription(self, conn_id: str, event: PartialTranscriptionEvent) -> list[ServerEvent]:
        """Stabilize a cumulative STT hypothesis into an append-only Realtime delta."""
        st = self._state(conn_id)
        item_id = self._input_item_id(conn_id, event.turn_id, event.turn_revision)
        if item_id is None:
            logger.debug(
                "Ignoring partial transcription for unknown turn=%s rev=%s",
                event.turn_id,
                event.turn_revision,
            )
            return []
        input_item = st.input_items.get(item_id)
        if input_item is None:
            logger.debug("Ignoring partial transcription for released item=%s", item_id)
            return []
        hypothesis = event.delta.strip()
        if not hypothesis or hypothesis == input_item.latest_transcript:
            return []

        previous = input_item.latest_transcript
        input_item.latest_transcript = hypothesis
        if not previous:
            return []

        stable_words = _stable_transcript_words(previous, hypothesis)
        emitted_words = _transcript_words(input_item.transcript_prefix)
        emitted_comparison = [word[1] for word in emitted_words]
        stable_comparison = [word[1] for word in stable_words]
        if stable_comparison[: len(emitted_comparison)] != emitted_comparison:
            # A word that already reached the wire was later revised. Realtime
            # has no retraction event, so wait for a future hypothesis that
            # extends the committed prefix or for the authoritative completion.
            logger.debug(
                "Withholding revised stable transcription for item=%s (emitted=%r, hypothesis=%r)",
                item_id,
                input_item.transcript_prefix[-40:],
                hypothesis[-40:],
            )
            return []

        new_words = stable_words[len(emitted_words) :]
        if not new_words:
            return []

        delta = (" " if emitted_words else "") + " ".join(word[0] for word in new_words)
        input_item.transcript_prefix += delta
        return [
            ConversationItemInputAudioTranscriptionDeltaEvent(
                type="conversation.item.input_audio_transcription.delta",
                event_id=self._next_event_id(),
                content_index=0,
                item_id=item_id,
                delta=delta,
            )
        ]

    def terminalize_input_item(
        self,
        conn_id: str,
        item_id: str,
    ) -> tuple[str, float] | None:
        """Release one input item's active transcript and routing state."""
        st = self._state(conn_id)
        input_item = st.input_items.pop(item_id, None)
        if input_item is None:
            logger.debug("Ignoring input terminal for released item=%s", item_id)
            return None
        duration_s = input_item.audio_duration_s
        st.input_item_by_turn_revision = {
            turn: tracked_item_id
            for turn, tracked_item_id in st.input_item_by_turn_revision.items()
            if tracked_item_id != item_id
        }
        if st.current_input_item_id == item_id:
            st.current_input_item_id = None
        return item_id, duration_s

    def _completion_input_item_id(
        self,
        conn_id: str,
        turn_id: str | None,
        turn_revision: int | None,
    ) -> str | None:
        """Resolve a final to its routed item, falling back to the active input."""
        st = self._state(conn_id)
        if turn_id is not None:
            routed_item_id = st.input_item_by_turn_revision.get((turn_id, turn_revision))
            if routed_item_id is not None:
                return routed_item_id
        return st.current_input_item_id

    def on_transcription_completed(
        self,
        conn_id: str,
        event: TranscriptionCompletedEvent,
    ) -> list[ConversationItemInputAudioTranscriptionCompletedEvent]:
        """Terminalize one transcript item and emit its authoritative final event."""
        st = self._state(conn_id)
        item_id = self._completion_input_item_id(conn_id, event.turn_id, event.turn_revision)
        if item_id is None:
            # Preserve the pre-routing fallback for protocol-neutral pipelines
            # that do not publish speech lifecycle metadata. #485 tracks a
            # stricter standalone/ambiguous-terminal policy.
            item_id = self._service.response._current_item_id(conn_id)
            duration_s = st.input_audio_duration_s
        else:
            terminal = self.terminalize_input_item(conn_id, item_id)
            if terminal is None:
                return []
            item_id, duration_s = terminal
        st.response_usage.audio_duration_s += duration_s
        return [
            ConversationItemInputAudioTranscriptionCompletedEvent(
                type="conversation.item.input_audio_transcription.completed",
                event_id=self._next_event_id(),
                content_index=0,
                item_id=item_id,
                transcript=event.transcript,
                usage=UsageTranscriptTextUsageDuration(
                    seconds=duration_s,
                    type="duration",
                ),
            )
        ]
