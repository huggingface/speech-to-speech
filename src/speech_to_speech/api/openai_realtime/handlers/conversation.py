from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unicodedata import category

from openai.types.realtime import (
    ConversationItem,
    ConversationItemCreatedEvent,
    ConversationItemCreateEvent,
    ConversationItemDeletedEvent,
    ConversationItemDeleteEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionDeltaEvent,
    ConversationItemInputAudioTranscriptionFailedEvent,
    ConversationItemRetrieveEvent,
)
from openai.types.realtime.conversation_item_input_audio_transcription_completed_event import (
    UsageTranscriptTextUsageDuration,
)
from openai.types.realtime.conversation_item_input_audio_transcription_failed_event import (
    Error as InputAudioTranscriptionError,
)
from openai.types.realtime.realtime_conversation_item_assistant_message import (
    Content as AssistantContent,
)
from openai.types.realtime.realtime_conversation_item_assistant_message import (
    RealtimeConversationItemAssistantMessage,
)
from openai.types.realtime.realtime_conversation_item_function_call import (
    RealtimeConversationItemFunctionCall,
)
from openai.types.realtime.realtime_conversation_item_function_call_output import (
    RealtimeConversationItemFunctionCallOutput,
)
from openai.types.realtime.realtime_conversation_item_user_message import (
    Content as UserContent,
)
from openai.types.realtime.realtime_conversation_item_user_message import (
    RealtimeConversationItemUserMessage,
)
from openai.types.realtime.realtime_server_event import ConversationItemRetrieved
from openai.types.responses import ResponseReasoningItem

from speech_to_speech.api.openai_realtime.handlers.base import RealtimeBaseHandler
from speech_to_speech.api.openai_realtime.input_state import (
    InputItemState,
    SpeechToSpeechInputAudioTranscriptionSnapshotEvent,
)
from speech_to_speech.LLM.chat import (
    ChatItemError,
    ResponsesAssistantMessage,
    ResponsesFunctionCall,
    SupportedItem,
    add_supported_item,
)
from speech_to_speech.pipeline.events import (
    PartialTranscriptionEvent,
    TranscriptionCompletedEvent,
    TranscriptionFailedEvent,
)
from speech_to_speech.pipeline.transcript_logging import transcript_for_log
from speech_to_speech.utils.utils import is_out_of_band, response_wants_audio

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
        items: list[ConversationItem | ConversationItemDeleteEvent],
    ) -> bool:
        """Return whether deferred items form a prefetch-safe tool batch.

        Image items are allowed immediately before a function output when
        ``previous_item_id`` confirms that insertion order. The field does not
        imply ownership, so every image remains an ordinary conversation item
        and must survive if the tool response is later rolled back.
        """
        st = self._state(conn_id)
        for index, item in enumerate(items):
            if isinstance(item, ConversationItemDeleteEvent):
                return False
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
        if st.in_response or st.response_pending:
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

    def handle_conversation_item_delete(self, conn_id: str, event: ConversationItemDeleteEvent) -> list[ServerEvent]:
        st = self._state(conn_id)
        if st.in_response or st.response_pending:
            st.deferred_items.append(event)
            return []
        return self._apply_delete(conn_id, event)

    def _apply_delete(self, conn_id: str, event: ConversationItemDeleteEvent) -> list[ServerEvent]:
        st = self._state(conn_id)
        chat = st.runtime_config.chat
        removed = chat.delete_item(event.item_id)
        # Committed speech is already on the wire while STT still owns its
        # input state; its Chat entry is only written when transcription ends.
        committed_input = event.item_id in st.input_items and event.item_id in st.acknowledged_item_ids
        empty_input = st.published_input_transcripts.pop(event.item_id, None) == ""
        if removed is None and not committed_input and not empty_input:
            error = self.make_error(
                f"No conversation item with id '{event.item_id}' found.", "conversation_item_not_found"
            )
            # Deferred errors are emitted outside the client dispatch callback.
            error.error.event_id = event.event_id
            return [error]
        reference = st.input_turn_by_item_id.pop(event.item_id, None)
        if reference is None:
            reference = next(
                (
                    (turn_id, revision, None)
                    for (turn_id, revision), item_id in st.input_item_by_turn_revision.items()
                    if item_id == event.item_id
                ),
                None,
            )
        if reference is not None and reference[0] is not None:
            st.deleted_input_turn_ids[reference[0]] = None
            while len(st.deleted_input_turn_ids) > 128:
                st.deleted_input_turn_ids.pop(next(iter(st.deleted_input_turn_ids)))
        st.pending_input_terminals.pop(event.item_id, None)
        self._service.audio._release_input_item_state_by_id(conn_id, event.item_id)
        retained_ids = chat.item_ids() | st.input_items.keys()
        st.acknowledged_item_ids = [item_id for item_id in st.acknowledged_item_ids if item_id in retained_ids]
        if removed is not None and removed.type == "function_call" and removed.call_id is not None:
            st.input_turn_by_call_id.pop(removed.call_id, None)
        if st.last_item_id == event.item_id:
            st.last_item_id = st.acknowledged_item_ids[-1] if st.acknowledged_item_ids else None
        return [
            ConversationItemDeletedEvent(
                type="conversation.item.deleted", event_id=self._next_event_id(), item_id=event.item_id
            )
        ]

    def handle_conversation_item_retrieve(
        self,
        conn_id: str,
        event: ConversationItemRetrieveEvent,
    ) -> list[ServerEvent]:
        """Answer ``conversation.item.retrieve`` with the server's copy of one item.

        Per the OpenAI Realtime client events reference, the server answers
        with ``conversation.item.retrieved`` "unless the item does not exist in
        the conversation history, in which case the server will respond with an
        error". Retrieval is read-only, so it is answered immediately, including
        while a response is active.
        """
        item = self._retrievable_item(conn_id, event.item_id)
        if item is None:
            return [
                self.make_error(f"No conversation item with id '{event.item_id}' found.", "conversation_item_not_found")
            ]
        return [
            ConversationItemRetrieved(
                type="conversation.item.retrieved",
                event_id=self._next_event_id(),
                item=item,
            )
        ]

    def _retrievable_item(self, conn_id: str, item_id: str) -> ConversationItem | None:
        """Return the wire form of an item the client has seen and the server still holds."""
        st = self._state(conn_id)
        published = st.published_input_transcripts
        if item_id not in st.acknowledged_item_ids and item_id not in published:
            return None

        # An active response's output items live in its pending state until
        # response.done binds them into history.
        if st.in_response and not is_out_of_band(st.current_response_params):
            wants_audio = response_wants_audio(st.current_response_params)
            for pending in st.pending_text_outputs:
                if pending["item_id"] == item_id:
                    item = self._service.response._build_message_item(pending, wants_audio, "completed")
                    item.status = pending.get("status", "in_progress")
                    return item
            for call in st.pending_function_calls.values():
                if call.id == item_id:
                    return call.model_copy(update={"object": "realtime.item", "status": call.status or "in_progress"})

        speech = item_id in published or item_id in st.input_items
        held = st.runtime_config.chat.get_item(item_id)
        if held is not None:
            if (
                speech
                and isinstance(held, RealtimeConversationItemUserMessage)
                and all(part.type == "input_text" for part in held.content or [])
            ):
                # Transcribed speech is held as LLM text, possibly annotated
                # with its speaker. Return the input_audio item the client saw,
                # with the transcript it was sent (null until it is published).
                return self._user_audio_item(item_id, published.get(item_id))
            wire = self._wire_item(held)
            if isinstance(wire, RealtimeConversationItemAssistantMessage) and item_id in st.audio_output_item_ids:
                # The chat holds an audio reply as LLM text. Restore the
                # output_audio part, joined the way the transcript streamed.
                transcript = " ".join(part.text.strip() for part in wire.content if part.text and part.text.strip())
                wire.content = [AssistantContent(type="output_audio", transcript=transcript)]
            return wire
        if item_id in st.input_items:
            # Committed speech whose transcription has not finished yet.
            return self._user_audio_item(item_id, None)
        if published.get(item_id) == "":
            # An empty transcript never enters the chat, but the client was
            # told about the item and its empty completed transcription.
            return self._user_audio_item(item_id, "")
        return None

    @staticmethod
    def _user_audio_item(item_id: str, transcript: str | None) -> RealtimeConversationItemUserMessage:
        return RealtimeConversationItemUserMessage(
            id=item_id,
            object="realtime.item",
            type="message",
            role="user",
            status="completed",
            content=[UserContent(type="input_audio", transcript=transcript)],
        )

    @staticmethod
    def _wire_item(item: SupportedItem) -> ConversationItem | None:
        """Convert one held chat item to the Realtime item shape the client saw."""
        if isinstance(item, ResponseReasoningItem):
            return None
        wire: ConversationItem
        if isinstance(item, ResponsesAssistantMessage):
            wire = RealtimeConversationItemAssistantMessage.model_validate(item.model_dump(exclude={"response_item"}))
        elif isinstance(item, ResponsesFunctionCall):
            wire = RealtimeConversationItemFunctionCall.model_validate(item.model_dump(exclude={"response_item"}))
        else:
            wire = item
        return wire.model_copy(update={"object": "realtime.item", "status": wire.status or "completed"})

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
        if item.id is not None:
            self.record_acknowledged_item(conn_id, item.id)
        return event

    def record_acknowledged_item(self, conn_id: str, item_id: str) -> None:
        """Track retained wire chronology so deleting the tail restores its predecessor."""
        st = self._state(conn_id)
        retained_ids = st.runtime_config.chat.item_ids() | st.input_items.keys()
        retained_ids.update(str(item["item_id"]) for item in st.pending_text_outputs)
        retained_ids.update(call.id for call in st.pending_function_calls.values() if call.id is not None)
        st.acknowledged_item_ids = [
            known_id for known_id in st.acknowledged_item_ids if known_id in retained_ids and known_id != item_id
        ]
        st.acknowledged_item_ids.append(item_id)
        st.last_item_id = item_id

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
        if st.response_pending and not tool_followup_inputs_only:
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
            if isinstance(item, ConversationItemDeleteEvent):
                events.extend(self._apply_delete(conn_id, item))
                continue
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

        events: list[ServerEvent] = []
        if st.runtime_config.input_audio_transcription_snapshots_enabled:
            events.append(
                SpeechToSpeechInputAudioTranscriptionSnapshotEvent(
                    type="speech_to_speech.input_audio_transcription.snapshot",
                    event_id=self._next_event_id(),
                    item_id=item_id,
                    content_index=0,
                    transcript=hypothesis,
                )
            )

        if not previous:
            return events

        stable_words = _stable_transcript_words(previous, hypothesis)
        emitted_words = _transcript_words(input_item.transcript_prefix)
        emitted_comparison = [word[1] for word in emitted_words]
        stable_comparison = [word[1] for word in stable_words]
        if stable_comparison[: len(emitted_comparison)] != emitted_comparison:
            # A word that already reached the wire was later revised. Realtime
            # has no retraction event, so wait for a future hypothesis that
            # extends the committed prefix or for the authoritative completion.
            logger.debug(
                "Withholding revised stable transcription for item=%s (emitted=%s, hypothesis=%s)",
                item_id,
                transcript_for_log(input_item.transcript_prefix),
                transcript_for_log(hypothesis),
            )
            return events

        new_words = stable_words[len(emitted_words) :]
        if not new_words:
            return events

        delta = (" " if emitted_words else "") + " ".join(word[0] for word in new_words)
        input_item.transcript_prefix += delta
        events.append(
            ConversationItemInputAudioTranscriptionDeltaEvent(
                type="conversation.item.input_audio_transcription.delta",
                event_id=self._next_event_id(),
                content_index=0,
                item_id=item_id,
                delta=delta,
            )
        )
        return events

    def _closing_input_item(self, conn_id: str, item_id: str) -> InputItemState | None:
        """Return the item a terminal closes, or ``None`` once it was released.

        Its transcript and routing state survive until
        :meth:`AudioHandler.resolve_input_terminals` publishes the terminal,
        because speech that resumes first keeps revising the same item.
        """
        input_item = self._state(conn_id).input_items.get(item_id)
        if input_item is None:
            logger.debug("Ignoring input terminal for released item=%s", item_id)
        return input_item

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
            input_item = self._closing_input_item(conn_id, item_id)
            if input_item is None:
                return []
            duration_s = input_item.audio_duration_s
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

    def on_transcription_failed(
        self,
        conn_id: str,
        event: TranscriptionFailedEvent,
    ) -> list[ConversationItemInputAudioTranscriptionFailedEvent]:
        """Terminalize one transcript item and emit its item-scoped failure."""
        st = self._state(conn_id)
        if event.turn_id is not None:
            item_id = st.input_item_by_turn_revision.get((event.turn_id, event.turn_revision))
        else:
            item_id = st.current_input_item_id
        if item_id is None:
            logger.debug(
                "Ignoring transcription failure for unknown turn=%s rev=%s",
                event.turn_id,
                event.turn_revision,
            )
            return []
        if self._closing_input_item(conn_id, item_id) is None:
            return []
        return [
            ConversationItemInputAudioTranscriptionFailedEvent(
                type="conversation.item.input_audio_transcription.failed",
                event_id=self._next_event_id(),
                content_index=0,
                item_id=item_id,
                error=InputAudioTranscriptionError(
                    type="transcription_error",
                    code="transcription_failed",
                    message=event.message,
                    param=None,
                ),
            )
        ]
