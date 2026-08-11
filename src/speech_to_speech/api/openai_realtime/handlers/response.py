from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, Optional

from openai.types.realtime import (
    ConversationItem,
    RealtimeConversationItemFunctionCall,
    RealtimeResponse,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseCreatedEvent,
    ResponseCreateEvent,
    ResponseDoneEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from openai.types.realtime.conversation_item import RealtimeConversationItemAssistantMessage
from openai.types.realtime.realtime_conversation_item_assistant_message import Content
from openai.types.realtime.realtime_response import Audio, AudioOutput
from openai.types.realtime.realtime_response_status import Error as RealtimeResponseStatusError
from openai.types.realtime.realtime_response_status import RealtimeResponseStatus
from openai.types.realtime.realtime_response_usage import RealtimeResponseUsage

from speech_to_speech.api.openai_realtime.handlers.base import RealtimeBaseHandler
from speech_to_speech.LLM.chat import ChatItemError, add_supported_item
from speech_to_speech.pipeline.events import (
    AssistantOutputEvent,
    AssistantResponseDoneEvent,
    ResponseGenerationDoneEvent,
)
from speech_to_speech.pipeline.messages import (
    AssistantTextPart,
    AssistantToolCallPart,
    GenerateResponseRequest,
    ResponsePrefetchTransaction,
)
from speech_to_speech.utils.utils import _generate_id, is_out_of_band, response_wants_audio

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.service import ServerEvent, _ResponseStatus, _StatusReason

logger = logging.getLogger(__name__)


class ResponseHandler(RealtimeBaseHandler):
    """Owns the response lifecycle: create, cancel, finish, and ID management."""

    # ── ID / state helpers ────────────────────────

    def _ensure_response(self, conn_id: str, response_key: str | None = None) -> tuple[str, str]:
        """Ensure a response and output item exist, creating them if needed."""
        st = self._state(conn_id)
        effective_response_key = response_key or st.current_response_key
        if (
            st.current_response_id is not None
            and response_key is not None
            and st.current_response_key is not None
            and response_key != st.current_response_key
        ):
            raise RuntimeError("Cannot attach output to a different active response")
        if st.current_response_id is None:
            st.current_response_id = _generate_id("resp")
            st.current_response_key = response_key
            self._start_item(conn_id)
            st.in_response = True
        elif st.current_response_key is None:
            st.current_response_key = response_key
        st.clear_pending_response(effective_response_key)
        return st.current_response_id, self._current_item_id(conn_id)

    def _end_response(self, conn_id: str, status: _ResponseStatus = "completed") -> None:
        st = self._state(conn_id)
        if status == "cancelled":
            st.response_usage.responses_cancelled += 1
        else:
            st.response_usage.responses_completed += 1
        self._service.total_usage += st.response_usage
        logger.info(
            "Response done (status=%s) — this response: input_tokens=%d, output_tokens=%d, audio=%.2fs"
            " | cumulative: input_tokens=%d, output_tokens=%d, audio=%.2fs",
            status,
            st.response_usage.input_tokens,
            st.response_usage.output_tokens,
            st.response_usage.audio_duration_s,
            self._service.total_usage.input_tokens,
            self._service.total_usage.output_tokens,
            self._service.total_usage.audio_duration_s,
        )
        st.response_usage.reset()
        completed_response_key = st.current_response_key
        completed_with_tools = bool(st.pending_function_calls)
        if (
            status == "completed"
            and completed_response_key is not None
            and completed_with_tools
            and not is_out_of_band(st.current_response_params)
            and completed_response_key not in st.generation_done_tool_calls
        ):
            # The side-channel can trail the ordered TTS terminal when its
            # queue is backlogged. Remember only successfully completed tool
            # responses so that late logical completion remains actionable.
            st.completed_tool_response_keys[completed_response_key] = None
            while len(st.completed_tool_response_keys) > 128:
                st.completed_tool_response_keys.pop(next(iter(st.completed_tool_response_keys)))
        elif status != "completed":
            self.discard_tool_followup_prefetch(conn_id, origin_response_key=completed_response_key)
            if completed_response_key is not None:
                st.completed_tool_response_keys.pop(completed_response_key, None)
        st.current_response_id = None
        st.current_response_key = None
        st.response_failed = False
        st.response_error_type = None
        st.current_item_id = None
        st.content_index = 0
        st.in_response = False
        self._service.close_response_key(conn_id, completed_response_key)
        st.current_response_params = None
        st.pending_assistant_item_id = None
        st.pending_assistant_output_index = None
        st.next_output_index = 0
        st.current_output_index = None
        st.current_output_kind = None
        st.audio_output_started = False
        st.pending_text_outputs = []
        st.pending_function_calls = {}

    @staticmethod
    def _prefetch_matches(event: ResponseCreateEvent) -> bool:
        """Whether a standard create can adopt generation made from session defaults."""
        response = event.response
        if response is None:
            return True
        fields = response.model_dump(exclude_unset=True)
        # Metadata is response bookkeeping and does not affect model or TTS
        # output. Any other per-response override requires a fresh generation.
        return set(fields).issubset({"metadata"})

    def is_response_output_blocked(self, conn_id: str, response_key: str | None) -> bool:
        """Return whether output must remain private from the client."""
        st = self._state(conn_id)
        request = st.tool_followup_prefetch_request
        return response_key is not None and (
            request is not None
            and response_key == request.response_key
            or response_key == st.response_created_pending_key
        )

    def mark_response_created_sent(self, conn_id: str, response_key: str | None) -> None:
        """Release output only after the matching response.created send completes."""
        st = self._state(conn_id)
        if response_key is not None and st.response_created_pending_key == response_key:
            st.response_created_pending_key = None

    def maybe_start_tool_followup_prefetch(self, conn_id: str) -> bool:
        """Speculatively generate after every tool output, without opening a response."""
        st = self._state(conn_id)
        if (
            st.tool_followup_prefetch_request is not None
            or st.deferred_items
            or st.runtime_config.chat.has_pending_tool_calls()
        ):
            return False

        origin_response_key = next(
            (key for key, call_ids in st.generation_done_tool_calls.items() if call_ids),
            None,
        )
        queue = self._queue(conn_id)
        if origin_response_key is None or queue is None:
            return False

        request = GenerateResponseRequest(
            runtime_config=st.runtime_config,
            turn_id=st.speculative_user_turn_id,
            turn_revision=st.speculative_user_turn_revision,
            speech_stopped_at_s=st.speculative_user_speech_stopped_at_s,
            prefetch_transaction=ResponsePrefetchTransaction(),
        )
        st.tool_followup_prefetch_request = request
        st.tool_followup_prefetch_origin_response_key = origin_response_key
        st.mark_response_pending(request.response_key)
        queue.put(request)
        logger.debug("Started internal tool follow-up prefetch")
        return True

    def discard_tool_followup_prefetch(
        self,
        conn_id: str,
        *,
        origin_response_key: str | None = None,
    ) -> None:
        """Invalidate unclaimed speculative work while preserving tool inputs."""
        st = self._state(conn_id)
        request = st.tool_followup_prefetch_request
        if request is None:
            if origin_response_key is not None:
                st.generation_done_tool_calls.pop(origin_response_key, None)
                st.completed_tool_response_keys.pop(origin_response_key, None)
            return
        if origin_response_key is not None and st.tool_followup_prefetch_origin_response_key != origin_response_key:
            st.generation_done_tool_calls.pop(origin_response_key, None)
            st.completed_tool_response_keys.pop(origin_response_key, None)
            return

        queue = self._queue(conn_id)
        if queue is not None:
            # Avoid making a replacement response wait behind speculation that
            # has not reached the LM worker yet. If the worker already owns it,
            # response-key tombstoning below still suppresses every output.
            with queue.mutex:
                try:
                    queue.queue.remove(request)
                except ValueError:
                    pass
                else:
                    queue.not_full.notify()
        if request.prefetch_transaction is not None:
            request.prefetch_transaction.discard()
        st.runtime_config.chat.rollback_provisional_generation(request.response_key)
        self._service.close_response_key(conn_id, request.response_key)
        st.generation_done_tool_calls.pop(request.response_key, None)
        st.completed_tool_response_keys.pop(request.response_key, None)
        st.tool_followup_prefetch_request = None
        st.tool_followup_prefetch_origin_response_key = None
        if origin_response_key is not None:
            st.generation_done_tool_calls.pop(origin_response_key, None)
            st.completed_tool_response_keys.pop(origin_response_key, None)

    def _claim_tool_followup_prefetch(
        self,
        conn_id: str,
        event: ResponseCreateEvent,
    ) -> ServerEvent | None:
        """Expose an internal prefetch as the response requested by the client."""
        st = self._state(conn_id)
        request = st.tool_followup_prefetch_request
        assert request is not None
        if request.prefetch_transaction is not None:
            try:
                claim_succeeded = request.prefetch_transaction.claim()
            except Exception:
                # Deferred image/history cleanup runs at the standard create
                # boundary. A failure must not escape the transport or leave
                # the hidden response occupying the session indefinitely.
                logger.exception("Failed to commit tool follow-up prefetch history")
                self.discard_tool_followup_prefetch(
                    conn_id,
                    origin_response_key=st.tool_followup_prefetch_origin_response_key,
                )
                # The failure belongs to invisible speculative work. Returning
                # None lets this same standard response.create fall through to
                # fresh generation without exposing the prefetch failure.
                return None
            if not claim_succeeded:
                self.discard_tool_followup_prefetch(
                    conn_id,
                    origin_response_key=st.tool_followup_prefetch_origin_response_key,
                )
                return None

        origin_response_key = st.tool_followup_prefetch_origin_response_key
        st.tool_followup_prefetch_request = None
        st.tool_followup_prefetch_origin_response_key = None
        if origin_response_key is not None:
            st.generation_done_tool_calls.pop(origin_response_key, None)
            st.completed_tool_response_keys.pop(origin_response_key, None)
        st.in_response = True
        st.clear_pending_response(request.response_key)
        st.current_response_params = event.response
        st.current_response_id = _generate_id("resp")
        st.current_response_key = request.response_key
        st.response_created_pending_key = request.response_key
        self._start_item(conn_id)
        logger.debug("Standard response.create claimed internal tool follow-up prefetch")
        return ResponseCreatedEvent(
            type="response.created",
            event_id=self._next_event_id(),
            response=self._build_response(conn_id, "in_progress"),
        )

    def on_response_generation_done(
        self,
        conn_id: str,
        event: ResponseGenerationDoneEvent,
    ) -> list[ServerEvent]:
        """Record logical LM completion independently of downstream TTS."""
        if event.response_key is None:
            return []
        st = self._state(conn_id)
        prefetch_request = st.tool_followup_prefetch_request
        if not event.succeeded and prefetch_request is not None and prefetch_request.response_key == event.response_key:
            # A hidden failure must remain invisible. Remove it now so the
            # client's later standard response.create starts fresh generation
            # instead of claiming an already-failed speculative response.
            self.discard_tool_followup_prefetch(
                conn_id,
                origin_response_key=st.tool_followup_prefetch_origin_response_key,
            )
            return []
        if event.response_key in st.closed_response_keys:
            if event.response_key not in st.completed_tool_response_keys:
                return []
            st.completed_tool_response_keys.pop(event.response_key, None)
        if st.in_response and st.current_response_key not in (None, event.response_key):
            if prefetch_request is not None and prefetch_request.response_key == event.response_key:
                # A prefetched follow-up can finish its own LM pass while the
                # origin response is still delivering TTS. Remember its tool
                # calls for the next round, but do not touch the origin's
                # deferred inputs or start another speculative response.
                if event.succeeded and event.call_ids:
                    st.generation_done_tool_calls[event.response_key] = set(event.call_ids)
                    while len(st.generation_done_tool_calls) > 128:
                        st.generation_done_tool_calls.pop(next(iter(st.generation_done_tool_calls)))
                return []
            # A newer response already owns the slot and its deferred items.
            # This late signal cannot open another follow-up or flush that
            # response's conversation inputs.
            return []
        if not event.succeeded:
            self.discard_tool_followup_prefetch(conn_id, origin_response_key=event.response_key)
            return []
        if st.current_response_key == event.response_key and is_out_of_band(st.current_response_params):
            # Out-of-band output is absent from the default Chat. It therefore
            # cannot seed a tool follow-up prefetch against that conversation.
            return []
        events = self._service.conversation.flush_deferred_items(
            conn_id,
            tool_followup_inputs_only=True,
            defer_acknowledgements=True,
        )
        if event.call_ids:
            st.generation_done_tool_calls[event.response_key] = set(event.call_ids)
            while len(st.generation_done_tool_calls) > 128:
                st.generation_done_tool_calls.pop(next(iter(st.generation_done_tool_calls)))
            self.maybe_start_tool_followup_prefetch(conn_id)
        return events

    def _start_item(self, conn_id: str) -> str:
        """Generate a new item ID, reset content index, and store it."""
        st = self._state(conn_id)
        item_id = _generate_id("item")
        st.current_item_id = item_id
        st.content_index = 0
        st.input_audio_duration_s = 0.0
        return item_id

    def _current_item_id(self, conn_id: str) -> str:
        return self._state(conn_id).current_item_id or self._start_item(conn_id)

    def _ensure_assistant_output_item(
        self,
        conn_id: str,
        item_id: str,
    ) -> tuple[str, int]:
        """Reserve the assistant item used by the outbound audio stream."""
        st = self._state(conn_id)
        if st.pending_assistant_item_id is not None and st.pending_assistant_output_index is not None:
            return st.pending_assistant_item_id, st.pending_assistant_output_index

        output_index, item_id = self._output_part_context(conn_id, "text", preferred_item_id=item_id)
        st.pending_text_outputs.append({"item_id": item_id, "output_index": output_index, "parts": []})
        st.pending_assistant_item_id = item_id
        st.pending_assistant_output_index = output_index
        if st.last_item_id is None:
            st.last_item_id = item_id
        return item_id, output_index

    def _next_content_index(self, conn_id: str) -> int:
        """Return the index of the output item's sole audio content part."""
        return self._state(conn_id).content_index

    def _output_part_context(
        self,
        conn_id: str,
        kind: Literal["text", "tool_call"],
        *,
        preferred_item_id: str | None = None,
    ) -> tuple[int, str]:
        """Return a stable output index and item id for an ordered part.

        Consecutive text chunks share one assistant output item. Every tool
        call starts a new item, and text after a tool starts a new assistant
        item, preserving order even when parts arrive in separate events.
        """
        st = self._state(conn_id)
        if (
            kind == "text"
            and st.current_output_kind == "text"
            and st.current_output_index is not None
            and st.current_item_id is not None
        ):
            return st.current_output_index, st.current_item_id

        if preferred_item_id is not None:
            item_id = preferred_item_id
            st.current_item_id = item_id
            st.content_index = 0
        else:
            item_id = self._current_item_id(conn_id) if st.next_output_index == 0 else self._start_item(conn_id)
        output_index = st.next_output_index
        st.next_output_index += 1
        st.current_output_index = output_index
        st.current_output_kind = kind
        return output_index, item_id

    def _build_response(
        self,
        conn_id: str,
        status: _ResponseStatus,
        reason: _StatusReason | None = None,
    ) -> RealtimeResponse:
        """Build a fully-populated RealtimeResponse from the current connection state."""
        st = self._state(conn_id)
        status_details = None
        if reason or status in ("completed", "cancelled", "incomplete", "failed"):
            error = (
                RealtimeResponseStatusError(type=st.response_error_type or "response_failed")
                if status == "failed"
                else None
            )
            status_details = RealtimeResponseStatus(type=status, reason=reason, error=error)  # type: ignore[arg-type]

        rp = st.current_response_params
        metadata = rp.metadata if rp and rp.metadata else None

        voice: Optional[str] = None
        if rp and rp.audio and rp.audio.output and rp.audio.output.voice:
            voice = str(rp.audio.output.voice)
        if not voice:
            audio_cfg = st.runtime_config.session.audio
            audio_output = audio_cfg.output if audio_cfg is not None else None
            voice = str(audio_output.voice) if audio_output is not None and audio_output.voice else None

        # Out-of-band responses are not threaded into any conversation: report a null id.
        conversation_id = None if is_out_of_band(rp) else st.conversation_id

        return RealtimeResponse(
            id=st.current_response_id,
            object="realtime.response",
            status=status,
            status_details=status_details,
            audio=Audio(output=AudioOutput(voice=str(voice) if voice else None)),  # type: ignore[arg-type]
            conversation_id=conversation_id,
            metadata=metadata,
            output=self._build_output_items(conn_id, status),
            usage=RealtimeResponseUsage(
                input_tokens=st.response_usage.input_tokens,
                output_tokens=st.response_usage.output_tokens,
                total_tokens=st.response_usage.input_tokens + st.response_usage.output_tokens,
            ),
        )

    # Annotated with ConversationItem (the SDK's own 9-type item union) rather
    # than the two types actually produced: list is invariant, so a narrower
    # element type is rejected where RealtimeResponse.output is assigned.
    def _build_output_items(self, conn_id: str, status: _ResponseStatus) -> list[ConversationItem]:
        """Build response.output in the same item order used by streaming events,
        per the OpenAI Realtime protocol - see
        https://platform.openai.com/docs/api-reference/realtime-server-events/session/updated
        ("response.done will also have the complete data we need to call our function").
        """
        st = self._state(conn_id)
        assistant_status: Literal["completed", "incomplete"] = "completed" if status == "completed" else "incomplete"
        output_by_index: dict[int, ConversationItem] = {}
        for pending in st.pending_text_outputs:
            output_index = int(pending["output_index"])
            text = self._assistant_text(pending, response_wants_audio(st.current_response_params))
            if response_wants_audio(st.current_response_params):
                content = Content(type="output_audio", transcript=text)
            else:
                content = Content(type="output_text", text=text)
            output_by_index[output_index] = RealtimeConversationItemAssistantMessage(
                type="message",
                role="assistant",
                id=str(pending["item_id"]),
                object="realtime.item",
                status=assistant_status,
                content=[content],
            )

        for output_index, call in st.pending_function_calls.items():
            if call.status in ("completed", "incomplete"):
                call_status = call.status
            elif status == "completed":
                call_status = "completed"
            else:
                call_status = "incomplete"
            output_by_index[output_index] = call.model_copy(update={"object": "realtime.item", "status": call_status})
        return [output_by_index[index] for index in sorted(output_by_index)]

    @staticmethod
    def _assistant_text(pending: dict[str, object], wants_audio: bool) -> str:
        """Assemble transcript parts using the active output modality's semantics."""
        parts = pending["parts"]
        assert isinstance(parts, list)
        if wants_audio:
            return " ".join(str(part).strip() for part in parts if str(part).strip())
        return "".join(str(part) for part in parts)

    # ── Public handlers ───────────────────────────

    def handle_response_create(self, conn_id: str, event: ResponseCreateEvent) -> ServerEvent | None:
        """Trigger a response.

        Returns a ``ResponseCreatedEvent`` on success, a ``RealtimeErrorEvent``
        on failure, or ``None`` if there is no text_prompt_queue.
        """
        st = self._state(conn_id)
        if event.response:
            if event.response.tool_choice and not isinstance(event.response.tool_choice, str):
                return self.make_error(
                    message="Only string tool_choice values are supported for now (auto, required, none).",
                    _type="tool_choice_not_supported",
                )
        prefetch_request = st.tool_followup_prefetch_request if not st.in_response else None
        if prefetch_request is not None:
            if self._prefetch_matches(event):
                claimed = self._claim_tool_followup_prefetch(conn_id, event)
                if claimed is not None:
                    return claimed
                prefetch_request = None
        replacing_prefetch = prefetch_request is not None
        if st.in_response or (st.response_pending and not replacing_prefetch):
            return self.make_error(
                message="Cannot create response while another response is in progress or pending.",
                _type="conversation_already_has_active_response",
            )

        out_of_band = is_out_of_band(event.response)

        # In-band: response.input items are added to the default conversation here so
        # they appear in history. Out-of-band: leave the default conversation untouched —
        # the input rides along on the request and seeds a throwaway chat in the LM.
        input_items: list[ConversationItem] = []
        if not out_of_band:
            input_items = list(event.response.input) if event.response and event.response.input else []
            candidate_chat = (
                st.runtime_config.chat.copy_without_provisional_generation(prefetch_request.response_key)
                if prefetch_request is not None
                else st.runtime_config.chat.copy(deep=True)
            )
            try:
                for input_item in input_items:
                    add_supported_item(candidate_chat, input_item.model_copy(deep=True))
            except ChatItemError as exc:
                return self.make_error(message=str(exc), _type="invalid_input_item")
            if candidate_chat.has_pending_tool_calls():
                return self.make_error(
                    message="Cannot create a response while function call outputs are pending.",
                    _type="function_call_output_pending",
                )

        if replacing_prefetch:
            # The speculative request used session defaults. Validate a
            # replacement before discarding otherwise reusable work.
            origin_response_key = st.tool_followup_prefetch_origin_response_key
            self.discard_tool_followup_prefetch(conn_id)
            if origin_response_key is not None:
                st.generation_done_tool_calls.pop(origin_response_key, None)
                st.completed_tool_response_keys.pop(origin_response_key, None)
            if st.response_pending:
                return self.make_error(
                    message="Cannot create response while another response is pending.",
                    _type="conversation_already_has_active_response",
                )

        if not out_of_band:
            for input_item in input_items:
                self._service.conversation._append_item(conn_id, input_item)
            # A normal in-band create has consumed the current conversation.
            # Any delayed logical-done marker for the prior tool response must
            # not manufacture a second follow-up behind this one.
            st.completed_tool_response_keys.clear()
            st.generation_done_tool_calls.clear()

        cfg = st.runtime_config
        queue = self._queue(conn_id)
        request = GenerateResponseRequest(
            runtime_config=cfg,
            response=event.response,
            turn_id=None if out_of_band else st.speculative_user_turn_id,
            turn_revision=None if out_of_band else st.speculative_user_turn_revision,
            speech_stopped_at_s=None if out_of_band else st.speculative_user_speech_stopped_at_s,
        )
        st.in_response = True
        st.clear_pending_response(request.response_key)
        st.current_response_params = event.response
        st.current_response_id = _generate_id("resp")
        st.current_response_key = request.response_key
        st.response_created_pending_key = request.response_key
        self._start_item(conn_id)

        if queue:
            # Out-of-band responses carry no turn identity: a null turn_id makes every
            # speculative-turn staleness gate treat them as always-latest, so a new user
            # turn mid-generation can never silently drop their output.
            queue.put(request)
        logger.debug("response.create received, LLM generation triggered")
        return ResponseCreatedEvent(
            type="response.created",
            event_id=self._next_event_id(),
            response=self._build_response(conn_id, "in_progress"),
        )

    def handle_response_cancel(self, conn_id: str) -> list[ServerEvent]:
        """Cancel the in-progress response and re-enable listening."""
        events = self.finish_response(conn_id, status="cancelled", reason="client_cancelled")
        self._service.close_pending_responses(conn_id)
        should_listen = self._should_listen(conn_id)
        if should_listen:
            should_listen.set()
        logger.info("Response cancelled, listening re-enabled")
        return events

    def finish_audio_output(
        self,
        conn_id: str,
        response_key: str | None = None,
    ) -> list[ServerEvent]:
        """Close one synthesized assistant audio item exactly once."""
        st = self._state(conn_id)
        if response_key is not None and st.current_response_key not in (None, response_key):
            return []
        if (
            not st.audio_output_started
            or st.pending_assistant_item_id is None
            or st.pending_assistant_output_index is None
        ):
            return []
        item_id = st.pending_assistant_item_id
        output_index = st.pending_assistant_output_index
        st.audio_output_started = False
        st.pending_assistant_item_id = None
        st.pending_assistant_output_index = None
        resp_id, _ = self._ensure_response(conn_id, response_key)
        return [
            ResponseAudioDoneEvent(
                type="response.output_audio.done",
                event_id=self._next_event_id(),
                content_index=0,
                item_id=item_id,
                output_index=output_index,
                response_id=resp_id,
            )
        ]

    def finish_response(
        self,
        conn_id: str,
        status: _ResponseStatus = "completed",
        reason: _StatusReason | None = None,
        *,
        response_key: str | None = None,
    ) -> list[ServerEvent]:
        """Close the current response (audio/text done + response done).

        Audio responses emit ``response.output_audio.done`` unless their only
        output is a function call, followed by one terminal transcript event
        for each ordered assistant message. Text-only responses likewise close
        each assistant message, but only on ``status="completed"``.
        """
        st = self._state(conn_id)
        events: list[ServerEvent] = []
        if response_key is not None and st.current_response_key not in (None, response_key):
            return events
        if st.in_response:
            if status == "completed" and st.response_failed:
                status = "failed"
            resp_id, _ = self._ensure_response(conn_id)
            wants_audio = response_wants_audio(st.current_response_params)
            function_call_only = bool(st.pending_function_calls) and not st.pending_text_outputs
            if wants_audio and not function_call_only:
                events.extend(self.finish_audio_output(conn_id, response_key))
                for pending in st.pending_text_outputs:
                    transcript = self._assistant_text(pending, wants_audio=True)
                    if not transcript:
                        continue
                    events.append(
                        ResponseAudioTranscriptDoneEvent(
                            type="response.output_audio_transcript.done",
                            event_id=self._next_event_id(),
                            content_index=0,
                            item_id=str(pending["item_id"]),
                            output_index=int(pending["output_index"]),
                            response_id=resp_id,
                            transcript=transcript,
                        )
                    )
            else:
                for pending in st.pending_text_outputs:
                    text = self._assistant_text(pending, wants_audio=False)
                    if not text:
                        continue
                    events.append(
                        ResponseTextDoneEvent(
                            type="response.output_text.done",
                            event_id=self._next_event_id(),
                            content_index=0,
                            item_id=str(pending["item_id"]),
                            output_index=int(pending["output_index"]),
                            response_id=resp_id,
                            text=text,
                        )
                    )
            events.append(
                ResponseDoneEvent(
                    type="response.done",
                    event_id=self._next_event_id(),
                    response=self._build_response(conn_id, status, reason),
                )
            )
            if status == "completed":
                st.runtime_config.chat.finalize_provisional_generation(st.current_response_key)
            elif status in ("cancelled", "failed", "incomplete"):
                # Tool calls are recorded before their chunks reach the client.
                # Remove incomplete response history before deferred client items
                # are applied, so an unseen call cannot poison the next turn.
                st.runtime_config.chat.rollback_provisional_generation(st.current_response_key)
            self._end_response(conn_id, status)
        # Apply any client items that arrived mid-generation now that in_response
        # is cleared and the generation's own write-back has landed. Done outside
        # the in_response guard so a stray terminal call still drains the buffer.
        events.extend(
            self._service.conversation.flush_pending_item_acks(
                conn_id,
                revalidate_tool_outputs=status in ("cancelled", "failed", "incomplete"),
            )
        )
        events.extend(self._service.conversation.flush_deferred_items(conn_id))
        return events

    # ── Pipeline event handlers ───────────────────

    def on_assistant_output(
        self,
        conn_id: str,
        event: AssistantOutputEvent,
        *,
        wait_for_pending_reopen: bool = True,
    ) -> list[ServerEvent] | None:
        """Translate ordered assistant output into OpenAI Realtime events."""
        if self._service.speculative_turns:
            commit_result: bool | None
            if wait_for_pending_reopen:
                commit_result = self._service.speculative_turns.commit_if_latest_after_reopen_grace(
                    event.turn_id,
                    event.turn_revision,
                )
            else:
                commit_result = self._service.speculative_turns.try_commit_if_latest_after_reopen_grace(
                    event.turn_id,
                    event.turn_revision,
                )
            if commit_result is None:
                return None
            if not commit_result:
                logger.debug("Dropping stale assistant output for turn=%s rev=%s", event.turn_id, event.turn_revision)
                return []
        st = self._state(conn_id)
        events: list[ServerEvent] = []
        wants_audio = response_wants_audio(st.current_response_params)
        meaningful_parts = [
            part
            for part in event.parts
            if isinstance(part, AssistantToolCallPart)
            or (isinstance(part, AssistantTextPart) and (bool(part.text.strip()) if wants_audio else bool(part.text)))
        ]
        if not meaningful_parts:
            return events
        response_was_missing = st.current_response_id is None
        resp_id, _ = self._ensure_response(conn_id, event.response_key)
        if response_was_missing:
            events.append(
                ResponseCreatedEvent(
                    type="response.created",
                    event_id=self._next_event_id(),
                    response=self._build_response(conn_id, "in_progress"),
                )
            )
        self._service._apply_pending_token_usage(conn_id, event.response_key)
        for part in meaningful_parts:
            if isinstance(part, AssistantTextPart):
                text = part.text.strip() if wants_audio else part.text
                if not text:
                    continue
                output_idx, item_id = self._output_part_context(conn_id, "text")
                if st.pending_text_outputs and st.pending_text_outputs[-1]["output_index"] == output_idx:
                    pending = st.pending_text_outputs[-1]
                else:
                    st.pending_text_outputs.append({"item_id": item_id, "output_index": output_idx, "parts": []})
                    pending = st.pending_text_outputs[-1]
                parts = pending["parts"]
                assert isinstance(parts, list)
                if wants_audio:
                    st.pending_assistant_item_id = item_id
                    st.pending_assistant_output_index = output_idx
                    delta = (" " if parts else "") + text
                    parts.append(text)
                    events.append(
                        ResponseAudioTranscriptDeltaEvent(
                            type="response.output_audio_transcript.delta",
                            event_id=self._next_event_id(),
                            content_index=0,
                            delta=delta,
                            item_id=item_id,
                            output_index=output_idx,
                            response_id=resp_id,
                        )
                    )
                else:
                    parts.append(text)
                    events.append(
                        ResponseTextDeltaEvent(
                            type="response.output_text.delta",
                            event_id=self._next_event_id(),
                            content_index=0,
                            item_id=item_id,
                            output_index=output_idx,
                            response_id=resp_id,
                            delta=text,
                        )
                    )
                st.last_item_id = item_id
            elif isinstance(part, AssistantToolCallPart):
                events.extend(self.finish_audio_output(conn_id, event.response_key))
                tool = part.tool
                function_item_id = tool.id or _generate_id("item")
                output_idx, function_item_id = self._output_part_context(
                    conn_id,
                    "tool_call",
                    preferred_item_id=function_item_id,
                )
                st.response_usage.tool_calls += 1
                events.append(
                    ResponseFunctionCallArgumentsDoneEvent(
                        type="response.function_call_arguments.done",
                        event_id=self._next_event_id(),
                        call_id=tool.call_id,
                        name=tool.name,
                        arguments=tool.arguments,
                        item_id=function_item_id,
                        output_index=output_idx,
                        response_id=resp_id,
                    )
                )
                # Same item_id as the event above, so a client can correlate the
                # streamed arguments with the item that lands in response.output.
                # Status is stamped on at close, once the outcome is known.
                st.pending_function_calls[output_idx] = RealtimeConversationItemFunctionCall(
                    type="function_call",
                    object="realtime.item",
                    id=function_item_id,
                    call_id=tool.call_id,
                    name=tool.name,
                    arguments=tool.arguments,
                    status=tool.status or "completed",
                )
                st.last_item_id = function_item_id
        return events

    def on_assistant_response_done(
        self,
        conn_id: str,
        event: AssistantResponseDoneEvent,
    ) -> list[ServerEvent]:
        """Record that all ordered text/tool output for one response was emitted."""
        st = self._state(conn_id)
        response_was_missing = st.current_response_id is None
        self._ensure_response(conn_id, event.response_key)
        events = self.finish_audio_output(conn_id, event.response_key)
        events.extend(
            [
                ResponseCreatedEvent(
                    type="response.created",
                    event_id=self._next_event_id(),
                    response=self._build_response(conn_id, "in_progress"),
                )
            ]
            if response_was_missing
            else []
        )
        self._service._apply_pending_token_usage(conn_id, event.response_key)
        return events
