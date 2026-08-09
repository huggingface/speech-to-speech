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
from speech_to_speech.pipeline.events import AssistantResponseDoneEvent, AssistantTextEvent
from speech_to_speech.pipeline.messages import AssistantTextPart, AssistantToolCallPart, GenerateResponseRequest
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
        st.current_response_id = None
        st.current_response_key = None
        st.response_text_complete = False
        st.response_failed = False
        st.response_error_type = None
        st.current_item_id = None
        st.content_index = 0
        st.in_response = False
        st.close_response_key(completed_response_key)
        st.current_response_params = None
        st.pending_assistant_item_id = None
        st.pending_assistant_output_index = None
        st.next_output_index = 0
        st.current_output_index = None
        st.current_output_kind = None
        st.assistant_output_items = {}
        st.completed_audio_output_indices = set()
        st.started_audio_output_indices = set()
        st.pending_text_outputs = []
        st.pending_function_calls = {}

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
        assistant_output_ordinal: int | None = None,
    ) -> tuple[str, int]:
        """Reserve the assistant item used by the outbound audio stream."""
        st = self._state(conn_id)
        target: tuple[str, int] | None = None
        if assistant_output_ordinal is not None:
            target = st.assistant_output_items.get(assistant_output_ordinal)
            if target is None:
                output_index, output_item_id = self._output_part_context(
                    conn_id,
                    "text",
                    preferred_item_id=item_id if st.next_output_index == 0 else None,
                )
                st.pending_text_outputs.append({"item_id": output_item_id, "output_index": output_index, "parts": []})
                target = (output_item_id, output_index)
                st.assistant_output_items[assistant_output_ordinal] = target
        elif st.pending_assistant_item_id is not None and st.pending_assistant_output_index is not None:
            return st.pending_assistant_item_id, st.pending_assistant_output_index

        if target is None:
            if st.pending_text_outputs:
                pending = st.pending_text_outputs[0]
                target = (str(pending["item_id"]), int(pending["output_index"]))
            else:
                output_index, item_id = self._output_part_context(conn_id, "text", preferred_item_id=item_id)
                st.pending_text_outputs.append({"item_id": item_id, "output_index": output_index, "parts": []})
                target = (item_id, output_index)

        st.pending_assistant_item_id, st.pending_assistant_output_index = target
        if st.last_item_id is None:
            st.last_item_id = st.pending_assistant_item_id
        return target

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
        if st.in_response or st.response_pending:
            return self.make_error(
                message="Cannot create response while another response is in progress or pending.",
                _type="conversation_already_has_active_response",
            )

        out_of_band = is_out_of_band(event.response)

        # In-band: response.input items are added to the default conversation here so
        # they appear in history. Out-of-band: leave the default conversation untouched —
        # the input rides along on the request and seeds a throwaway chat in the LM.
        if not out_of_band:
            input_items = list(event.response.input) if event.response and event.response.input else []
            candidate_chat = st.runtime_config.chat.copy(deep=True)
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
            for input_item in input_items:
                self._service.conversation._append_item(conn_id, input_item)

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
        self._state(conn_id).close_pending_responses()
        should_listen = self._should_listen(conn_id)
        if should_listen:
            should_listen.set()
        logger.info("Response cancelled, listening re-enabled")
        return events

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
            resp_id, item_id = self._ensure_response(conn_id)
            wants_audio = response_wants_audio(st.current_response_params)
            function_call_only = bool(st.pending_function_calls) and not st.pending_text_outputs
            if wants_audio and not function_call_only:
                if st.pending_text_outputs:
                    for pending in st.pending_text_outputs:
                        output_index = int(pending["output_index"])
                        if (
                            output_index in st.completed_audio_output_indices
                            or output_index not in st.started_audio_output_indices
                        ):
                            continue
                        events.append(
                            ResponseAudioDoneEvent(
                                type="response.output_audio.done",
                                event_id=self._next_event_id(),
                                content_index=0,
                                item_id=str(pending["item_id"]),
                                output_index=output_index,
                                response_id=resp_id,
                            )
                        )
                elif st.pending_assistant_output_index in st.started_audio_output_indices:
                    events.append(
                        ResponseAudioDoneEvent(
                            type="response.output_audio.done",
                            event_id=self._next_event_id(),
                            content_index=0,
                            item_id=st.pending_assistant_item_id or item_id,
                            output_index=st.pending_assistant_output_index or 0,
                            response_id=resp_id,
                        )
                    )
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
            self._end_response(conn_id, status)
        # Apply any client items that arrived mid-generation now that in_response
        # is cleared and the generation's own write-back has landed. Done outside
        # the in_response guard so a stray terminal call still drains the buffer.
        events.extend(self._service.conversation.flush_deferred_items(conn_id))
        return events

    # ── Pipeline event handlers ───────────────────

    def on_assistant_text(
        self,
        conn_id: str,
        event: AssistantTextEvent,
        *,
        wait_for_pending_reopen: bool = True,
    ) -> list[ServerEvent] | None:
        """Handle assistant_text: emit transcript and/or tool-call events."""
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
                logger.debug("Dropping stale assistant text for turn=%s rev=%s", event.turn_id, event.turn_revision)
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
                pending = None
                mapped_output = st.assistant_output_items.get(part.ordinal) if part.ordinal is not None else None
                if (
                    mapped_output is None
                    and part.ordinal is None
                    and wants_audio
                    and st.pending_assistant_item_id is not None
                    and st.pending_assistant_output_index is not None
                    and any(
                        candidate["output_index"] == st.pending_assistant_output_index and not candidate["parts"]
                        for candidate in st.pending_text_outputs
                    )
                ):
                    # Compatibility path for untagged audio that reserved its
                    # item before the transcript side channel arrived.
                    mapped_output = (st.pending_assistant_item_id, st.pending_assistant_output_index)
                if mapped_output is not None:
                    item_id, output_idx = mapped_output
                    pending = next(
                        (candidate for candidate in st.pending_text_outputs if candidate["output_index"] == output_idx),
                        None,
                    )
                    rejoined_earlier_output = st.current_output_index != output_idx
                    st.current_item_id = item_id
                    st.current_output_index = output_idx
                    st.current_output_kind = "text"
                else:
                    rejoined_earlier_output = False
                    output_idx, item_id = self._output_part_context(conn_id, "text")
                if pending is None:
                    if st.pending_text_outputs and st.pending_text_outputs[-1]["output_index"] == output_idx:
                        pending = st.pending_text_outputs[-1]
                    else:
                        st.pending_text_outputs.append({"item_id": item_id, "output_index": output_idx, "parts": []})
                        pending = st.pending_text_outputs[-1]
                if part.ordinal is not None:
                    st.assistant_output_items[part.ordinal] = (item_id, output_idx)
                assert pending is not None
                parts = pending["parts"]
                assert isinstance(parts, list)
                if wants_audio:
                    if st.pending_assistant_item_id is None:
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
                if not rejoined_earlier_output:
                    st.last_item_id = item_id
            elif isinstance(part, AssistantToolCallPart):
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
        st.response_text_complete = True
        events: list[ServerEvent] = (
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
