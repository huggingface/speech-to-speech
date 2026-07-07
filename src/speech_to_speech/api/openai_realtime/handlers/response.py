from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from openai.types.realtime import (
    RealtimeResponse,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseCreatedEvent,
    ResponseCreateEvent,
    ResponseDoneEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from openai.types.realtime.realtime_response import Audio, AudioOutput
from openai.types.realtime.realtime_response_status import RealtimeResponseStatus
from openai.types.realtime.realtime_response_usage import RealtimeResponseUsage

from speech_to_speech.api.openai_realtime.handlers.base import RealtimeBaseHandler
from speech_to_speech.LLM.chat import ChatItemError
from speech_to_speech.pipeline.events import AssistantTextEvent
from speech_to_speech.pipeline.messages import GenerateResponseRequest
from speech_to_speech.utils.utils import _generate_id, is_out_of_band, response_wants_audio

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.service import ServerEvent, _ResponseStatus, _StatusReason

logger = logging.getLogger(__name__)


class ResponseHandler(RealtimeBaseHandler):
    """Owns the response lifecycle: create, cancel, finish, and ID management."""

    # ── ID / state helpers ────────────────────────

    def _ensure_response(self, conn_id: str) -> tuple[str, str]:
        """Ensure a response and output item exist, creating them if needed."""
        st = self._state(conn_id)
        if st.current_response_id is None:
            st.current_response_id = _generate_id("resp")
            self._start_item(conn_id)
            st.in_response = True
        st.response_pending = False
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
        st.current_response_id = None
        st.current_item_id = None
        st.content_index = 0
        st.in_response = False
        st.response_pending = False
        st.current_response_params = None
        st.pending_output_text_parts = []

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

    def _next_content_index(self, conn_id: str) -> int:
        """Return the current content index and advance it."""
        st = self._state(conn_id)
        idx = st.content_index
        st.content_index += 1
        return idx

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
            status_details = RealtimeResponseStatus(type=status, reason=reason)  # type: ignore[arg-type]

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
            usage=RealtimeResponseUsage(
                input_tokens=st.response_usage.input_tokens,
                output_tokens=st.response_usage.output_tokens,
                total_tokens=st.response_usage.input_tokens + st.response_usage.output_tokens,
            ),
        )

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
        if st.in_response:
            return self.make_error(
                message="Cannot create response while another response is in progress.",
                _type="conversation_already_has_active_response",
            )

        out_of_band = is_out_of_band(event.response)

        # In-band: response.input items are added to the default conversation here so
        # they appear in history. Out-of-band: leave the default conversation untouched —
        # the input rides along on the request and seeds a throwaway chat in the LM.
        if not out_of_band and event.response and event.response.input:
            for input_item in event.response.input:
                try:
                    self._service.conversation._append_item(conn_id, input_item)
                except ChatItemError as exc:
                    return self.make_error(message=str(exc), _type="invalid_input_item")

        st.in_response = True
        st.response_pending = False

        st.current_response_params = event.response
        st.current_response_id = _generate_id("resp")
        self._start_item(conn_id)

        cfg = st.runtime_config
        queue = self._queue(conn_id)
        if queue:
            # Out-of-band responses carry no turn identity: a null turn_id makes every
            # speculative-turn staleness gate treat them as always-latest, so a new user
            # turn mid-generation can never silently drop their output.
            queue.put(
                GenerateResponseRequest(
                    runtime_config=cfg,
                    response=event.response,
                    turn_id=None if out_of_band else st.speculative_user_turn_id,
                    turn_revision=None if out_of_band else st.speculative_user_turn_revision,
                    speech_stopped_at_s=None if out_of_band else st.speculative_user_speech_stopped_at_s,
                )
            )
        logger.debug("response.create received, LLM generation triggered")
        return ResponseCreatedEvent(
            type="response.created",
            event_id=self._next_event_id(),
            response=self._build_response(conn_id, "in_progress"),
        )

    def handle_response_cancel(self, conn_id: str) -> list[ServerEvent]:
        """Cancel the in-progress response and re-enable listening."""
        events = self.finish_response(conn_id, status="cancelled", reason="client_cancelled")
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
    ) -> list[ServerEvent]:
        """Close the current response (audio/text done + response done).

        Audio responses emit ``response.output_audio.done`` for any terminal
        status. Text-only responses emit a single ``response.output_text.done``
        carrying the full streamed text, but only on ``status="completed"`` —
        a cancelled or failed text response sends no audio, so it just closes
        with ``response.done``.
        """
        st = self._state(conn_id)
        events: list[ServerEvent] = []
        if st.in_response:
            resp_id, item_id = self._ensure_response(conn_id)
            if response_wants_audio(st.current_response_params):
                events.append(
                    ResponseAudioDoneEvent(
                        type="response.output_audio.done",
                        event_id=self._next_event_id(),
                        content_index=0,
                        item_id=item_id,
                        output_index=0,
                        response_id=resp_id,
                    )
                )
            elif status == "completed" and st.pending_output_text_parts:
                events.append(
                    ResponseTextDoneEvent(
                        type="response.output_text.done",
                        event_id=self._next_event_id(),
                        content_index=0,
                        item_id=item_id,
                        output_index=0,
                        response_id=resp_id,
                        text="".join(st.pending_output_text_parts),
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
        resp_id, item_id = self._ensure_response(conn_id)
        st.last_item_id = item_id
        output_idx = 0
        if event.text:
            if response_wants_audio(st.current_response_params):
                events.append(
                    ResponseAudioTranscriptDoneEvent(
                        type="response.output_audio_transcript.done",
                        event_id=self._next_event_id(),
                        content_index=0,
                        item_id=item_id,
                        output_index=output_idx,
                        response_id=resp_id,
                        transcript=event.text,
                    )
                )
            else:
                # Stream the delta now; the matching response.output_text.done is
                # emitted once at close in finish_response, carrying the per-chunk
                # parts collected here, space-joined ("" when there were none).
                st.pending_output_text_parts.append(event.text)
                events.append(
                    ResponseTextDeltaEvent(
                        type="response.output_text.delta",
                        event_id=self._next_event_id(),
                        content_index=0,
                        item_id=item_id,
                        output_index=output_idx,
                        response_id=resp_id,
                        delta=event.text,
                    )
                )
            output_idx += 1
        if event.tools:
            st.response_usage.tool_calls += len(event.tools)
            for tool in event.tools:
                events.append(
                    ResponseFunctionCallArgumentsDoneEvent(
                        type="response.function_call_arguments.done",
                        event_id=self._next_event_id(),
                        call_id=tool.call_id,
                        name=tool.name,
                        arguments=tool.arguments,
                        item_id=item_id,
                        output_index=output_idx,
                        response_id=resp_id,
                    )
                )
                output_idx += 1
        return events
