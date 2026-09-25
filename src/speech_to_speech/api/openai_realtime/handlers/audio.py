from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Literal

from openai.types.realtime import (
    ConversationItemInputAudioTranscriptionFailedEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferCommittedEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    RealtimeConversationItemUserMessage,
    RealtimeErrorEvent,
    ResponseAudioDeltaEvent,
    ResponseCreatedEvent,
)
from openai.types.realtime.realtime_conversation_item_user_message import Content

from speech_to_speech.api.openai_realtime.handlers.base import RealtimeBaseHandler
from speech_to_speech.api.openai_realtime.input_state import (
    InputItemState,
    InputTranscriptionTerminal,
    PendingInputTerminal,
)
from speech_to_speech.api.openai_realtime.utils import StreamingPcm16Resampler
from speech_to_speech.pipeline.events import SpeechStartedEvent, SpeechStoppedEvent

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.service import ServerEvent

logger = logging.getLogger(__name__)

PIPELINE_SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512
BYTES_PER_SAMPLE = 2
CHUNK_SIZE_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE


class AudioHandler(RealtimeBaseHandler):
    """Owns inbound audio decoding/chunking and outbound audio encoding."""

    def _start_input_item(
        self,
        conn_id: str,
        *,
        turn_id: str | None = None,
        turn_revision: int | None = None,
        preserve_active_response: bool = False,
    ) -> str:
        response = self._service.response
        st = self._state(conn_id)
        if not preserve_active_response:
            item_id = response._start_item(conn_id)
        else:
            response_item_id = st.current_item_id
            response_content_index = st.content_index
            item_id = response._start_item(conn_id)
            st.current_item_id = response_item_id
            st.content_index = response_content_index
        st.current_input_item_id = item_id
        st.input_items[item_id] = InputItemState()
        if turn_id is not None:
            st.input_item_by_turn_revision[(turn_id, turn_revision)] = item_id
        return item_id

    def _reuse_input_item(
        self,
        conn_id: str,
        item_id: str,
        *,
        turn_id: str,
        turn_revision: int | None,
        preserve_active_response: bool,
    ) -> str:
        """Move an incomplete input item to the latest speculative revision."""
        st = self._state(conn_id)
        if not preserve_active_response:
            st.current_item_id = item_id
            st.content_index = 0
        st.current_input_item_id = item_id
        st.input_item_by_turn_revision = {
            turn: tracked_item_id
            for turn, tracked_item_id in st.input_item_by_turn_revision.items()
            if tracked_item_id != item_id
        }
        st.input_item_by_turn_revision[(turn_id, turn_revision)] = item_id
        return item_id

    def release_input_item_state(
        self,
        conn_id: str,
        turn_id: str | None,
        turn_revision: int | None,
    ) -> None:
        """Close a direct-audio item without publishing a transcription terminal."""
        item_id = self._input_item_id(conn_id, turn_id, turn_revision)
        if item_id is None:
            return
        self.hold_input_terminal(conn_id, item_id, turn_id, turn_revision)

    def _release_input_item_state_by_id(self, conn_id: str, item_id: str) -> None:
        st = self._state(conn_id)
        st.input_item_by_turn_revision = {
            turn: tracked_item_id
            for turn, tracked_item_id in st.input_item_by_turn_revision.items()
            if tracked_item_id != item_id
        }
        st.input_items.pop(item_id, None)
        if st.current_input_item_id == item_id:
            st.current_input_item_id = None

    # ── Committed input-item lifecycle ─────────────

    def _pending_terminal(
        self,
        conn_id: str,
        item_id: str,
        turn_id: str | None,
        turn_revision: int | None,
    ) -> PendingInputTerminal:
        """Return the held lifecycle for an item, moved to its latest revision."""
        st = self._state(conn_id)
        pending = st.pending_input_terminals.get(item_id)
        if pending is None:
            pending = PendingInputTerminal()
            st.pending_input_terminals[item_id] = pending
        pending.turn_id = turn_id
        pending.turn_revision = turn_revision
        return pending

    def hold_input_terminal(
        self,
        conn_id: str,
        item_id: str,
        turn_id: str | None,
        turn_revision: int | None,
        terminal: InputTranscriptionTerminal | None = None,
    ) -> None:
        """Close one input item, holding its terminal until the turn commits."""
        pending = self._pending_terminal(conn_id, item_id, turn_id, turn_revision)
        if terminal is not None:
            pending.transcription = terminal
        pending.input_closed = True

    def resolve_input_terminals(
        self,
        conn_id: str,
        *,
        started_turn_id: str | None = None,
    ) -> list[ServerEvent]:
        """Publish or discard held input lifecycle events for settled turns.

        ``started_turn_id`` names the turn whose speech is beginning. Every
        other turn's held events describe a user item the pipeline can no
        longer revise, so they are published before the new turn opens.
        """
        st = self._state(conn_id)
        events: list[ServerEvent] = []
        for item_id, pending in list(st.pending_input_terminals.items()):
            disposition = self._input_terminal_disposition(conn_id, pending, started_turn_id)
            if disposition == "hold":
                continue
            if disposition == "publish":
                events.extend(self._publish_input_terminal(conn_id, item_id, pending))
                continue
            logger.debug(
                "Discarding superseded input lifecycle for item=%s turn=%s rev=%s",
                item_id,
                pending.turn_id,
                pending.turn_revision,
            )
            st.pending_input_terminals.pop(item_id, None)
        return events

    def _input_terminal_disposition(
        self,
        conn_id: str,
        pending: PendingInputTerminal,
        started_turn_id: str | None,
    ) -> Literal["publish", "hold", "discard"]:
        turns = self._service.speculative_turns
        if turns is None or pending.turn_id is None:
            return "publish"
        if not turns.is_latest(pending.turn_id, pending.turn_revision):
            # A later revision replaced this one. The client never learned that
            # the item stopped, so nothing has to be retracted.
            return "discard"
        if started_turn_id is not None and pending.turn_id != started_turn_id:
            return "publish"
        st = self._state(conn_id)
        unanswered = pending.input_closed and not (st.in_response or st.response_pending)
        if unanswered or isinstance(pending.transcription, ConversationItemInputAudioTranscriptionFailedEvent):
            # No output will commit this turn: its transcription failed, was
            # empty, or its response ended without output. The item cannot
            # reopen once published, so commit after the same reopen gate
            # used for accepted output.
            committed = turns.try_commit_if_latest_after_reopen_grace(pending.turn_id, pending.turn_revision)
            if committed is None:
                return "hold"
            return "publish" if committed else "discard"
        if turns.is_committed(pending.turn_id, pending.turn_revision):
            return "publish"
        return "hold"

    def _publish_input_terminal(
        self,
        conn_id: str,
        item_id: str,
        pending: PendingInputTerminal,
    ) -> list[ServerEvent]:
        """Emit one item's held events, releasing it once the input is closed."""
        st = self._state(conn_id)
        events: list[ServerEvent] = []
        if pending.speech_stopped is not None:
            events.append(pending.speech_stopped)
            pending.speech_stopped = None
            events.append(
                InputAudioBufferCommittedEvent(
                    type="input_audio_buffer.committed",
                    event_id=self._next_event_id(),
                    item_id=item_id,
                    previous_item_id=st.last_item_id,
                )
            )
            events.append(
                self._service.conversation._ack_item(
                    conn_id,
                    RealtimeConversationItemUserMessage(
                        id=item_id,
                        object="realtime.item",
                        type="message",
                        role="user",
                        status="completed",
                        content=[Content(type="input_audio")],
                    ),
                )
            )
        if pending.transcription is not None:
            events.append(pending.transcription)
            pending.transcription = None
        if pending.input_closed:
            st.pending_input_terminals.pop(item_id, None)
            self._release_input_item_state_by_id(conn_id, item_id)
        return events

    def handle_audio_append(self, conn_id: str, event: InputAudioBufferAppendEvent) -> list[bytes]:
        """Decode base64 audio, resample to pipeline rate, and split into 512-sample PCM16 chunks for the VAD."""
        try:
            pcm_bytes = base64.b64decode(event.audio)
        except Exception as e:
            logger.error(f"Base64 decode error: {e}")
            return []

        st = self._state(conn_id)

        audio_cfg = st.runtime_config.session.audio
        if audio_cfg is not None and audio_cfg.input is not None:
            client_in_rate = getattr(audio_cfg.input.format, "rate", None) or PIPELINE_SAMPLE_RATE
        else:
            client_in_rate = PIPELINE_SAMPLE_RATE
        return self.append_pcm(conn_id, pcm_bytes, client_in_rate)

    def append_pcm(self, conn_id: str, pcm_bytes: bytes, src_rate: int) -> list[bytes]:
        """Resample raw PCM16 to the pipeline rate and split into 512-sample chunks for the VAD.

        Shared by both transports: the WebSocket route feeds it decoded
        ``input_audio_buffer.append`` payloads, the WebRTC transport feeds it
        PCM decoded from inbound media-track frames. Keeps the sub-chunk
        remainder and the commit bookkeeping (``audio_buffer_has_data``) in
        one place regardless of how audio arrives.
        """
        st = self._state(conn_id)
        if src_rate != PIPELINE_SAMPLE_RATE:
            # The filter state carries over between appends, so chunk boundaries
            # leave no trace in the audio the VAD and the STT receive.
            if st.input_audio_resampler_rate != src_rate:
                st.input_audio_resampler = StreamingPcm16Resampler(src_rate, PIPELINE_SAMPLE_RATE)
                st.input_audio_resampler_rate = src_rate
            pcm_bytes = st.input_audio_resampler.push(pcm_bytes)

        pcm_bytes = st.audio_remainder + pcm_bytes

        chunks = []
        for i in range(0, len(pcm_bytes), CHUNK_SIZE_BYTES):
            chunk = pcm_bytes[i : i + CHUNK_SIZE_BYTES]
            if len(chunk) == CHUNK_SIZE_BYTES:
                chunks.append(chunk)
            else:
                st.audio_remainder = chunk
                break
        else:
            st.audio_remainder = b""

        if chunks:
            st.audio_buffer_has_data = True
        return chunks

    def handle_audio_commit(self, conn_id: str) -> tuple[list[bytes], RealtimeErrorEvent | None]:
        """Finish resampling before committing the buffered input audio."""
        st = self._state(conn_id)
        resampler = st.input_audio_resampler
        pending_samples = resampler.pending_output_samples if resampler is not None else 0
        if (
            not st.audio_buffer_has_data
            and len(st.audio_remainder) + pending_samples * BYTES_PER_SAMPLE < CHUNK_SIZE_BYTES
        ):
            return [], self.make_error(
                message="Input audio buffer is empty, nothing to commit.",
                _type="input_audio_buffer_commit_empty",
            )

        chunks = self.append_pcm(conn_id, resampler.flush(), PIPELINE_SAMPLE_RATE) if resampler is not None else []
        if not st.audio_buffer_has_data:
            return chunks, self.make_error(
                message="Input audio buffer is empty, nothing to commit.",
                _type="input_audio_buffer_commit_empty",
            )
        st.input_audio_resampler = None
        st.input_audio_resampler_rate = None
        st.audio_buffer_has_data = False
        logger.debug("Audio buffer committed")
        return chunks, None

    # ── Pipeline event handlers ────────────────────

    def on_speech_started(self, conn_id: str, event: SpeechStartedEvent) -> list[ServerEvent]:
        """Handle VAD speech_started: publish the settled turn, cancel an active
        response if interrupts are enabled, and start or reopen an input item."""
        response = self._service.response
        st = self._state(conn_id)
        # A turn other than the one now starting can no longer be revised, so
        # its user item becomes permanent before the new turn opens.
        events: list[ServerEvent] = self.resolve_input_terminals(conn_id, started_turn_id=event.turn_id)
        interrupt_enabled = event.interrupt_response and st.runtime_config.interrupt_response_enabled
        if st.in_response and interrupt_enabled:
            events.extend(response.finish_response(conn_id, status="cancelled", reason="turn_detected"))
        if interrupt_enabled:
            response.discard_tool_followup_prefetch(conn_id)
            st.generation_done_tool_calls.clear()
            st.completed_tool_response_keys.clear()
        is_reopen = bool(event.reopened and event.turn_id is not None and event.turn_id == st.speculative_turn_id)
        preserve_active_response = st.in_response
        previous_input_item_id = (
            st.input_item_by_turn_revision.get((event.turn_id, st.speculative_turn_revision))
            if is_reopen and event.turn_id is not None
            else None
        )
        previous_input_item = st.input_items.get(previous_input_item_id) if previous_input_item_id is not None else None
        if previous_input_item_id is not None and previous_input_item is not None:
            assert event.turn_id is not None
            input_item_id = self._reuse_input_item(
                conn_id,
                previous_input_item_id,
                turn_id=event.turn_id,
                turn_revision=event.turn_revision,
                preserve_active_response=preserve_active_response,
            )
        else:
            input_item_id = self._start_input_item(
                conn_id,
                turn_id=event.turn_id,
                turn_revision=event.turn_revision,
                preserve_active_response=preserve_active_response,
            )
        if not is_reopen:
            st.response_usage.turns += 1
        st.speculative_turn_id = event.turn_id
        st.speculative_turn_revision = event.turn_revision
        # A reopened revision is still the same externally open speech item.
        # Keep cancellation and revision bookkeeping above, but do not reset
        # a generic client's audio onset with another speech_started event.
        if previous_input_item is not None:
            return events
        events.append(
            InputAudioBufferSpeechStartedEvent(
                type="input_audio_buffer.speech_started",
                event_id=self._next_event_id(),
                audio_start_ms=event.audio_start_ms,
                item_id=input_item_id,
            )
        )
        return events

    def on_speech_stopped(self, conn_id: str, event: SpeechStoppedEvent) -> list[ServerEvent]:
        """Handle VAD speech_stopped: record duration and hold the stop candidate.

        A speculative pause is not yet the end of the user's turn: speech can
        resume and keep the same item open. The protocol event therefore waits
        until the turn commits, so the client sees exactly one stop per item
        and it is the one that precedes the item's transcription.
        """
        st = self._state(conn_id)
        item_id = self._input_item_id(conn_id, event.turn_id, event.turn_revision)
        if item_id is None:
            logger.debug(
                "Ignoring speech stop for unknown turn=%s rev=%s",
                event.turn_id,
                event.turn_revision,
            )
            return []
        if event.duration_s:
            st.input_audio_duration_s = event.duration_s
            input_item = st.input_items.get(item_id)
            if input_item is not None:
                input_item.audio_duration_s = event.duration_s
        pending = self._pending_terminal(conn_id, item_id, event.turn_id, event.turn_revision)
        pending.speech_stopped = InputAudioBufferSpeechStoppedEvent(
            type="input_audio_buffer.speech_stopped",
            event_id=self._next_event_id(),
            audio_end_ms=event.audio_end_ms,
            item_id=item_id,
        )
        return self.resolve_input_terminals(conn_id)

    # ── Outbound audio encoding ──────────────────

    def begin_audio_response(
        self,
        conn_id: str,
        response_key: str | None = None,
    ) -> tuple[str, str, list[ServerEvent]]:
        """Ensure a response exists for outbound audio, emitting ResponseCreated once.

        When ``handle_response_create`` already allocated the response,
        ``current_response_id`` is set and no duplicate event is emitted.
        For the implicit-response path (VAD -> STT -> LLM -> TTS, no
        ``response.create``), ``current_response_id`` is still ``None``
        and the event is emitted here on the first audio chunk.

        Returns ``(response_id, item_id, events)``. Shared by both
        transports: the WebSocket path appends the base64 audio delta to the
        returned events, the WebRTC path sends only the bookkeeping events
        over the data channel while audio travels on the media track.
        """
        response = self._service.response
        st = self._state(conn_id)

        # Accepted assistant audio commits the turn it answers, so the user
        # item it replies to must reach the client first.
        events: list[ServerEvent] = self.resolve_input_terminals(conn_id)
        need_created = st.current_response_id is None
        resp_id, item_id = response._ensure_response(conn_id, response_key)
        if need_created:
            events.append(
                ResponseCreatedEvent(
                    type="response.created",
                    event_id=self._next_event_id(),
                    response=response._build_response(conn_id, "in_progress"),
                )
            )
        self._service._apply_pending_token_usage(conn_id, response_key)
        return resp_id, item_id, events

    def begin_audio_output(
        self,
        conn_id: str,
        response_key: str | None = None,
    ) -> tuple[str, str, int, list[ServerEvent]]:
        """Ensure an audio response and reserve its assistant output identity."""
        resp_id, item_id, events = self.begin_audio_response(conn_id, response_key)
        st = self._state(conn_id)
        assistant_item_id, output_index = self._service.response._ensure_assistant_output_item(
            conn_id,
            item_id,
        )
        pending = next(item for item in st.pending_text_outputs if int(item["output_index"]) == output_index)
        events.extend(self._service.response._begin_message_output(conn_id, pending, wants_audio=True))
        st.audio_output_started = True
        return resp_id, assistant_item_id, output_index, events

    def encode_audio_chunk(
        self,
        conn_id: str,
        audio: bytes,
        response_key: str | None = None,
    ) -> list[ServerEvent]:
        """Encode a raw PCM audio chunk as a base64 delta event for the WebSocket transport."""
        response = self._service.response
        st = self._state(conn_id)

        resp_id, assistant_item_id, assistant_output_index, events = self.begin_audio_output(
            conn_id,
            response_key,
        )
        rp = st.current_response_params
        client_out_rate = None
        if rp and rp.audio and rp.audio.output and rp.audio.output.format:
            client_out_rate = getattr(rp.audio.output.format, "rate", None)
        if client_out_rate is None:
            audio_cfg = st.runtime_config.session.audio
            if audio_cfg is not None and audio_cfg.output is not None:
                client_out_rate = getattr(audio_cfg.output.format, "rate", None) or PIPELINE_SAMPLE_RATE
            else:
                client_out_rate = PIPELINE_SAMPLE_RATE
        if client_out_rate != PIPELINE_SAMPLE_RATE:
            resampler_key = (resp_id, assistant_item_id, client_out_rate)
            if st.output_audio_resampler_key != resampler_key:
                st.output_audio_resampler = StreamingPcm16Resampler(PIPELINE_SAMPLE_RATE, client_out_rate)
                st.output_audio_resampler_key = resampler_key
            audio = st.output_audio_resampler.push(audio)
        else:
            st.output_audio_resampler = None
            st.output_audio_resampler_key = None
        if not audio:
            return events
        b64 = base64.b64encode(audio).decode("ascii")
        events.append(
            ResponseAudioDeltaEvent(
                type="response.output_audio.delta",
                event_id=self._next_event_id(),
                content_index=response._next_content_index(conn_id),
                delta=b64,
                item_id=assistant_item_id,
                output_index=assistant_output_index,
                response_id=resp_id,
            )
        )
        return events
