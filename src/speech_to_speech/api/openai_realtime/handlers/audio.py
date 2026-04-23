from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING

from openai.types.realtime import (
    InputAudioBufferAppendEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    RealtimeErrorEvent,
    ResponseAudioDeltaEvent,
    ResponseCreatedEvent,
)

from speech_to_speech.api.openai_realtime.handlers.base import RealtimeBaseHandler
from speech_to_speech.api.openai_realtime.utils import resample
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
        pcm_bytes = resample(pcm_bytes, client_in_rate, PIPELINE_SAMPLE_RATE)

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

    def handle_audio_commit(self, conn_id: str) -> RealtimeErrorEvent | None:
        """Commit the audio buffer. Returns an error if no audio was appended."""
        st = self._state(conn_id)
        if not st.audio_buffer_has_data:
            return self.make_error(
                message="Input audio buffer is empty, nothing to commit.",
                _type="input_audio_buffer_commit_empty",
            )
        st.audio_buffer_has_data = False
        logger.debug("Audio buffer committed")
        return None

    # ── Pipeline event handlers ────────────────────

    def on_speech_started(self, conn_id: str, event: SpeechStartedEvent) -> list[ServerEvent]:
        """Handle VAD speech_started: cancel active response if interrupts enabled, start new input item."""
        response = self._service.response
        events: list[ServerEvent] = []
        st = self._state(conn_id)
        if st.in_response and st.runtime_config.interrupt_response_enabled:
            events.extend(response.finish_audio_response(conn_id, status="cancelled", reason="turn_detected"))
        input_item_id = response._start_item(conn_id)
        st.last_item_id = input_item_id
        st.response_usage.turns += 1
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
        """Handle VAD speech_stopped: record duration and emit stopped event."""
        response = self._service.response
        if event.duration_s:
            self._state(conn_id).input_audio_duration_s = event.duration_s
        return [
            InputAudioBufferSpeechStoppedEvent(
                type="input_audio_buffer.speech_stopped",
                event_id=self._next_event_id(),
                audio_end_ms=event.audio_end_ms,
                item_id=response._current_item_id(conn_id),
            )
        ]

    # ── Outbound audio encoding ──────────────────

    def encode_audio_chunk(self, conn_id: str, audio: bytes) -> list[ServerEvent]:
        """Encode a raw PCM audio chunk, emitting ResponseCreated on the first chunk.

        When ``handle_response_create`` already allocated the response,
        ``current_response_id`` is set and no duplicate event is emitted.
        For the implicit-response path (VAD -> STT -> LLM -> TTS, no
        ``response.create``), ``current_response_id`` is still ``None``
        and the event is emitted here on the first audio chunk.
        """
        response = self._service.response
        st = self._state(conn_id)

        events: list[ServerEvent] = []
        need_created = st.current_response_id is None
        resp_id, item_id = response._ensure_response(conn_id)
        if need_created:
            events.append(
                ResponseCreatedEvent(
                    type="response.created",
                    event_id=self._next_event_id(),
                    response=response._build_response(conn_id, "in_progress"),
                )
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
        audio = resample(audio, PIPELINE_SAMPLE_RATE, client_out_rate)
        b64 = base64.b64encode(audio).decode("ascii")
        events.append(
            ResponseAudioDeltaEvent(
                type="response.output_audio.delta",
                event_id=self._next_event_id(),
                content_index=response._next_content_index(conn_id),
                delta=b64,
                item_id=item_id,
                output_index=0,
                response_id=resp_id,
            )
        )
        return events
