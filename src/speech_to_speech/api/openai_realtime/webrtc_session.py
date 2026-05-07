from __future__ import annotations

import asyncio
import json
import logging
from queue import Empty, Queue
from threading import Event as ThreadingEvent
from typing import TYPE_CHECKING, Any, Callable, Coroutine

import numpy as np
from aiortc import RTCPeerConnection
from aiortc.mediastreams import AudioStreamTrack, MediaStreamError

from speech_to_speech.api.openai_realtime.utils import resample
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.control import is_control_message
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, PIPELINE_END
from speech_to_speech.pipeline.queue_types import AudioInItem, AudioOutItem

if TYPE_CHECKING:
    import av
    from speech_to_speech.api.openai_realtime.service import RealtimeService, ServerEvent

logger = logging.getLogger(__name__)

PIPELINE_SAMPLE_RATE = 16_000
WEBRTC_SAMPLE_RATE = 48_000
WEBRTC_FRAME_SAMPLES = 960   # 20 ms @ 48 kHz
PIPELINE_CHUNK_BYTES = 512 * 2  # 512 int16 samples


def _to_audio_bytes(chunk: Any) -> bytes:
    if isinstance(chunk, np.ndarray) or hasattr(chunk, "tobytes"):
        return chunk.tobytes()
    return chunk  # type: ignore[return-value]


class PipelineAudioTrack(AudioStreamTrack):
    """Outbound audio track: reads PCM from output_queue, delivers av.AudioFrame at 48 kHz."""

    kind = "audio"

    def __init__(
        self,
        output_queue: Queue[AudioOutItem],
        on_response_done: Callable[[], Coroutine[Any, Any, None]],
        on_pipeline_end: Callable[[], None],
        cancel_scope: CancelScope | None = None,
        response_playing: ThreadingEvent | None = None,
        should_listen: ThreadingEvent | None = None,
    ) -> None:
        super().__init__()
        self._output_queue = output_queue
        self._on_response_done = on_response_done
        self._on_pipeline_end = on_pipeline_end
        self._cancel_scope = cancel_scope
        self._response_playing = response_playing
        self._should_listen = should_listen
        self._pcm_buffer: bytearray = bytearray()
        self._pts: int = 0

    async def recv(self) -> av.AudioFrame:
        import av as _av

        needed = WEBRTC_FRAME_SAMPLES * 2  # bytes

        while len(self._pcm_buffer) < needed:
            try:
                chunk = await asyncio.get_running_loop().run_in_executor(
                    None, self._output_queue.get, True, 0.02
                )
            except Empty:
                return self._silence_frame(_av)

            if isinstance(chunk, bytes) and chunk == PIPELINE_END:
                self._on_pipeline_end()
                return self._silence_frame(_av)

            if isinstance(chunk, bytes) and chunk == AUDIO_RESPONSE_DONE:
                self._handle_response_done()
                return self._silence_frame(_av)

            if is_control_message(chunk):
                continue

            if self._cancel_scope and self._cancel_scope.discarding:
                continue

            raw = _to_audio_bytes(chunk)
            upsampled = resample(raw, PIPELINE_SAMPLE_RATE, WEBRTC_SAMPLE_RATE)
            self._pcm_buffer.extend(upsampled)

            if self._response_playing and not self._response_playing.is_set():
                self._response_playing.set()
                if self._should_listen:
                    self._should_listen.set()

        data = bytes(self._pcm_buffer[:needed])
        self._pcm_buffer = self._pcm_buffer[needed:]
        return self._make_frame(_av, data)

    def _handle_response_done(self) -> None:
        if self._cancel_scope:
            self._cancel_scope.response_done()
        if self._response_playing:
            self._response_playing.clear()
        if self._should_listen:
            self._should_listen.set()
        asyncio.ensure_future(self._on_response_done())

    def _silence_frame(self, av: Any) -> av.AudioFrame:
        samples = np.zeros(WEBRTC_FRAME_SAMPLES, dtype=np.int16)
        frame = av.AudioFrame.from_ndarray(samples[np.newaxis, :], format="s16", layout="mono")
        frame.sample_rate = WEBRTC_SAMPLE_RATE
        frame.pts = self._pts
        frame.time_base = f"1/{WEBRTC_SAMPLE_RATE}"
        self._pts += WEBRTC_FRAME_SAMPLES
        return frame

    def _make_frame(self, av: Any, data: bytes) -> av.AudioFrame:
        samples = np.frombuffer(data, dtype=np.int16)
        frame = av.AudioFrame.from_ndarray(samples[np.newaxis, :], format="s16", layout="mono")
        frame.sample_rate = WEBRTC_SAMPLE_RATE
        frame.pts = self._pts
        frame.time_base = f"1/{WEBRTC_SAMPLE_RATE}"
        self._pts += WEBRTC_FRAME_SAMPLES
        return frame


class WebRTCSession:
    """Manages one WebRTC peer connection: SDP negotiation, data channel, audio tracks."""

    def __init__(
        self,
        session_id: str,
        pc: RTCPeerConnection,
        service: RealtimeService,
        input_queue: Queue[AudioInItem],
        output_queue: Queue[AudioOutItem],
        on_closed: Callable[[str], None],
        on_cancel_response: Callable[[], None] | None = None,
        cancel_scope: CancelScope | None = None,
        response_playing: ThreadingEvent | None = None,
        should_listen: ThreadingEvent | None = None,
    ) -> None:
        self.session_id = session_id
        self._pc = pc
        self._service = service
        self._input_queue = input_queue
        self._on_closed = on_closed
        self._on_cancel_response = on_cancel_response
        self._cancel_scope = cancel_scope
        self._dc: Any = None
        self._closed = False
        self._audio_track = PipelineAudioTrack(
            output_queue=output_queue,
            on_response_done=self._finish_response_via_dc,
            on_pipeline_end=lambda: asyncio.ensure_future(self.close()),
            cancel_scope=cancel_scope,
            response_playing=response_playing,
            should_listen=should_listen,
        )

    def setup_handlers(self) -> None:
        """Wire aiortc event callbacks. Call before setRemoteDescription()."""

        @self._pc.on("datachannel")
        def on_datachannel(dc: Any) -> None:
            if dc.label != "oai-events":
                logger.warning("[WebRTC] Unexpected data channel: %s", dc.label)
                return
            self._dc = dc
            logger.info("[WebRTC] Data channel 'oai-events' received (session %s)", self.session_id)

            @dc.on("open")
            def on_dc_open() -> None:
                created = self._service.build_session_created(self.session_id)
                asyncio.ensure_future(self.send_events([created]))
                logger.info("[WebRTC] session.created sent (session %s)", self.session_id)

            @dc.on("message")
            def on_message(msg: str) -> None:
                asyncio.ensure_future(self._handle_dc_message(msg))

            @dc.on("close")
            def on_dc_close() -> None:
                logger.info("[WebRTC] Data channel closed (session %s)", self.session_id)
                asyncio.ensure_future(self.close())

        @self._pc.on("track")
        def on_track(track: Any) -> None:
            if track.kind == "audio":
                logger.info("[WebRTC] Inbound audio track received (session %s)", self.session_id)
                asyncio.ensure_future(self._consume_inbound_audio(track))

        @self._pc.on("connectionstatechange")
        async def on_connection_state_change() -> None:
            state = self._pc.connectionState
            logger.info("[WebRTC] Connection state: %s (session %s)", state, self.session_id)
            if state in ("failed", "closed", "disconnected"):
                await self.close()

    def add_outbound_track(self) -> None:
        self._pc.addTrack(self._audio_track)

    async def send_events(self, events: list[ServerEvent]) -> None:
        if self._dc is None or self._dc.readyState != "open":
            return
        for event in events:
            try:
                self._dc.send(json.dumps(event.model_dump()))
            except Exception as e:
                logger.error("[WebRTC] Data channel send error: %s", e)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._pc.close()
        except Exception as e:
            logger.warning("[WebRTC] Error closing peer connection: %s", e)
        self._on_closed(self.session_id)
        logger.info("[WebRTC] Session %s closed", self.session_id)

    async def _finish_response_via_dc(self) -> None:
        events = self._service.finish_audio_response(self.session_id)
        await self.send_events(events)
        logger.info("[WebRTC] response.done sent (session %s)", self.session_id)

    async def _handle_dc_message(self, msg: str) -> None:
        from openai.types.realtime import (
            ConversationItemCreateEvent,
            InputAudioBufferAppendEvent,
            InputAudioBufferCommitEvent,
            ResponseCancelEvent,
            ResponseCreateEvent,
            SessionUpdateEvent,
        )

        try:
            raw = json.loads(msg)
        except json.JSONDecodeError:
            logger.error("[WebRTC] Invalid JSON on data channel: %r", msg)
            return

        event = self._service.parse_client_event(raw)
        if event is None:
            await self.send_events([
                self._service.make_error(
                    f"Unknown or invalid event: {raw.get('type')}", "unknown_or_invalid_event"
                )
            ])
            return

        if isinstance(event, InputAudioBufferAppendEvent):
            # Audio must arrive via the media track in WebRTC mode
            await self.send_events([
                self._service.make_error(
                    "In WebRTC mode audio is sent via the media track, not via the data channel.",
                    "invalid_event_for_transport",
                )
            ])

        elif isinstance(event, SessionUpdateEvent):
            err = self._service.handle_session_update(self.session_id, event)
            if err:
                await self.send_events([err])

        elif isinstance(event, InputAudioBufferCommitEvent):
            err = self._service.handle_audio_commit(self.session_id)
            if err:
                await self.send_events([err])

        elif isinstance(event, ConversationItemCreateEvent):
            events = self._service.handle_conversation_item_create(self.session_id, event)
            await self.send_events(events)

        elif isinstance(event, ResponseCreateEvent):
            result = self._service.handle_response_create(self.session_id, event)
            if result:
                if result.type != "error" and self._cancel_scope:
                    self._cancel_scope.new_response()
                await self.send_events([result])

        elif isinstance(event, ResponseCancelEvent):
            was_active = self._service._state(self.session_id).in_response
            if was_active and self._on_cancel_response:
                self._on_cancel_response()
            events = self._service.handle_response_cancel(self.session_id)
            await self.send_events(events)

    async def _consume_inbound_audio(self, track: Any) -> None:
        """Read WebRTC audio frames (48 kHz) → resample → push 512-sample chunks to input_queue."""
        buffer = bytearray()

        while not self._closed:
            try:
                frame = await track.recv()
            except MediaStreamError:
                logger.info("[WebRTC] Inbound audio track ended (session %s)", self.session_id)
                break

            src_rate = frame.sample_rate or WEBRTC_SAMPLE_RATE

            # to_ndarray with format="s16p" gives shape (channels, samples); flatten for mono
            pcm_src = frame.to_ndarray(format="s16p").flatten().astype(np.int16).tobytes()

            pcm_16k = resample(pcm_src, src_rate, PIPELINE_SAMPLE_RATE) if src_rate != PIPELINE_SAMPLE_RATE else pcm_src
            buffer.extend(pcm_16k)

            rt_cfg = self._service._state(self.session_id).runtime_config
            while len(buffer) >= PIPELINE_CHUNK_BYTES:
                self._input_queue.put((bytes(buffer[:PIPELINE_CHUNK_BYTES]), rt_cfg))
                buffer = buffer[PIPELINE_CHUNK_BYTES:]
