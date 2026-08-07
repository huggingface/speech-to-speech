"""WebRTC transport for the OpenAI Realtime API emulation.

Requires the ``webrtc`` extra (aiortc). Audio travels over RTP media tracks
(Opus at 48 kHz, resampled to/from the 16 kHz pipeline rate); all JSON events
use the same protocol as the WebSocket transport, carried on the
``oai-events`` data channel.

``WebRTCSession`` subclasses ``SessionTransport`` from ``transports``,
so the per-unit send loop in ``websocket_router`` drives it
exactly like a WebSocket session: it stays the sole consumer of the pipeline
output queues, and this module only turns delivered PCM into paced RTP frames.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Awaitable
from fractions import Fraction
from typing import TYPE_CHECKING, Callable, Optional

import av
import numpy as np
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack

from speech_to_speech.api.openai_realtime.service import PIPELINE_SAMPLE_RATE
from speech_to_speech.api.openai_realtime.transports import SessionTransport

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.service import RealtimeService, ServerEvent

logger = logging.getLogger(__name__)

WEBRTC_SAMPLE_RATE = 48_000
AUDIO_PTIME = 0.02  # 20 ms frames
WEBRTC_FRAME_SAMPLES = int(WEBRTC_SAMPLE_RATE * AUDIO_PTIME)
DATA_CHANNEL_LABEL = "oai-events"
ICE_SERVERS_ENV = "SPEECH_TO_SPEECH_ICE_SERVERS"
ICE_GATHERING_TIMEOUT_S = 5.0
# How long a negotiated session may sit without the peer connection reaching
# "connected" before we release its pipeline unit. Without this, a client that
# receives the SDP answer and never completes ICE would hold the unit forever.
CONNECT_TIMEOUT_S = 30.0


def rtc_configuration_from_env() -> Optional[RTCConfiguration]:
    """Build an RTCConfiguration from the SPEECH_TO_SPEECH_ICE_SERVERS env var.

    The variable holds a JSON list of RTCIceServer kwargs, e.g.
    ``[{"urls": "stun:stun.example.com:3478"},
       {"urls": "turn:turn.example.com", "username": "u", "credential": "c"}]``.
    Returns None (aiortc defaults) when unset or invalid.
    """
    raw = os.environ.get(ICE_SERVERS_ENV)
    if not raw:
        return None
    try:
        entries = json.loads(raw)
        servers = [RTCIceServer(**entry) for entry in entries]
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Ignoring invalid {ICE_SERVERS_ENV}: {e}")
        return None
    return RTCConfiguration(iceServers=servers)


class PcmResampler:
    """Stateful mono/s16 resampler around av.AudioResampler.

    One instance per direction per session: the libswresample filter state
    carries across calls, so 20 ms frames resample without the boundary
    artifacts a stateless per-chunk resample would introduce. Also downmixes
    multi-channel input (browser Opus is typically stereo) to mono.
    """

    def __init__(self, target_rate: int) -> None:
        self._resampler = av.AudioResampler(format="s16", layout="mono", rate=target_rate)
        self._pts = 0

    def resample_frame(self, frame: av.AudioFrame) -> bytes:
        out = bytearray()
        for resampled in self._resampler.resample(frame):
            out += resampled.to_ndarray().tobytes()
        return bytes(out)

    def resample_pcm(self, pcm: bytes, src_rate: int) -> bytes:
        samples = np.frombuffer(pcm, dtype=np.int16)
        frame = av.AudioFrame.from_ndarray(samples[np.newaxis, :], format="s16", layout="mono")
        frame.sample_rate = src_rate
        frame.pts = self._pts
        frame.time_base = Fraction(1, src_rate)
        self._pts += samples.shape[0]
        return self.resample_frame(frame)


class PipelineAudioTrack(MediaStreamTrack):
    """Outbound audio track: paced 20 ms 48 kHz frames from a PCM buffer.

    The send loop pushes generated audio in via ``write()`` (faster than
    real time); ``recv()`` paces delivery against the wall clock like
    aiortc's built-in AudioStreamTrack, emitting silence when the buffer is
    empty so the RTP stream stays continuous. ``clear()`` drops unplayed
    audio — this is the server-side equivalent of the client's speaker
    buffer, so barge-in must flush it for interruption to be audible.
    """

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._buffer = bytearray()
        self._start: Optional[float] = None
        self._timestamp = 0

    def write(self, pcm: bytes) -> None:
        self._buffer.extend(pcm)

    def clear(self) -> None:
        del self._buffer[:]

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    async def recv(self) -> av.AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError

        if self._start is None:
            self._start = time.time()
            self._timestamp = 0
        else:
            self._timestamp += WEBRTC_FRAME_SAMPLES
            wait = self._start + (self._timestamp / WEBRTC_SAMPLE_RATE) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)

        needed = WEBRTC_FRAME_SAMPLES * 2  # bytes of s16 mono
        payload = bytes(self._buffer[:needed])
        del self._buffer[: len(payload)]
        if len(payload) < needed:
            payload += b"\x00" * (needed - len(payload))

        samples = np.frombuffer(payload, dtype=np.int16)
        frame = av.AudioFrame.from_ndarray(samples[np.newaxis, :], format="s16", layout="mono")
        frame.sample_rate = WEBRTC_SAMPLE_RATE
        frame.pts = self._timestamp
        frame.time_base = Fraction(1, WEBRTC_SAMPLE_RATE)
        return frame


class WebRTCSession(SessionTransport):
    """One WebRTC peer connection, used as the SessionState transport.

    All pipeline integration arrives through callbacks supplied by the route
    handler (which owns the PipelineUnit): parsed client events, inbound PCM,
    channel-open, and close. This module never touches queues or services
    directly except through the SessionTransport methods the send loop calls.
    """

    kind = "webrtc"

    def __init__(
        self,
        pc: RTCPeerConnection,
        *,
        on_client_event: Callable[[dict], Awaitable[None]],
        on_audio: Callable[[bytes], None],
        on_open: Callable[[], Awaitable[None]],
        on_closed: Callable[[], None],
    ) -> None:
        self._pc = pc
        self._on_client_event = on_client_event
        self._on_audio = on_audio
        self._on_open = on_open
        self._on_closed = on_closed
        self._dc = None
        self._closed = False
        self._track = PipelineAudioTrack()
        self._out_resampler = PcmResampler(WEBRTC_SAMPLE_RATE)
        self._in_resampler = PcmResampler(PIPELINE_SAMPLE_RATE)
        # Data-channel messages funnel through one queue + consumer task so
        # client events apply in arrival order; dispatching each message in
        # its own task could reorder e.g. session.update vs response.create.
        self._dc_messages: asyncio.Queue[str] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []

    # ── Lifecycle ─────────────────────────────────

    def setup(self) -> None:
        """Wire aiortc event callbacks. Call before negotiate()."""
        self._pc.addTrack(self._track)

        @self._pc.on("datachannel")
        def on_datachannel(dc) -> None:
            if dc.label != DATA_CHANNEL_LABEL:
                logger.warning(f"[WebRTC] Ignoring unexpected data channel: {dc.label}")
                return
            self._dc = dc
            self._spawn(self._consume_dc_messages())
            logger.info(f"[WebRTC] Data channel '{DATA_CHANNEL_LABEL}' received")

            # aiortc may deliver the channel already open, in which case the
            # "open" event never fires.
            if dc.readyState == "open":
                self._spawn(self._on_open())
            else:

                @dc.on("open")
                def on_dc_open() -> None:
                    self._spawn(self._on_open())

            @dc.on("message")
            def on_message(msg) -> None:
                if isinstance(msg, str):
                    self._dc_messages.put_nowait(msg)
                else:
                    logger.warning("[WebRTC] Ignoring binary data-channel message")

            @dc.on("close")
            def on_dc_close() -> None:
                logger.info("[WebRTC] Data channel closed")
                self._spawn(self.close())

        @self._pc.on("track")
        def on_track(track) -> None:
            if track.kind == "audio":
                logger.info("[WebRTC] Inbound audio track received")
                self._spawn(self._consume_inbound_audio(track))

        @self._pc.on("connectionstatechange")
        async def on_connection_state_change() -> None:
            state = self._pc.connectionState
            logger.info(f"[WebRTC] Connection state: {state}")
            if state in ("failed", "closed"):
                await self.close()

    async def negotiate(self, offer_sdp: str) -> str:
        """Apply the client's SDP offer and return the SDP answer.

        Waits for ICE gathering so the answer carries the server's candidates
        — there is no trickle-ICE channel in the HTTP handshake.
        """
        await self._pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type="offer"))
        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)

        if self._pc.iceGatheringState != "complete":
            done: asyncio.Event = asyncio.Event()

            @self._pc.on("icegatheringstatechange")
            def on_ice_change() -> None:
                if self._pc.iceGatheringState == "complete":
                    done.set()

            if self._pc.iceGatheringState == "complete":  # raced to completion
                done.set()
            try:
                await asyncio.wait_for(done.wait(), timeout=ICE_GATHERING_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.warning("[WebRTC] ICE gathering timed out, returning partial SDP")

        self._spawn(self._connect_watchdog())
        return self._pc.localDescription.sdp

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # close() itself often runs as a _spawn()ed task (dc close handler),
        # so it is in _tasks — cancelling the current task here would abort
        # this method before the release callback runs.
        current = asyncio.current_task()
        for task in self._tasks:
            if task is not current:
                task.cancel()
        self._track.stop()
        try:
            await self._pc.close()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[WebRTC] Error closing peer connection: {e}")
        self._on_closed()
        logger.info("[WebRTC] Session closed")

    # ── SessionTransport interface ────────────────

    async def send_events(self, events: list[ServerEvent]) -> None:
        dc = self._dc
        if dc is None or dc.readyState != "open":
            return
        for event in events:
            try:
                dc.send(json.dumps(event.model_dump()))
            except Exception as e:  # noqa: BLE001
                logger.error(f"[WebRTC] Data channel send error: {e}")

    async def send_audio_chunk(self, service: RealtimeService, session_id: str, pcm: bytes) -> None:
        # Bookkeeping events (response.created on the implicit VAD path) go
        # over the data channel; the audio itself goes on the media track.
        _resp_id, _item_id, events = service.begin_audio_response(session_id)
        if events:
            await self.send_events(events)
        self._track.write(self._out_resampler.resample_pcm(pcm, PIPELINE_SAMPLE_RATE))

    def discard_pending_audio(self) -> None:
        self._track.clear()

    # ── Internals ─────────────────────────────────

    def _spawn(self, coro: Awaitable[None]) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.append(task)

    async def _connect_watchdog(self) -> None:
        await asyncio.sleep(CONNECT_TIMEOUT_S)
        if not self._closed and self._pc.connectionState != "connected":
            logger.warning(
                f"[WebRTC] Peer not connected after {CONNECT_TIMEOUT_S:.0f}s "
                f"(state: {self._pc.connectionState}); releasing session"
            )
            await self.close()

    async def _consume_dc_messages(self) -> None:
        while not self._closed:
            msg = await self._dc_messages.get()
            try:
                raw = json.loads(msg)
            except json.JSONDecodeError:
                logger.error(f"[WebRTC] Invalid JSON on data channel: {msg!r}")
                continue
            if not isinstance(raw, dict):
                logger.error(f"[WebRTC] Non-object event on data channel: {msg!r}")
                continue
            try:
                await self._on_client_event(raw)
            except Exception:  # noqa: BLE001
                logger.exception("[WebRTC] Error handling client event")

    async def _consume_inbound_audio(self, track) -> None:
        while not self._closed:
            try:
                frame = await track.recv()
            except MediaStreamError:
                logger.info("[WebRTC] Inbound audio track ended")
                break
            pcm = self._in_resampler.resample_frame(frame)
            if pcm:
                self._on_audio(pcm)
