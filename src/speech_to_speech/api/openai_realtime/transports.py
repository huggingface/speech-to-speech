"""Session transports: how server events and outbound audio reach a client.

The per-unit send loop in ``websocket_router`` is transport-agnostic: it owns
the pipeline output queues (sentinels, generation discards, SESSION_END drain)
and hands client-visible traffic to the transport attached to the current
``SessionState``. Two implementations exist:

- ``WebSocketTransport`` (here): events and base64 audio deltas as JSON frames.
- ``WebRTCSession`` (in ``webrtc_session``, requires the ``webrtc`` extra):
  events over the ``oai-events`` data channel, audio over the RTP media track.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from speech_to_speech.api.openai_realtime.handlers.audio import PIPELINE_SAMPLE_RATE

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.service import RealtimeService, ServerEvent

logger = logging.getLogger(__name__)


class SessionTransport(ABC):
    """What the send loop and client-event dispatch need from a transport."""

    kind: str

    @abstractmethod
    async def send_events(self, events: list[ServerEvent]) -> None: ...

    @abstractmethod
    async def send_audio_chunk(
        self,
        service: RealtimeService,
        session_id: str,
        pcm: bytes,
        response_key: str | None = None,
    ) -> None:
        """Deliver a pipeline-rate PCM16 chunk to the client."""

    @abstractmethod
    def discard_pending_audio(self) -> float:
        """Drop assistant audio the client has not played yet.

        The WebRTC transport paces playback server-side and must flush its
        track buffer for barge-in to actually silence the assistant. WebSocket
        clients buffer audio on their side and flush it themselves; that
        transport only estimates how much was left.
        Returns the duration of unplayed audio in milliseconds.
        """

    @abstractmethod
    async def close(self) -> None: ...


async def send_ws_event(ws: WebSocket, event: ServerEvent) -> None:
    # Skip cleanly when the ws is already closing/closed — happens during Ctrl-C
    # shutdown, where the lifespan starts closing sockets while the route handler
    # or send loop is still in flight pushing events.
    if ws.application_state != WebSocketState.CONNECTED:
        return
    try:
        await ws.send_json(event.model_dump())
    except WebSocketDisconnect:
        logger.debug("Skipped event: ws disconnected mid-send")
    except RuntimeError as e:
        # Race: ws closed between the state check above and the send. Starlette
        # raises a plain RuntimeError("Unexpected ASGI message 'websocket.send'
        # after sending 'websocket.close' ...") — harmless during shutdown.
        msg = str(e)
        if "websocket.close" in msg or "websocket.disconnect" in msg or "response already completed" in msg:
            logger.debug(f"Skipped event: ws already closed ({msg})")
        else:
            logger.error(f"Failed to send event to client: {e}")
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to send event to client: {e}")


class WebSocketTransport(SessionTransport):
    """JSON-over-WebSocket transport: audio is sent as base64 delta events."""

    kind = "websocket"

    def __init__(self, websocket: WebSocket, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.websocket = websocket
        self._send_lock = asyncio.Lock()
        self._clock = clock
        # When the client should finish playing the audio sent so far, assuming
        # it plays in real time from the moment each chunk is sent.
        self._playback_end = 0.0
        # Audio of sends still in progress, and a counter that tells a send
        # finishing after a discard not to add its discarded audio back.
        self._sending_seconds = 0.0
        self._discards = 0

    async def send_events(self, events: list[ServerEvent]) -> None:
        async with self._send_lock:
            for event in events:
                await send_ws_event(self.websocket, event)

    async def send_audio_chunk(
        self,
        service: RealtimeService,
        session_id: str,
        pcm: bytes,
        response_key: str | None = None,
    ) -> None:
        events = service.encode_audio_chunk(session_id, pcm, response_key)
        # History counts this audio as delivered from here on, so a discard
        # during the send must count it as unplayed.
        seconds = len(pcm) / 2 / PIPELINE_SAMPLE_RATE if events else 0.0
        discards = self._discards
        self._sending_seconds += seconds
        try:
            await self.send_events(events)
        finally:
            if discards == self._discards:
                self._sending_seconds -= seconds
        if seconds and discards == self._discards:
            self._playback_end = max(self._playback_end, self._clock()) + seconds

    def discard_pending_audio(self) -> float:
        # The client flushes its own player on barge-in. Report the audio it
        # had not played yet, including sends still in progress, so clients
        # that never send conversation.item.truncate still get history cut to
        # what they heard.
        now = self._clock()
        unplayed = max(0.0, self._playback_end - now) + self._sending_seconds
        self._playback_end = now
        self._sending_seconds = 0.0
        self._discards += 1
        return unplayed * 1000

    async def close(self) -> None:
        try:
            await self.websocket.close()
        except Exception:  # noqa: BLE001
            pass
