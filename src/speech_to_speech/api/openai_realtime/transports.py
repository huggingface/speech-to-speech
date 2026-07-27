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

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.service import RealtimeService, ServerEvent

logger = logging.getLogger(__name__)


class SessionTransport(ABC):
    """What the send loop and client-event dispatch need from a transport."""

    kind: str

    @abstractmethod
    async def send_events(self, events: list[ServerEvent]) -> None: ...

    @abstractmethod
    async def send_audio_chunk(self, service: RealtimeService, session_id: str, pcm: bytes) -> None:
        """Deliver a pipeline-rate PCM16 chunk to the client."""

    @abstractmethod
    def discard_pending_audio(self) -> None:
        """Drop transport-buffered audio that has not reached the client yet.

        WebSocket clients buffer audio on their side, so this is a no-op there;
        the WebRTC transport paces playback server-side and must flush its
        track buffer for barge-in to actually silence the assistant.
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

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket

    async def send_events(self, events: list[ServerEvent]) -> None:
        for event in events:
            await send_ws_event(self.websocket, event)

    async def send_audio_chunk(self, service: RealtimeService, session_id: str, pcm: bytes) -> None:
        await self.send_events(service.encode_audio_chunk(session_id, pcm))

    def discard_pending_audio(self) -> None:
        # Unplayed audio lives client-side over WebSocket; truncation is the
        # client's responsibility.
        pass

    async def close(self) -> None:
        try:
            await self.websocket.close()
        except Exception:  # noqa: BLE001
            pass
