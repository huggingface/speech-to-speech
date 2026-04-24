from __future__ import annotations

import logging
import uuid
from queue import Queue
from threading import Event as ThreadingEvent
from typing import TYPE_CHECKING

from openai.types.realtime import RealtimeErrorEvent

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.service import ConnState, RealtimeService

logger = logging.getLogger(__name__)


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class RealtimeBaseHandler:
    """Shared base for domain handlers.

    Provides conn_id-keyed accessors for per-connection state, config,
    and queues.  Each connection owns its own ``RuntimeConfig`` via
    ``ConnState``.
    """

    def __init__(self, service: RealtimeService) -> None:
        self._service = service

    # ── conn_id-keyed accessors ──

    def _queue(self, conn_id: str) -> Queue | None:
        return self._service.text_prompt_queue

    def _should_listen(self, conn_id: str) -> ThreadingEvent | None:
        return self._service.should_listen

    # ── shared helpers ────────────────────────────────

    def _state(self, conn_id: str) -> ConnState:
        return self._service._state(conn_id)

    @staticmethod
    def _next_event_id() -> str:
        return _generate_id("event")

    def make_error(self, message: str, _type: str) -> RealtimeErrorEvent:
        return self._service.make_error(message, _type)
