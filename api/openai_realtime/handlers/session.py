from __future__ import annotations

import logging
from typing import Optional

from openai.types.realtime import (
    RealtimeErrorEvent,
    SessionCreatedEvent,
    SessionUpdateEvent,
)
from openai.types.realtime.realtime_transcription_session_create_request import (
    RealtimeTranscriptionSessionCreateRequest,
)

from api.openai_realtime.handlers.base import RealtimeBaseHandler

logger = logging.getLogger(__name__)


class SessionHandler(RealtimeBaseHandler):
    """Owns session lifecycle: config updates and session.created events."""

    def handle_session_update(self, conn_id: str, event: SessionUpdateEvent) -> Optional[RealtimeErrorEvent]:
        """Apply session config changes.

        Only ``RealtimeSessionCreateRequest`` sessions are accepted;
        ``RealtimeTranscriptionSessionCreateRequest`` sessions not yet supported.
        Incoming fields are deep-merged into the existing session so that
        partial updates preserve previously-set values.
        """
        s = event.session
        if s is None:
            return None

        if isinstance(s, RealtimeTranscriptionSessionCreateRequest):
            return self.make_error(
                message="Only 'realtime' session type is supported; transcription sessions are not.",
                _type="invalid_session_type",
            )

        model = getattr(s, "model", None)
        if model is not None:
            logger.info(f"Session model set to: {model}")

        cfg = self._state(conn_id).runtime_config
        current = cfg.session
        if current is None:
            cfg.session = s
        else:
            cfg.apply_session_update(s)
        logger.info("Session configuration updated")
        return None

    def build_session_created(self, conn_id: str) -> SessionCreatedEvent:
        """Build a SessionCreatedEvent populated with the current config."""
        cfg = self._state(conn_id).runtime_config
        session = cfg.session
        return SessionCreatedEvent(
            type="session.created",
            event_id=self._next_event_id(),
            session=session,
        )
