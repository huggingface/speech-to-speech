from __future__ import annotations

import logging
from typing import Optional

from openai.types.realtime import (
    RealtimeErrorEvent,
    RealtimeSessionCreateRequest,
    SessionCreatedEvent,
    SessionUpdatedEvent,
    SessionUpdateEvent,
)
from openai.types.realtime.realtime_transcription_session_create_request import (
    RealtimeTranscriptionSessionCreateRequest,
)

from speech_to_speech.api.openai_realtime.handlers.base import RealtimeBaseHandler
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.session_routing import SessionRouting

logger = logging.getLogger(__name__)


class SessionHandler(RealtimeBaseHandler):
    """Owns session lifecycle: config updates and lifecycle events."""

    def handle_session_update(
        self, conn_id: str, event: SessionUpdateEvent, *, routing: SessionRouting | None = None
    ) -> Optional[RealtimeErrorEvent]:
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
        if routing is not None:
            return self._apply_routed_update(conn_id, event, routing)
        if "models" in s.model_fields_set:
            return self.make_error("Model selection requires the trusted session proxy.", "invalid_request_error")
        if cfg.routing is not None and "model" in s.model_fields_set and model != cfg.session.model:
            return self.make_error(
                message="The admitted model cannot be changed within this session.",
                _type="invalid_request_error",
            )
        current = cfg.session
        if current is None:
            cfg.session = s
        else:
            cfg.apply_session_update(s)
        logger.info("Session configuration updated")
        return None

    def _apply_routed_update(
        self, conn_id: str, event: SessionUpdateEvent, routing: SessionRouting
    ) -> Optional[RealtimeErrorEvent]:
        assert isinstance(event.session, RealtimeSessionCreateRequest)
        state = self._state(conn_id)
        cfg = state.runtime_config
        try:
            if (
                cfg.routing is None
                or not cfg.routing.updates_enabled
                or not routing.updates_enabled
                or routing.id != cfg.routing.id
            ):
                raise ValueError("Invalid session routing update.")
            if (
                state.in_response
                or state.response_pending
                or state.input_items
                or state.tool_followup_prefetch_request is not None
            ):
                raise ValueError("Finish or cancel the pending turn and wait for cleanup before changing models.")
            candidate = cfg.model_copy(update={"session": cfg.session.model_copy(deep=True), "routing": routing})
            candidate.apply_session_update(event.session)
            candidate.session = candidate.session  # Reestablish the optional audio structure after a clear.
            self._validate_routed_settings(cfg, candidate)
            old_tts, new_tts = cfg.routing.routes.tts, routing.routes.tts
            assert candidate.session.audio is not None and candidate.session.audio.output is not None
            voice = candidate.session.audio.output.voice
            audio = event.session.audio
            explicit_voice = audio is not None and audio.output is not None and "voice" in audio.output.model_fields_set
            if new_tts is not None:
                if old_tts != new_tts and not explicit_voice:
                    voice = new_tts.voice
                if voice not in [new_tts.voice, *new_tts.voices]:
                    raise ValueError("The selected TTS model does not support this voice.")
            if (
                new_tts is None
                and "output_modalities" in event.session.model_fields_set
                and "audio" in (event.session.output_modalities or [])
            ):
                raise ValueError("Audio output requires a selected TTS model.")
            candidate.apply_routing_defaults()
            assert candidate.session.audio is not None and candidate.session.audio.output is not None
            if new_tts is not None:
                candidate.session.audio.output.voice = voice
                if old_tts is None and "output_modalities" not in event.session.model_fields_set:
                    candidate.session.output_modalities = ["audio"]
            if not cfg.chat.prepare_route_change():
                raise ValueError(
                    "Resolve pending tools or wait for generation/compaction cleanup before changing models."
                )
            # The WebSocket dispatcher has drained serial work. No await occurs
            # between validation and replacement; queued new requests see this
            # complete configuration and the same retained Chat.
            state.runtime_config = candidate
        except ValueError as exc:
            return self.make_error(str(exc), "invalid_request_error")
        return None

    @staticmethod
    def _validate_routed_settings(previous: RuntimeConfig, candidate: RuntimeConfig) -> None:
        assert previous.routing is not None and candidate.routing is not None
        old, new = previous.routing.routes.llm, candidate.routing.routes.llm
        if new is None:
            return
        caps = new.capabilities
        if caps.context_window is None or caps.continuation != "full_context":
            raise ValueError("The selected LLM must declare its context window and support full retained context.")
        if old is not None and (
            new.protocol != old.protocol
            or caps.context_window < (old.capabilities.context_window or caps.context_window)
        ):
            raise ValueError("The destination must use the same protocol and an equal or larger context window.")
        if candidate.session.tools and not caps.tools:
            raise ValueError("The selected LLM does not support the session's tools.")
        snapshot = candidate.chat.copy(deep=True)

        def validate(value):
            if isinstance(value, dict):
                kind = value.get("type")
                if kind in {"input_image", "image_url"} and not caps.images:
                    raise ValueError("The selected LLM does not support retained images.")
                if kind == "input_audio" and (not caps.audio_input or new.protocol != "chat_completions"):
                    raise ValueError("The selected LLM cannot preserve the retained audio input.")
                if kind in {"function_call", "function_call_output"} and not caps.tools:
                    raise ValueError("The selected LLM does not support retained tool history.")
                for item in value.values():
                    validate(item)
            elif isinstance(value, list):
                for item in value:
                    validate(item)

        for item in snapshot.buffer:
            validate(item.model_dump(mode="json"))

    def build_session_created(self, conn_id: str) -> SessionCreatedEvent:
        """Build a SessionCreatedEvent populated with the current config."""
        cfg = self._state(conn_id).runtime_config
        # The OpenAI GA protocol includes the session id in session.created.
        # The SDK model has no `id` field but allows extras, and model_dump()
        # carries them onto the wire.
        session = cfg.session.model_copy(update={"id": conn_id})
        return SessionCreatedEvent(
            type="session.created",
            event_id=self._next_event_id(),
            session=session,
        )

    def build_session_updated(self, conn_id: str) -> SessionUpdatedEvent:
        """Build a SessionUpdatedEvent populated with the current config.

        Sent after a successful session.update, per the OpenAI Realtime
        protocol: https://platform.openai.com/docs/api-reference/realtime-server-events/session/updated
        """
        cfg = self._state(conn_id).runtime_config
        session = cfg.session.model_copy(update={"id": conn_id})
        return SessionUpdatedEvent(
            type="session.updated",
            event_id=self._next_event_id(),
            session=session,
        )
