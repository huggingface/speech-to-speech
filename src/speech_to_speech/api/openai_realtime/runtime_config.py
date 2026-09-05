from openai.types.realtime import RealtimeSessionCreateRequest
from openai.types.realtime.realtime_audio_config import RealtimeAudioConfig
from openai.types.realtime.realtime_audio_config_input import RealtimeAudioConfigInput
from openai.types.realtime.realtime_audio_config_output import RealtimeAudioConfigOutput
from pydantic import BaseModel, ConfigDict, Field, field_validator

from speech_to_speech.api.openai_realtime.session_routing import SessionRouting
from speech_to_speech.LLM.chat import Chat


def _apply_update(current: BaseModel, update: BaseModel) -> None:
    """Apply explicitly-set fields from *update* onto *current* in-place,
    recursing into nested BaseModel children so partial nested updates
    don't overwrite unset fields.

    Only fields present in update.model_fields_set (i.e. actually
    sent by the client) are considered.
    """
    for field_name in update.model_fields_set:
        new_val = getattr(update, field_name)
        old_val = getattr(current, field_name, None)
        if isinstance(new_val, BaseModel) and isinstance(old_val, BaseModel):
            _apply_update(old_val, new_val)
        else:
            setattr(current, field_name, new_val)


class RuntimeConfig(BaseModel):
    """
    Shared mutable configuration written by the RealtimeService on
    session.update and read by pipeline handlers (VAD, LLM, TTS) during
    processing.  Python's GIL makes simple attribute reads/writes atomic,
    so no explicit locking is needed for primitive values.

    The canonical state lives in 'session' (a full
    'RealtimeSessionCreateRequest').
    """

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    chat: Chat = Field(default_factory=lambda: Chat(10))
    routing: SessionRouting | None = Field(default=None, frozen=True)
    # Retained history survives disabling LLM, so its window constraint must too.
    llm_context_window_floor: int = Field(default=0, ge=0)
    session: RealtimeSessionCreateRequest = Field(
        default_factory=lambda: RealtimeSessionCreateRequest(type="realtime"),
        validate_default=True,
    )

    @field_validator("session", mode="after")
    @classmethod
    def _ensure_audio_structure(cls, v: RealtimeSessionCreateRequest) -> RealtimeSessionCreateRequest:
        """Guarantee 'audio.input' and 'audio.output' are never None."""
        if v.audio is None:
            v.audio = RealtimeAudioConfig()
        if v.audio.input is None:
            v.audio.input = RealtimeAudioConfigInput()
        if v.audio.output is None:
            v.audio.output = RealtimeAudioConfigOutput()
        return v

    @property
    def interrupt_response_enabled(self) -> bool:
        """Whether barge-in should cancel an active response.

        Reads 'turn_detection.interrupt_response' from the session config,
        handling both Pydantic models ('ServerVad') and plain dicts.
        Defaults to 'True' (OpenAI API default).
        """
        assert self.session.audio is not None and self.session.audio.input is not None
        td = self.session.audio.input.turn_detection
        if td is None:
            return True
        if hasattr(td, "interrupt_response"):
            val = td.interrupt_response
        elif isinstance(td, dict):
            val = td.get("interrupt_response", True)
        else:
            return True
        return val if val is not None else True

    @property
    def accepts_audio_input(self) -> bool:
        if self.routing is None or self.routing.routes.stt is not None:
            return True
        llm = self.routing.routes.llm
        return llm is not None and llm.protocol == "chat_completions" and llm.capabilities.audio_input

    def apply_routing_defaults(self) -> None:
        if self.routing is None:
            return
        routes = self.routing.routes
        self.session.model = routes.llm.model if routes.llm is not None else None
        assert self.session.audio is not None and self.session.audio.output is not None
        self.session.audio.output.voice = routes.tts.voice if routes.tts is not None else None
        if self.routing.updates_enabled:
            if routes.llm is not None:
                self.llm_context_window_floor = max(
                    self.llm_context_window_floor, routes.llm.capabilities.context_window or 0
                )
            self.session = self.session.model_copy(update={"models": self.routing.models()})
            if routes.tts is None:
                self.session.output_modalities = ["text"]

    def apply_session_update(self, update: RealtimeSessionCreateRequest) -> None:
        """Merge non-None, explicitly-set fields from 'update' into the
        current 'session', preserving any fields not present in the update."""
        _apply_update(self.session, update)
