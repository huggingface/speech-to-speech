from openai.types.realtime import RealtimeSessionCreateRequest
from openai.types.realtime.realtime_audio_config import RealtimeAudioConfig
from openai.types.realtime.realtime_audio_config_input import RealtimeAudioConfigInput
from openai.types.realtime.realtime_audio_config_output import RealtimeAudioConfigOutput
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from speech_to_speech.LLM.chat import Chat
from speech_to_speech.utils.request_tracing import safe_identifier


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
    session: RealtimeSessionCreateRequest = Field(
        default_factory=lambda: RealtimeSessionCreateRequest(type="realtime"),
        validate_default=True,
    )
    last_assistant_language: str | None = Field(default=None, exclude=True)
    _trace_session_id: str | None = PrivateAttr(default=None)

    @property
    def trace_session_id(self) -> str | None:
        """Server ownership, separate from client-editable session parameters."""
        return self._trace_session_id

    def bind_trace_session(self, session_id: str) -> None:
        if safe_identifier(session_id) is None:
            raise ValueError("invalid server trace session ID")
        if self._trace_session_id not in (None, session_id):
            raise ValueError("runtime config already belongs to another trace session")
        self._trace_session_id = session_id

    def for_trace_session(self, session_id: str) -> "RuntimeConfig":
        """Adopt settings without relabeling an earlier session's config."""
        if self.trace_session_id not in (None, session_id):
            # Public settings/history may be reused by a server-side replacement.
            # Construct a new wrapper from declared fields, never copied private
            # ownership or client-supplied extras. In-flight work keeps the old one.
            replacement = type(self).model_construct(**{name: getattr(self, name) for name in type(self).model_fields})
            replacement.bind_trace_session(session_id)
            return replacement
        self.bind_trace_session(session_id)
        return self

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
    def create_response_enabled(self) -> bool:
        """Whether finalized speech should automatically schedule generation.

        Preserve the existing automatic default when the flag is omitted.
        """
        assert self.session.audio is not None and self.session.audio.input is not None
        td = self.session.audio.input.turn_detection
        value = td.get("create_response") if isinstance(td, dict) else getattr(td, "create_response", None)
        return value if value is not None else True

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
    def input_audio_transcription_snapshots_enabled(self) -> bool:
        """Whether the client has opted in to replaceable speculative transcript snapshots.

        Reads 'extensions' from the session config. Defaults to 'False'.
        """
        extensions = getattr(self.session, "extensions", None)
        return (
            isinstance(extensions, (list, tuple, set))
            and "speech_to_speech.input_audio_transcription.snapshot" in extensions
        )

    def apply_session_update(self, update: RealtimeSessionCreateRequest) -> None:
        """Merge non-None, explicitly-set fields from 'update' into the
        current 'session', preserving any fields not present in the update."""
        _apply_update(self.session, update)

    @property
    def selected_language(self) -> str | None:
        audio = self.session.audio
        input_audio = audio.input if audio is not None else None
        transcription = input_audio.transcription if input_audio is not None else None
        language = transcription.language if transcription is not None else None
        if isinstance(language, str) and language.strip().lower() == "auto":
            return "auto"
        return language
