from typing import Any

from openai.types.realtime import RealtimeSessionCreateRequest
from openai.types.realtime.realtime_audio_config import RealtimeAudioConfig
from openai.types.realtime.realtime_audio_config_input import RealtimeAudioConfigInput
from openai.types.realtime.realtime_audio_config_output import RealtimeAudioConfigOutput
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from speech_to_speech.LLM.chat import Chat
from speech_to_speech.utils.request_tracing import safe_identifier

# OpenAI Realtime ``reasoning.effort`` levels, plus ``none`` so clients can also
# switch reasoning off on providers that support it (e.g. gpt-5.x, HF router).
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")


def reasoning_effort_of(params: Any) -> str | None:
    """Read ``reasoning.effort`` from a Realtime session or response config.

    The pinned OpenAI SDK has no typed ``reasoning`` field yet, so it arrives as
    a pydantic extra (a plain dict); newer SDKs expose a model. Handle both.
    """
    reasoning = getattr(params, "reasoning", None) if params is not None else None
    if reasoning is None:
        return None
    effort = reasoning.get("effort") if isinstance(reasoning, dict) else getattr(reasoning, "effort", None)
    return effort or None


def validate_reasoning(params: Any) -> str | None:
    """Return an error message when ``params.reasoning`` is malformed, else ``None``."""
    reasoning = getattr(params, "reasoning", None) if params is not None else None
    if reasoning is None:
        return None
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
    elif isinstance(reasoning, BaseModel):
        effort = getattr(reasoning, "effort", None)
    else:
        return "reasoning must be an object."
    if effort is not None and effort not in REASONING_EFFORTS:
        return f"Invalid reasoning.effort {effort!r}; expected one of: {', '.join(REASONING_EFFORTS)}."
    return None


def resolve_reasoning_effort(runtime_config: Any, response: Any = None) -> str | None:
    """Effective per-request effort: ``response.create`` overrides the session.

    ``None`` means the client set nothing and the backend's startup default applies.
    """
    session = getattr(runtime_config, "session", None) if runtime_config is not None else None
    return reasoning_effort_of(response) or reasoning_effort_of(session)


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
        """Label remote-request traces with the owning server session (set once at registration)."""
        if safe_identifier(session_id) is None:
            raise ValueError("invalid server trace session ID")
        self._trace_session_id = session_id

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
        # The pinned SDK keeps ``reasoning`` as a plain dict extra, which
        # _apply_update would replace wholesale; merge its keys so a partial
        # object (e.g. ``{}``) keeps the current effort. ``null`` still resets.
        old_reasoning = getattr(self.session, "reasoning", None)
        new_reasoning = getattr(update, "reasoning", None)
        if (
            "reasoning" in update.model_fields_set
            and isinstance(new_reasoning, dict)
            and isinstance(old_reasoning, dict)
        ):
            setattr(update, "reasoning", {**old_reasoning, **new_reasoning})
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
