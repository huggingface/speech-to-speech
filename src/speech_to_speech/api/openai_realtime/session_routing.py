"""Immutable initial routes supplied by a trusted admission proxy, never clients."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Identity = Annotated[str, Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9._/@:-]*$")]


class Route(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, hide_input_in_errors=True)

    model: Identity
    provider: Identity


class TranscriptionRoute(Route):
    protocol: Literal["transcriptions"]


class LanguageRoute(Route):
    protocol: Literal["chat_completions", "responses"]
    capabilities: "LanguageCapabilities" = Field(default_factory=lambda: LanguageCapabilities())


class LanguageCapabilities(BaseModel):
    model_config = Route.model_config

    tools: bool = False
    images: bool = False
    audio_input: bool = False
    context_window: int | None = Field(default=None, gt=0)
    continuation: Literal["full_context", "provider_state"] = "full_context"


class SpeechRoute(Route):
    protocol: Literal["speech"]
    voice: Identity
    voices: list[Identity] = Field(default_factory=list)


class SessionRoutes(BaseModel):
    model_config = Route.model_config

    stt: TranscriptionRoute | None
    llm: LanguageRoute | None
    tts: SpeechRoute | None


class SessionRouting(BaseModel):
    model_config = Route.model_config

    id: Annotated[str, Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9_-]+$")]
    pipeline: Annotated[str, Field(min_length=1, max_length=800, pattern=r"^[A-Za-z0-9@][A-Za-z0-9._/@:+-]*$")]
    routes: SessionRoutes
    updates_enabled: bool = False

    @model_validator(mode="after")
    def require_legacy_stages(self) -> "SessionRouting":
        if not self.updates_enabled and any(getattr(self.routes, stage) is None for stage in ("stt", "llm", "tts")):
            raise ValueError("partial routes require session updates to be enabled")
        return self

    def models(self) -> dict[str, dict[str, str] | None]:
        return {
            stage: {"model": route.model, "provider": route.provider} if route is not None else None
            for stage in ("stt", "llm", "tts")
            for route in (getattr(self.routes, stage),)
        }

    def headers(self, stage: Literal["stt", "llm", "tts"]) -> dict[str, str]:
        route = getattr(self.routes, stage)
        if route is None:
            raise ValueError(f"No {stage.upper()} model selected")
        return {"X-Speech-Provider": route.provider, "X-Speech-Session-Id": self.id}
