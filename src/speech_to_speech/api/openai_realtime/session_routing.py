"""Immutable initial routes supplied by a trusted admission proxy, never clients."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Identity = Annotated[str, Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9._/@:-]*$")]


class Route(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, hide_input_in_errors=True)

    model: Identity
    provider: Identity


class TranscriptionRoute(Route):
    protocol: Literal["transcriptions"]


class LanguageRoute(Route):
    protocol: Literal["chat_completions", "responses"]


class SpeechRoute(Route):
    protocol: Literal["speech"]
    voice: Identity


class SessionRoutes(BaseModel):
    model_config = Route.model_config

    stt: TranscriptionRoute
    llm: LanguageRoute
    tts: SpeechRoute


class SessionRouting(BaseModel):
    model_config = Route.model_config

    id: Annotated[str, Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9_-]+$")]
    pipeline: Identity
    routes: SessionRoutes

    def headers(self, stage: Literal["stt", "llm", "tts"]) -> dict[str, str]:
        return {"X-Speech-Provider": getattr(self.routes, stage).provider, "X-Speech-Session-Id": self.id}
