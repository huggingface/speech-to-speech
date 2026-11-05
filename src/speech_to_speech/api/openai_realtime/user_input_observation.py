"""Opt-in, payload-free observations of committed default-conversation input."""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from typing import Annotated, Literal, TypedDict

from openai.types.realtime import ConversationItem
from openai.types.realtime.realtime_conversation_item_user_message import RealtimeConversationItemUserMessage
from pydantic import BaseModel, ConfigDict, Field, StrictInt

logger = logging.getLogger(__name__)
USER_INPUT_OBSERVATION_HEADER = "x-speech-to-speech-user-input-observation"
USER_INPUT_OBSERVATION_VERSION = "1"
MAX_PENDING_INPUT_OBSERVATIONS = 64
MAX_INPUT_OBSERVATION_SEQUENCE = 2**53 - 1
UserInputKind = Literal["text", "image", "audio", "mixed", "speech"]
Sequence = Annotated[StrictInt, Field(ge=0, le=MAX_INPUT_OBSERVATION_SEQUENCE)]


class _InputObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    event_id: Annotated[str, Field(strict=True, min_length=1, max_length=512)]
    session_id: Annotated[str, Field(strict=True, min_length=1, max_length=256)]
    timestamp_s: Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
    dropped_events: Sequence = 0


class UserInputObservationStarted(_InputObservation):
    type: Literal["speech_to_speech.user_input.observation_started"] = "speech_to_speech.user_input.observation_started"
    sequence: Literal[0] = 0


class UserInputAccepted(_InputObservation):
    type: Literal["speech_to_speech.user_input.accepted"] = "speech_to_speech.user_input.accepted"
    sequence: Annotated[StrictInt, Field(ge=1, le=MAX_INPUT_OBSERVATION_SEQUENCE)]
    input_kind: UserInputKind


InputObservation = UserInputObservationStarted | UserInputAccepted


class _ObservationFields(TypedDict):
    event_id: str
    session_id: str
    timestamp_s: float
    dropped_events: int


def user_item_kind(item: ConversationItem) -> UserInputKind | None:
    """Snapshot a supplied item's fixed kind without retaining or returning its content."""
    if not isinstance(item, RealtimeConversationItemUserMessage):
        return None
    kinds: set[UserInputKind] = set()
    for part in item.content:
        if part.type == "input_text" and part.text and part.text.strip():
            kinds.add("text")
        elif part.type == "input_image" and part.image_url and part.image_url.strip():
            kinds.add("image")
        elif part.type == "input_audio" and part.audio and part.audio.strip():
            kinds.add("audio")
    return next(iter(kinds)) if len(kinds) == 1 else "mixed" if kinds else None


class UserInputObservations:
    """One connection's bounded best-effort queue, owned by the existing event loop."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.sequence = 0
        self.dropped_events = 0
        self._pending: deque[InputObservation] = deque()
        self._append(None)

    def accepted(self, kind: UserInputKind) -> None:
        if self.sequence == MAX_INPUT_OBSERVATION_SEQUENCE:
            self.lost(1)
            return
        self.sequence += 1
        self._append(kind)

    def lost(self, count: int) -> None:
        self.dropped_events = min(MAX_INPUT_OBSERVATION_SEQUENCE, self.dropped_events + count)

    def _append(self, kind: UserInputKind | None) -> None:
        if len(self._pending) >= MAX_PENDING_INPUT_OBSERVATIONS:
            self.lost(1)
            return
        try:
            timestamp = time.time()
            if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
                raise ValueError("invalid observation clock")
            timestamp = float(timestamp)
            if not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError("invalid observation clock")
            fields: _ObservationFields = {
                "event_id": f"user_input_{self.session_id}_{self.sequence}",
                "session_id": self.session_id,
                "timestamp_s": timestamp,
                "dropped_events": self.dropped_events,
            }
            event: InputObservation
            if kind is None:
                event = UserInputObservationStarted(**fields)
            else:
                event = UserInputAccepted(**fields, sequence=self.sequence, input_kind=kind)
            self._pending.append(event)
        except Exception as exc:  # Observability cannot undo a committed input.
            self.lost(1)
            logger.warning("Could not queue user-input observation (%s)", type(exc).__name__)

    def drain(self) -> list[InputObservation]:
        events = [event.model_copy(update={"dropped_events": self.dropped_events}) for event in self._pending]
        self._pending.clear()
        return events
