from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import Event


class ControlKind(str, Enum):
    """Strongly-typed kinds for :class:`PipelineControlMessage`."""

    SESSION_END = "session_end"
    ROUTING_BARRIER = "routing_barrier"


@dataclass(frozen=True)
class PipelineControlMessage:
    kind: ControlKind
    # Session that enqueued the message, when known. Lets the pooled realtime
    # send loop ignore a SESSION_END from a force-released session so it can't
    # satisfy the drain wait of the session that claimed the unit afterwards.
    session_id: str | None = None
    reached: Event | None = None


SESSION_END = PipelineControlMessage(ControlKind.SESSION_END)


def is_control_message(message: object, kind: ControlKind | None = None) -> bool:
    return isinstance(message, PipelineControlMessage) and (kind is None or message.kind == kind)
