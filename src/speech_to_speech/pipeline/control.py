from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ControlKind(str, Enum):
    """Strongly-typed kinds for :class:`PipelineControlMessage`."""

    SESSION_END = "session_end"


@dataclass(frozen=True)
class PipelineControlMessage:
    kind: ControlKind


SESSION_END = PipelineControlMessage(ControlKind.SESSION_END)


def is_control_message(message: object, kind: ControlKind | None = None) -> bool:
    return isinstance(message, PipelineControlMessage) and (kind is None or message.kind == kind)
