from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineControlMessage:
    kind: str


SESSION_END = PipelineControlMessage("session_end")


def is_control_message(message, kind=None):
    return isinstance(message, PipelineControlMessage) and (
        kind is None or message.kind == kind
    )
