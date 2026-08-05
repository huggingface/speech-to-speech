from dataclasses import dataclass, field


@dataclass
class RealtimeServerArguments:
    host: str = field(
        default="0.0.0.0",
        metadata={
            "help": "Host interface for the Realtime server. Default is 0.0.0.0. "
            "Local mode always binds to 127.0.0.1."
        },
    )
    port: int = field(
        default=8765,
        metadata={"help": "Port for the Realtime HTTP/WebSocket server. Default is 8765."},
    )
