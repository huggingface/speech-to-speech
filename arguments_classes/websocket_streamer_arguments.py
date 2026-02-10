from dataclasses import dataclass, field


@dataclass
class WebSocketStreamerArguments:
    ws_host: str = field(
        default="0.0.0.0",
        metadata={
            "help": "The host IP address for the WebSocket server. Default is '0.0.0.0' which binds to all "
            "available interfaces on the host machine."
        },
    )
    ws_port: int = field(
        default=8765,
        metadata={
            "help": "The port number on which the WebSocket server listens. Default is 8765."
        },
    )
