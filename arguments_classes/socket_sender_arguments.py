from dataclasses import dataclass, field


@dataclass
class SocketSenderArguments:
    send_host: str = field(
        default="localhost",
        metadata={
            "help": "The host IP address for the socket connection. Default is '0.0.0.0' which binds to all "
            "available interfaces on the host machine."
        },
    )
    send_port: int = field(
        default=12346,
        metadata={
            "help": "The port number on which the socket server listens. Default is 12346."
        },
    )
