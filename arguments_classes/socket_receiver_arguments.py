from dataclasses import dataclass, field


@dataclass
class SocketReceiverArguments:
    recv_host: str = field(
        default="localhost",
        metadata={
            "help": "The host IP ddress for the socket connection. Default is '0.0.0.0' which binds to all "
            "available interfaces on the host machine."
        },
    )
    recv_port: int = field(
        default=12345,
        metadata={
            "help": "The port number on which the socket server listens. Default is 12346."
        },
    )
    chunk_size: int = field(
        default=1024,
        metadata={
            "help": "The size of each data chunk to be sent or received over the socket. Default is 1024 bytes."
        },
    )
