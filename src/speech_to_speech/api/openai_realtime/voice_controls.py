"""Optional terminal arrow controls for a connected audio client's voice."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import AsyncGenerator, Callable, Sequence
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.audio_client import RealtimeAudioClientConfig


def has_terminal_arrows(stream: TextIO | None = None) -> bool:
    stream = stream if stream is not None else sys.stdin
    try:
        return os.name == "posix" and stream.isatty()
    except (AttributeError, OSError, ValueError):
        return False


class ArrowDecoder:
    """Decode split ANSI/application-mode arrows without treating typed text as keys."""

    def __init__(self) -> None:
        self.pending = b""

    def feed(self, data: bytes) -> list[int]:
        self.pending += data
        steps = []
        while self.pending:
            start = self.pending.find(b"\x1b")
            if start < 0:
                self.pending = b""
                break
            self.pending = self.pending[start:]
            if len(self.pending) < 2:
                break
            if self.pending[1:2] not in (b"[", b"O"):
                self.pending = self.pending[1:]
                continue
            end = next((i for i in range(2, len(self.pending)) if 0x40 <= self.pending[i] <= 0x7E), None)
            if end is None:
                if len(self.pending) > 32:
                    self.pending = b""
                break
            sequence, self.pending = self.pending[: end + 1], self.pending[end + 1 :]
            if len(sequence) == 3 and sequence[-1:] in (b"A", b"B", b"C", b"D"):
                steps.append(1 if sequence[-1:] in (b"B", b"C") else -1)
        return steps


async def arrow_steps(stream: TextIO | None = None) -> AsyncGenerator[int, None]:
    """Read arrows on POSIX TTYs; restore echo and input mode on every exit path."""

    stream = stream if stream is not None else sys.stdin
    if not has_terminal_arrows(stream):
        return
    import termios
    import tty

    loop = asyncio.get_running_loop()
    fd = stream.fileno()
    previous = termios.tcgetattr(fd)
    chunks: asyncio.Queue[bytes] = asyncio.Queue()
    registered = False

    def readable() -> None:
        try:
            data = os.read(fd, 64)
        except OSError:
            data = b""
        chunks.put_nowait(data)
        if not data:
            loop.remove_reader(fd)

    try:
        # cbreak preserves ISIG, so Ctrl+C still reaches the normal shutdown handler.
        tty.setcbreak(fd)
        loop.add_reader(fd, readable)
        registered = True
        decoder = ArrowDecoder()
        while data := await chunks.get():
            for step in decoder.feed(data):
                yield step
    finally:
        if registered:
            loop.remove_reader(fd)
        termios.tcsetattr(fd, termios.TCSANOW, previous)


class VoiceSelection:
    """Send voice-only updates on the existing connection, between responses."""

    def __init__(self, conn: Any, config: RealtimeAudioClientConfig, show: Callable[[str], None]) -> None:
        self.conn = conn
        self.config = config
        self.choices: Sequence[str] = config.voice_choices
        self.index = (
            self.choices.index(config.voice) if config.voice is not None and config.voice in self.choices else 0
        )
        self.show = show
        self.active_response: str | None = None
        self.pending: str | None = None
        self.request_id: str | None = None
        self.requested_voice: str | None = None
        self.sequence = 0
        self.lock = asyncio.Lock()

    def announce(self, status: str = "") -> None:
        suffix = f" — {status}" if status else ""
        self.show(f"Voice [{self.index + 1}/{len(self.choices)}]: {self.choices[self.index]}{suffix}")

    async def select(self, step: int) -> None:
        self.index = (self.index + step) % len(self.choices)
        self.pending = self.choices[self.index]
        self.announce("next reply" if self.active_response else "switching")
        await self._flush()

    async def _flush(self) -> None:
        async with self.lock:
            if self.active_response or self.pending is None:
                return
            voice, self.pending = self.pending, None
            self.sequence += 1
            self.request_id = f"local_voice_{self.sequence}"
            self.requested_voice = voice
            await self.conn.send(
                {
                    "type": "session.update",
                    "event_id": self.request_id,
                    "session": {"type": "realtime", "audio": {"output": {"voice": voice}}},
                }
            )

    async def handle_event(self, event: Any) -> None:
        if event.type == "response.created":
            self.active_response = event.response.id
        elif event.type == "response.done" and event.response.id == self.active_response:
            self.active_response = None
            await self._flush()
        elif event.type == "session.updated":
            audio = getattr(event.session, "audio", None)
            output = getattr(audio, "output", None)
            voice = getattr(output, "voice", None)
            if voice in self.choices:
                self.config.voice = voice
            if voice == self.requested_voice and voice == self.choices[self.index]:
                self.request_id = None
                self.requested_voice = None
                self.announce("selected")
        elif (
            event.type == "error"
            and self.request_id is not None
            and getattr(event.error, "event_id", None) == self.request_id
        ):
            self.request_id = None
            self.requested_voice = None
            if self.pending is None and self.config.voice is not None and self.config.voice in self.choices:
                self.index = self.choices.index(self.config.voice)
                self.announce("voice change rejected; keeping previous voice")
