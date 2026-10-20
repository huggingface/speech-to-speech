"""Optional terminal arrow controls for a connected audio client's voice."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import AsyncGenerator, Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, TextIO

from rich.console import Console
from rich.text import Text

if TYPE_CHECKING:
    from speech_to_speech.api.openai_realtime.audio_client import RealtimeAudioClientConfig

ArrowKey = Literal["up", "down", "right", "left"]
_ARROWS: dict[bytes, ArrowKey] = {b"A": "up", b"B": "down", b"C": "right", b"D": "left"}
LANGUAGE_NAMES = {"en": "English", "fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese"}


class SelectionDisplay:
    """Keep selectors on one transient terminal line, away from streamed speech."""

    def __init__(self, console: Console, *, inline: bool = True) -> None:
        self.console = console
        self.inline = inline and console.is_terminal
        self.values: dict[str, str] = {}
        self.busy = False
        self.visible = False

    def update(self, message: str) -> None:
        self.values[message.partition(" [")[0]] = message
        if self.inline:
            self.draw()
        elif " — " not in message or message.endswith(" — selected") or "rejected" in message:
            self.console.print(message, markup=False, highlight=False)

    def clear(self) -> None:
        if self.visible:
            self.console.file.write("\r\x1b[2K")
            self.console.file.flush()
            self.visible = False

    def draw(self) -> None:
        if self.busy or not self.inline or not self.values:
            return
        self.clear()
        ordered = [self.values[key] for key in ("Language", "Voice") if key in self.values]
        line = Text("  |  ".join(ordered), style="cyan")
        line.truncate(max(1, self.console.width - 1), overflow="ellipsis")
        self.console.print(line, end="", soft_wrap=True)
        self.visible = True

    def pause(self) -> None:
        self.busy = True
        self.clear()

    def resume(self) -> None:
        self.busy = False
        self.draw()


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

    def feed(self, data: bytes) -> list[ArrowKey]:
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
                steps.append(_ARROWS[sequence[-1:]])
        return steps


async def arrow_steps(stream: TextIO | None = None) -> AsyncGenerator[ArrowKey, None]:
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

    field = "voice"
    label = "Voice"

    def __init__(self, conn: Any, config: RealtimeAudioClientConfig, show: Callable[[str], None]) -> None:
        self.conn = conn
        self.config = config
        self.choices: Sequence[str] = getattr(config, f"{self.field}_choices")
        current = getattr(config, self.field)
        self.index = self.choices.index(current) if current is not None and current in self.choices else 0
        self.show = show
        self.active_response: str | None = None
        self.pending: str | None = None
        self.request_id: str | None = None
        self.requested_voice: str | None = None
        self.sequence = 0
        self.lock = asyncio.Lock()

    def announce(self, status: str = "") -> None:
        suffix = f" — {status}" if status else ""
        value = self.choices[self.index].replace("_", " ")
        if self.field == "language":
            value = LANGUAGE_NAMES.get(value, value)
        self.show(f"{self.label} [{self.index + 1}/{len(self.choices)}]: {value}{suffix}")

    def _audio_update(self, value: str) -> dict[str, Any]:
        if self.field == "language":
            return {"output": {"language": value}}
        return {"output": {"voice": value}}

    def _updated_value(self, audio: Any) -> str | None:
        if self.field == "language":
            return getattr(getattr(audio, "output", None), "language", None)
        return getattr(getattr(audio, "output", None), "voice", None)

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
            self.request_id = f"local_{self.field}_{self.sequence}"
            self.requested_voice = voice
            await self.conn.send(
                {
                    "type": "session.update",
                    "event_id": self.request_id,
                    "session": {"type": "realtime", "audio": self._audio_update(voice)},
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
            voice = self._updated_value(audio)
            if voice in self.choices:
                setattr(self.config, self.field, voice)
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
            current = getattr(self.config, self.field)
            if self.pending is None and current is not None and current in self.choices:
                self.index = self.choices.index(current)
                self.announce(f"{self.field} change rejected; keeping previous {self.field}")


class LanguageSelection(VoiceSelection):
    """Update LLM and TTS reply language without changing the recognition hint."""

    field = "language"
    label = "Language"
