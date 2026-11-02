"""Backend terminal output, separate from the packaged client's conversation display."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from tempfile import TemporaryFile
from typing import Any, TextIO

from rich.console import Console

console = Console()


@contextmanager
def conversation_output(enabled: bool) -> Iterator[None]:
    """Let the client own transcript output for a CLI run, retaining warnings and errors."""
    previous_quiet = console.quiet
    previous_disable = logging.root.manager.disable
    if enabled:
        console.quiet = True
        # This also covers child loggers with their own levels and handlers (uvicorn).
        logging.disable(max(previous_disable, logging.INFO))
    try:
        yield
    finally:
        if enabled:
            console.quiet = previous_quiet
            logging.disable(previous_disable)


class _StartupStream:
    """Capture startup chatter, then forward writes from any retained library handlers."""

    def __init__(self, target: TextIO, captured: TextIO) -> None:
        self.target = target
        self.captured: TextIO | None = captured

    def write(self, text: str) -> int:
        return (self.captured if self.captured is not None else self.target).write(text)

    def flush(self) -> None:
        (self.captured if self.captured is not None else self.target).flush()

    def isatty(self) -> bool:
        return False if self.captured is not None else self.target.isatty()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.target, name)


@contextmanager
def model_startup_output(enabled: bool) -> Iterator[None]:
    """Hide third-party model-loading chatter, replaying it if initialization fails.

    Only wrap synchronous construction, before the audio/server threads start. Streams
    saved by libraries become pass-through streams on exit so later errors stay visible.
    """
    if not enabled:
        yield
        return

    Console(highlight=False).print("Loading models…", style="cyan")
    transformers_logger = logging.getLogger("transformers")
    previous_level = transformers_logger.level
    with TemporaryFile(mode="w+", encoding="utf-8") as captured:
        stdout = _StartupStream(sys.stdout, captured)
        stderr = _StartupStream(sys.stderr, captured)
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                transformers_logger.setLevel(logging.ERROR)
                try:
                    yield
                finally:
                    transformers_logger.setLevel(previous_level)
        except BaseException:
            captured.seek(0)
            stderr.target.write(captured.read())
            stderr.target.flush()
            raise
        finally:
            stdout.captured = None
            stderr.captured = None
