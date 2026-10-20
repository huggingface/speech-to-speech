"""Backend terminal output, separate from the packaged client's conversation display."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

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
