"""Per-pipeline logging context.

Each isolated pipeline unit sets the contextvar at thread / asyncio-task entry,
and every log record emitted from that thread/task is automatically tagged with
a `[pipeline N] ` prefix via PipelineLogFilter. Non-pipeline-bound code emits
records with an empty prefix.

Works for both threading (handler threads) and asyncio (websocket route, send loops)
because contextvars are per-thread and per-task.
"""

import contextvars
import logging
from typing import Optional

pipeline_log_ctx: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar("pipeline_index", default=None)


class PipelineLogFilter(logging.Filter):
    """Inject `pipeline_prefix` into every LogRecord based on the current contextvar."""

    def filter(self, record: logging.LogRecord) -> bool:
        idx = pipeline_log_ctx.get()
        record.pipeline_prefix = f"[pipeline {idx}] " if idx is not None else ""
        return True
