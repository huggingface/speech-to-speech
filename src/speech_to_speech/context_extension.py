"""Optional, deployment-owned context for a single model request."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from importlib.metadata import distribution
from pathlib import Path
from threading import Event
from time import monotonic, perf_counter
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from starlette.types import ASGIApp

logger = logging.getLogger(__name__)
STATE_KEY = "speech_to_speech.context_scope"
MAX_CONTEXT_WAIT_MS = 400
_MAX_WAIT_SECONDS = MAX_CONTEXT_WAIT_MS / 1000


def _bounded_wait(seconds: float) -> float:
    if type(seconds) not in (int, float) or seconds < 0 or not math.isfinite(seconds):
        raise ValueError("Context preparation budget must be a finite, nonnegative number of seconds")
    return min(seconds, _MAX_WAIT_SECONDS)


@dataclass(frozen=True)
class ContextScope:
    """Created by deployment middleware after authenticating the connection."""

    tenant: str
    issuer: str
    subject: str
    expires_at: float  # monotonic time in this process
    revoked: Event = field(default_factory=Event, repr=False)

    def valid(self) -> bool:
        return (
            all(isinstance(value, str) and value for value in (self.tenant, self.issuer, self.subject))
            and isinstance(self.expires_at, (int, float))
            and math.isfinite(self.expires_at)
            and monotonic() < self.expires_at
            and isinstance(self.revoked, Event)
            and not self.revoked.is_set()
        )


@dataclass(frozen=True)
class ContextRequest:
    session_id: str
    response_key: str
    text: str
    scope: ContextScope | None
    selected_language: str | None
    input_language: str | None
    deadline: float  # time.perf_counter() in this process
    max_bytes: int
    anchor_id: str
    input_generation: int
    turn_id: str | None
    turn_revision: int | None


class ContextProvider(Protocol):
    async def prepare_context(self, request: ContextRequest) -> tuple[str, ...] | None: ...


@dataclass(frozen=True)
class ContextIntegration:
    context_factory: Callable[[], AbstractAsyncContextManager[ContextProvider]]
    wait_seconds: float = _MAX_WAIT_SECONDS
    scope_middleware: Callable[[ASGIApp], ASGIApp] | None = None

    def __post_init__(self):
        object.__setattr__(self, "wait_seconds", _bounded_wait(self.wait_seconds))


def configured_integration() -> ContextIntegration | None:
    override = os.environ.get("S2S_CONTEXT_BINDING")
    path = Path(override) if override is not None else Path(sys.prefix, "etc/speech-to-speech/context.json")
    if not path.is_absolute():
        raise ValueError("S2S_CONTEXT_BINDING must be absolute")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        if override is not None:
            raise
        return None
    if not isinstance(record, dict) or type(record.get("format")) is not int or record["format"] != 1:
        raise ValueError("Unsupported context binding format")
    if type(record.get("enabled", True)) is not bool:
        raise ValueError("Invalid context binding enabled flag")
    if record.get("enabled") is False:
        return None
    if any(
        not isinstance(record.get(key), str) or not record[key].strip() for key in ("distribution", "entry", "config")
    ):
        raise ValueError("Expected a distribution, entry and config path in the context binding")
    config = Path(record["config"])
    if not config.is_absolute():
        raise ValueError("Expected an absolute context configuration path")
    entries = distribution(record["distribution"]).entry_points.select(
        group="speech_to_speech.context_extensions.v1", name=record["entry"]
    )
    if len(entries) != 1:
        raise ValueError("Expected exactly one configured context extension")
    integration = next(iter(entries)).load()(config)
    if not isinstance(integration, ContextIntegration):
        raise ValueError("Configured extension must return a ContextIntegration")
    return integration


def render_records(records: object, budget: int) -> str | None:
    if type(records) is not tuple or len(records) > 4 or any(type(x) is not str for x in records):
        return None
    accepted: list[str] = []
    block = None
    for record in records:
        if len(record) > budget or not record.strip():
            continue
        candidate = "Reference data " + json.dumps([*accepted, record], ensure_ascii=False)
        try:
            size = len(candidate.encode("utf-8"))
        except UnicodeEncodeError:
            continue
        if size <= budget:
            block = candidate
            accepted.append(record)
    return block


@dataclass(frozen=True)
class PreparedContext:
    anchor: tuple[int, str, tuple[str, ...]]
    block: str
    current: Callable[[], bool] = field(repr=False)

    def project(self, view) -> bool:
        if self.current() and view.context_anchor() == self.anchor:
            from openai.types.realtime.realtime_conversation_item_user_message import Content

            view.buffer[-1].content.append(Content(type="input_text", text=self.block))
            return True
        return False


class ContextDispatcher:
    """Owned by the serving loop. Slow lookups retain admission until they finish."""

    def __init__(self, provider: ContextProvider, *, wait_seconds: float = _MAX_WAIT_SECONDS, capacity: int = 4):
        self.provider = provider
        self.wait_seconds = _bounded_wait(wait_seconds)
        self.capacity = capacity
        self.active: dict[str, asyncio.Task[tuple[str, ...] | None]] = {}
        self.pending: dict[str, Callable[[], None]] = {}
        self.dispatches: set[asyncio.Task[None]] = set()
        self.closed = False

    def submit(self, session, request, queue, scope, current, original_text=None, input_item_id=None, *, retrieve=True):
        # Preserve pipeline order when a later turn bypasses outstanding recall.
        previous = self.pending.pop(session, None)
        if previous is not None:
            previous()
        if (
            not retrieve
            or self.closed
            or self.wait_seconds <= 0
            or session in self.active
            or len(self.active) >= self.capacity
        ):
            queue.put(request)
            return
        anchor = request.runtime_config.chat.context_anchor()
        if anchor is None or (scope is not None and not scope.valid()):
            queue.put(request)
            return
        generation, item_id, parts = anchor
        if original_text is not None and item_id != input_item_id:
            queue.put(request)
            return
        text = original_text if original_text is not None else " ".join(parts)
        if not text or len(text) > 2048 or len(text.encode("utf-8", errors="surrogatepass")) > 2048:
            queue.put(request)
            return
        deadline = perf_counter() + self.wait_seconds
        query = ContextRequest(
            session,
            request.response_key,
            text,
            scope,
            request.selected_language,
            request.language_code,
            deadline,
            1024,
            item_id,
            generation,
            request.turn_id,
            request.turn_revision,
        )

        async def prepare():
            return await self.provider.prepare_context(query)

        lookup = asyncio.create_task(prepare())
        self.active[session] = lookup

        def finished(task):
            self.active.pop(session, None)
            if not task.cancelled() and task.exception() is not None:
                logger.warning("Context lookup failed")

        lookup.add_done_callback(finished)

        def ordinary():
            if not self.closed and current():
                queue.put(request)

        self.pending[session] = ordinary

        async def dispatch():
            # Some asyncio timer implementations wake one clock tick early.
            while not lookup.done():
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    break
                await asyncio.wait({lookup}, timeout=remaining)
            done = lookup.done()
            if self.pending.get(session) is not ordinary:
                return
            del self.pending[session]
            if self.closed or not current():
                return
            logger.debug("Context preparation finished on time=%s completed=%s", perf_counter() <= deadline, bool(done))
            if done and perf_counter() <= deadline and not lookup.cancelled() and lookup.exception() is None:
                block = render_records(lookup.result(), query.max_bytes)
                logger.debug("Context provider supplied a usable block=%s", block is not None)
                if block and (scope is None or scope.valid()):

                    def still_authorized():
                        return (
                            current()
                            and request.runtime_config.chat.context_anchor() == anchor
                            and (scope is None or scope.valid())
                        )

                    request.prepared_context = PreparedContext(anchor, block, still_authorized)
            queue.put(request)

        task = asyncio.create_task(dispatch())
        self.dispatches.add(task)
        task.add_done_callback(self.dispatches.discard)

    async def close(self):
        self.closed = True
        self.pending.clear()
        tasks = [*self.dispatches, *self.active.values()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
