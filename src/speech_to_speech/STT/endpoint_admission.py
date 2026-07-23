from __future__ import annotations

import hashlib
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Condition, Lock, Thread
from time import monotonic
from typing import Callable, Generic, Literal, Protocol, TypeVar
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
TranscriptionMode = Literal["progressive", "final"]
CancellationReason = Literal[
    "superseded",
    "final_received",
    "turn_reopened",
    "session_end",
    "deadline_exceeded",
    "shutdown",
]


class CancellableOperation(Protocol[T_co]):
    """One operation whose transport can be closed from another thread."""

    def run(self) -> T_co: ...

    def cancel(self, reason: CancellationReason) -> None: ...


class TranscriptionCancelled(RuntimeError):
    def __init__(self, request_id: str, reason: CancellationReason) -> None:
        super().__init__(f"transcription request {request_id} cancelled: {reason}")
        self.request_id = request_id
        self.reason = reason


class AdmissionRejected(RuntimeError):
    pass


@dataclass(frozen=True)
class CancelTranscription:
    """Explicit cancellation message consumed by the endpoint operation owner."""

    owner_id: str
    reason: CancellationReason
    request_id: str | None = None
    turn_id: str | None = None
    turn_revision: int | None = None


@dataclass(frozen=True)
class EndpointAdmissionSettings:
    max_concurrency: int = 1
    max_queue_size: int = 8
    progressive_min_interval_s: float = 0.75

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if self.max_queue_size < 1:
            raise ValueError("max_queue_size must be >= 1")
        if self.progressive_min_interval_s < 0:
            raise ValueError("progressive_min_interval_s must be >= 0")


@dataclass(frozen=True)
class TranscriptionAdmissionRequest(Generic[T]):
    request_id: str
    owner_id: str
    turn_id: str
    turn_revision: int
    mode: TranscriptionMode
    operation_factory: Callable[[], CancellableOperation[T]]
    is_relevant: Callable[[], bool] = lambda: True

    @property
    def supersession_key(self) -> tuple[str, str]:
        return (self.owner_id, self.turn_id)


@dataclass
class _AdmissionItem(Generic[T]):
    request: TranscriptionAdmissionRequest[T]
    future: Future[T]
    sequence: int
    submitted_at_s: float = field(default_factory=monotonic)
    operation: CancellableOperation[T] | None = None
    cancelled_reason: CancellationReason | None = None


class EndpointAdmissionController:
    """Bounded, endpoint-wide scheduler for transcription operations.

    The controller owns operation dispatch and cancellation. Handlers only map
    pipeline messages to requests and map completed results back to messages.
    """

    _RELEVANCE_POLL_S = 0.05

    def __init__(self, safe_endpoint_label: str, settings: EndpointAdmissionSettings) -> None:
        self.safe_endpoint_label = safe_endpoint_label
        self.settings = settings
        self._condition = Condition()
        self._pending: list[_AdmissionItem] = []
        self._active: dict[str, _AdmissionItem] = {}
        self._last_progressive_dispatch: dict[tuple[str, str], float] = {}
        self._sequence = 0
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=settings.max_concurrency,
            thread_name_prefix="stt-http",
        )
        self._scheduler = Thread(
            target=self._scheduler_loop,
            name=f"stt-admission-{safe_endpoint_label}",
            daemon=True,
        )
        self._scheduler.start()

    def submit(self, request: TranscriptionAdmissionRequest[T]) -> Future[T]:
        future: Future[T] = Future()
        item = _AdmissionItem(request=request, future=future, sequence=0)
        cancelled: list[_AdmissionItem] = []

        with self._condition:
            if self._closed:
                future.set_exception(AdmissionRejected("endpoint admission controller is closed"))
                return future

            self._sequence += 1
            item.sequence = self._sequence

            newest_revision = self._newest_revision_locked(request.supersession_key)
            if newest_revision is not None and request.turn_revision < newest_revision:
                item.cancelled_reason = "superseded"
                cancelled.append(item)
            else:
                cancelled.extend(self._apply_supersession_locked(item))

            if item.cancelled_reason is None and len(self._pending) >= self.settings.max_queue_size:
                if request.mode == "final":
                    progressive = min(
                        (candidate for candidate in self._pending if candidate.request.mode == "progressive"),
                        key=lambda candidate: candidate.sequence,
                        default=None,
                    )
                    if progressive is not None:
                        self._mark_cancelled_locked(progressive, "superseded")
                        cancelled.append(progressive)
                if len(self._pending) >= self.settings.max_queue_size:
                    item.cancelled_reason = "superseded"
                    cancelled.append(item)

            if item.cancelled_reason is None:
                self._pending.append(item)
                self._condition.notify_all()

        self._complete_cancellations(cancelled)
        return future

    def cancel(self, message: CancelTranscription) -> int:
        cancelled: list[_AdmissionItem] = []
        with self._condition:
            for item in [*self._pending, *self._active.values()]:
                request = item.request
                if request.owner_id != message.owner_id:
                    continue
                if message.request_id is not None and request.request_id != message.request_id:
                    continue
                if message.turn_id is not None and request.turn_id != message.turn_id:
                    continue
                if message.turn_revision is not None and request.turn_revision != message.turn_revision:
                    continue
                if item.cancelled_reason is None:
                    self._mark_cancelled_locked(item, message.reason)
                    cancelled.append(item)
            self._condition.notify_all()

        self._complete_cancellations(cancelled)
        return len(cancelled)

    def close(self) -> None:
        cancelled: list[_AdmissionItem] = []
        with self._condition:
            if self._closed:
                return
            self._closed = True
            for item in [*self._pending, *self._active.values()]:
                if item.cancelled_reason is None:
                    self._mark_cancelled_locked(item, "shutdown")
                    cancelled.append(item)
            self._condition.notify_all()

        self._complete_cancellations(cancelled)
        self._scheduler.join(timeout=2.0)
        self._executor.shutdown(wait=False, cancel_futures=True)

    @property
    def pending_count(self) -> int:
        with self._condition:
            return len(self._pending)

    @property
    def active_count(self) -> int:
        with self._condition:
            return len(self._active)

    def _apply_supersession_locked(self, new_item: _AdmissionItem) -> list[_AdmissionItem]:
        request = new_item.request
        cancelled: list[_AdmissionItem] = []
        related = [
            item
            for item in [*self._pending, *self._active.values()]
            if item.request.supersession_key == request.supersession_key and item.cancelled_reason is None
        ]

        for item in related:
            existing = item.request
            reason: CancellationReason | None = None
            if existing.turn_revision < request.turn_revision:
                reason = "turn_reopened"
            elif existing.turn_revision > request.turn_revision:
                new_item.cancelled_reason = "superseded"
                break
            elif request.mode == "final" and existing.mode == "progressive":
                reason = "final_received"
            elif request.mode == "progressive" and existing.mode == "final":
                new_item.cancelled_reason = "superseded"
                break
            elif (
                request.mode == "progressive"
                and existing.mode == "progressive"
                and item in self._pending
            ):
                reason = "superseded"
            elif request.mode == "final" and existing.mode == "final":
                new_item.cancelled_reason = "superseded"
                break

            if reason is not None:
                self._mark_cancelled_locked(item, reason)
                cancelled.append(item)

        if new_item.cancelled_reason is not None:
            cancelled.append(new_item)
        return cancelled

    def _newest_revision_locked(self, key: tuple[str, str]) -> int | None:
        revisions = [
            item.request.turn_revision
            for item in [*self._pending, *self._active.values()]
            if item.request.supersession_key == key and item.cancelled_reason is None
        ]
        return max(revisions, default=None)

    def _mark_cancelled_locked(self, item: _AdmissionItem, reason: CancellationReason) -> None:
        if item.cancelled_reason is not None:
            return
        item.cancelled_reason = reason
        if item in self._pending:
            self._pending.remove(item)

    def _complete_cancellations(self, items: list[_AdmissionItem]) -> None:
        seen: set[str] = set()
        for item in items:
            request_id = item.request.request_id
            if request_id in seen:
                continue
            seen.add(request_id)
            reason = item.cancelled_reason or "superseded"
            operation = item.operation
            if operation is not None:
                try:
                    operation.cancel(reason)
                except Exception:
                    logger.debug("Error closing cancelled STT transport", exc_info=True)
            if not item.future.done():
                item.future.set_exception(TranscriptionCancelled(request_id, reason))

    def _scheduler_loop(self) -> None:
        while True:
            dispatch: list[_AdmissionItem] = []
            cancelled: list[_AdmissionItem] = []
            with self._condition:
                if self._closed and not self._active:
                    return

                for item in [*self._pending, *self._active.values()]:
                    if item.cancelled_reason is None and not item.request.is_relevant():
                        self._mark_cancelled_locked(item, "superseded")
                        cancelled.append(item)

                while len(self._active) < self.settings.max_concurrency:
                    next_item = self._select_next_locked()
                    if next_item is None:
                        break
                    self._pending.remove(next_item)
                    self._active[next_item.request.request_id] = next_item
                    if next_item.request.mode == "progressive":
                        self._last_progressive_dispatch[next_item.request.supersession_key] = monotonic()
                    dispatch.append(next_item)

                if not dispatch and not cancelled:
                    self._condition.wait(self._next_wait_s_locked())

            self._complete_cancellations(cancelled)
            for item in dispatch:
                self._executor.submit(self._execute, item)

    def _select_next_locked(self) -> _AdmissionItem | None:
        now = monotonic()
        active_keys = {
            item.request.supersession_key
            for item in self._active.values()
        }
        candidates = [
            item
            for item in self._pending
            if item.cancelled_reason is None
            and item.request.supersession_key not in active_keys
            and self._ready_at_locked(item) <= now
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda item: (0 if item.request.mode == "final" else 1, item.sequence),
        )

    def _ready_at_locked(self, item: _AdmissionItem) -> float:
        if item.request.mode == "final":
            return item.submitted_at_s
        last_dispatch = self._last_progressive_dispatch.get(item.request.supersession_key)
        if last_dispatch is None:
            return item.submitted_at_s
        return max(item.submitted_at_s, last_dispatch + self.settings.progressive_min_interval_s)

    def _next_wait_s_locked(self) -> float:
        if not self._pending:
            return self._RELEVANCE_POLL_S
        now = monotonic()
        ready_times = [self._ready_at_locked(item) for item in self._pending if item.cancelled_reason is None]
        if not ready_times:
            return self._RELEVANCE_POLL_S
        return max(0.001, min(self._RELEVANCE_POLL_S, min(ready_times) - now))

    def _execute(self, item: _AdmissionItem[T]) -> None:
        result: T | None = None
        error: BaseException | None = None
        operation: CancellableOperation[T] | None = None

        try:
            with self._condition:
                cancelled = item.cancelled_reason is not None or not item.request.is_relevant()
                if cancelled and item.cancelled_reason is None:
                    item.cancelled_reason = "superseded"
            if cancelled:
                raise TranscriptionCancelled(item.request.request_id, item.cancelled_reason or "superseded")

            # The factory performs WAV serialization. It is intentionally called
            # only after the final pre-dispatch relevance check.
            operation = item.request.operation_factory()
            with self._condition:
                item.operation = operation
                cancelled = item.cancelled_reason is not None or not item.request.is_relevant()
                if cancelled and item.cancelled_reason is None:
                    item.cancelled_reason = "superseded"
            if cancelled:
                operation.cancel(item.cancelled_reason or "superseded")
                raise TranscriptionCancelled(item.request.request_id, item.cancelled_reason or "superseded")

            result = operation.run()
        except BaseException as exc:
            error = exc
        finally:
            with self._condition:
                self._active.pop(item.request.request_id, None)
                item.operation = None
                cancelled_reason = item.cancelled_reason
                self._condition.notify_all()

        if item.future.done():
            return
        if cancelled_reason is not None:
            item.future.set_exception(TranscriptionCancelled(item.request.request_id, cancelled_reason))
        elif error is not None:
            item.future.set_exception(error)
        else:
            item.future.set_result(result)  # type: ignore[arg-type]


@dataclass
class _RegistryEntry:
    controller: EndpointAdmissionController
    settings: EndpointAdmissionSettings
    references: int = 0


class EndpointAdmissionLease:
    def __init__(self, registry_key: tuple[str, str], controller: EndpointAdmissionController) -> None:
        self._registry_key = registry_key
        self.controller = controller
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        EndpointAdmissionRegistry._release(self._registry_key)


class EndpointAdmissionRegistry:
    """Process-wide controller registry keyed by normalized endpoint and auth."""

    _lock = Lock()
    _entries: dict[tuple[str, str], _RegistryEntry] = {}

    @classmethod
    def acquire(
        cls,
        base_url: str,
        api_key: str | None,
        settings: EndpointAdmissionSettings,
    ) -> EndpointAdmissionLease:
        normalized_url, safe_label = normalize_endpoint(base_url)
        auth_digest = hashlib.sha256((api_key or "").encode()).hexdigest()
        key = (normalized_url, auth_digest)
        with cls._lock:
            entry = cls._entries.get(key)
            if entry is None:
                entry = _RegistryEntry(
                    controller=EndpointAdmissionController(safe_endpoint_label=safe_label, settings=settings),
                    settings=settings,
                )
                cls._entries[key] = entry
            elif entry.settings != settings:
                logger.warning(
                    "STT endpoint %s already has admission settings %s; reusing them instead of %s",
                    safe_label,
                    entry.settings,
                    settings,
                )
            entry.references += 1
            return EndpointAdmissionLease(key, entry.controller)

    @classmethod
    def _release(cls, key: tuple[str, str]) -> None:
        controller: EndpointAdmissionController | None = None
        with cls._lock:
            entry = cls._entries.get(key)
            if entry is None:
                return
            entry.references -= 1
            if entry.references <= 0:
                controller = entry.controller
                del cls._entries[key]
        if controller is not None:
            controller.close()

    @classmethod
    def close_all(cls) -> None:
        with cls._lock:
            controllers = [entry.controller for entry in cls._entries.values()]
            cls._entries.clear()
        for controller in controllers:
            controller.close()


def normalize_endpoint(base_url: str) -> tuple[str, str]:
    value = base_url.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Invalid HTTP endpoint base URL: {base_url!r}")

    hostname = parsed.hostname.lower()
    port = parsed.port
    default_port = (parsed.scheme == "http" and port == 80) or (parsed.scheme == "https" and port == 443)
    host_port = hostname if port is None or default_port else f"{hostname}:{port}"
    normalized = urlunsplit((parsed.scheme.lower(), host_port, parsed.path.rstrip("/"), "", ""))
    safe_label = f"{host_port}{parsed.path.rstrip('/') or '/'}"
    return normalized, safe_label
