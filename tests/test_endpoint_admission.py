from __future__ import annotations

from concurrent.futures import Future
from threading import Event, Lock
from time import monotonic, sleep

import pytest

from speech_to_speech.STT.endpoint_admission import (
    CancelTranscription,
    EndpointAdmissionController,
    EndpointAdmissionRegistry,
    EndpointAdmissionSettings,
    TranscriptionAdmissionRequest,
    TranscriptionCancelled,
)


class ControlledOperation:
    def __init__(self, name: str, starts: list[str], lock: Lock, release: Event | None = None) -> None:
        self.name = name
        self.starts = starts
        self.lock = lock
        self.release = release or Event()
        self.cancelled = Event()

    def run(self) -> str:
        with self.lock:
            self.starts.append(self.name)
        while not self.release.wait(0.01):
            if self.cancelled.is_set():
                raise RuntimeError("transport closed")
        return self.name

    def cancel(self, reason: str) -> None:
        del reason
        self.cancelled.set()
        self.release.set()


def _request(
    name: str,
    mode: str,
    operation: ControlledOperation,
    *,
    revision: int = 0,
    owner: str = "pipeline-1",
    turn: str = "turn-1",
) -> TranscriptionAdmissionRequest[str]:
    return TranscriptionAdmissionRequest(
        request_id=name,
        owner_id=owner,
        turn_id=turn,
        turn_revision=revision,
        mode=mode,  # type: ignore[arg-type]
        operation_factory=lambda: operation,
    )


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return
        sleep(0.01)
    raise AssertionError("condition did not become true")


def test_progressive_burst_keeps_only_active_and_latest_pending():
    starts: list[str] = []
    lock = Lock()
    first_release = Event()
    controller = EndpointAdmissionController(
        "test",
        EndpointAdmissionSettings(max_concurrency=1, max_queue_size=4, progressive_min_interval_s=0.05),
    )
    operations = [ControlledOperation(f"p{i}", starts, lock, first_release if i == 0 else Event()) for i in range(10)]

    futures: list[Future[str]] = []
    futures.append(controller.submit(_request("p0", "progressive", operations[0])))
    _wait_until(lambda: starts == ["p0"])
    for index in range(1, 10):
        futures.append(controller.submit(_request(f"p{index}", "progressive", operations[index])))

    assert controller.active_count == 1
    assert controller.pending_count == 1
    first_release.set()
    _wait_until(lambda: starts == ["p0", "p9"])
    operations[9].release.set()
    assert futures[0].result(timeout=1) == "p0"
    assert futures[9].result(timeout=1) == "p9"
    assert sum(isinstance(future.exception(), TranscriptionCancelled) for future in futures[1:9]) == 8
    assert starts == ["p0", "p9"]
    controller.close()


def test_final_cancels_active_progressive_and_dispatches_next():
    starts: list[str] = []
    lock = Lock()
    progressive = ControlledOperation("progressive", starts, lock)
    final = ControlledOperation("final", starts, lock)
    controller = EndpointAdmissionController(
        "test",
        EndpointAdmissionSettings(max_concurrency=1, max_queue_size=4, progressive_min_interval_s=0),
    )

    progressive_future = controller.submit(_request("progressive", "progressive", progressive))
    _wait_until(lambda: starts == ["progressive"])
    final_future = controller.submit(_request("final", "final", final))

    with pytest.raises(TranscriptionCancelled) as cancelled:
        progressive_future.result(timeout=1)
    assert cancelled.value.reason == "final_received"
    _wait_until(lambda: starts == ["progressive", "final"])
    final.release.set()
    assert final_future.result(timeout=1) == "final"
    assert progressive.cancelled.is_set()
    controller.close()


def test_new_revision_cancels_old_revision_and_session_message_is_idempotent():
    starts: list[str] = []
    lock = Lock()
    old = ControlledOperation("old", starts, lock)
    new = ControlledOperation("new", starts, lock)
    controller = EndpointAdmissionController(
        "test",
        EndpointAdmissionSettings(max_concurrency=1, max_queue_size=4, progressive_min_interval_s=0),
    )

    old_future = controller.submit(_request("old", "final", old, revision=0))
    _wait_until(lambda: starts == ["old"])
    new_future = controller.submit(_request("new", "final", new, revision=1))
    with pytest.raises(TranscriptionCancelled) as cancelled:
        old_future.result(timeout=1)
    assert cancelled.value.reason == "turn_reopened"
    _wait_until(lambda: starts == ["old", "new"])

    message = CancelTranscription(owner_id="pipeline-1", reason="session_end")
    assert controller.cancel(message) == 1
    assert controller.cancel(message) == 0
    with pytest.raises(TranscriptionCancelled):
        new_future.result(timeout=1)
    controller.close()


def test_registry_shares_capacity_for_same_endpoint_but_not_other_endpoints():
    settings = EndpointAdmissionSettings(max_concurrency=1, max_queue_size=2, progressive_min_interval_s=0)
    first = EndpointAdmissionRegistry.acquire("HTTP://LOCALHOST:80/v1/", "secret", settings)
    second = EndpointAdmissionRegistry.acquire("http://localhost/v1", "secret", settings)
    other = EndpointAdmissionRegistry.acquire("http://localhost:9000/v1", "secret", settings)

    assert first.controller is second.controller
    assert first.controller is not other.controller

    first.release()
    second.release()
    other.release()


def test_active_operation_is_cancelled_when_relevance_changes_without_new_submission():
    starts: list[str] = []
    lock = Lock()
    operation = ControlledOperation("active", starts, lock)
    relevant = True
    controller = EndpointAdmissionController(
        "test",
        EndpointAdmissionSettings(max_concurrency=1, max_queue_size=2, progressive_min_interval_s=0),
    )
    request = TranscriptionAdmissionRequest(
        request_id="active",
        owner_id="pipeline-1",
        turn_id="turn-1",
        turn_revision=0,
        mode="final",
        operation_factory=lambda: operation,
        is_relevant=lambda: relevant,
    )

    future = controller.submit(request)
    _wait_until(lambda: starts == ["active"])
    relevant = False

    with pytest.raises(TranscriptionCancelled):
        future.result(timeout=1)
    assert operation.cancelled.is_set()
    controller.close()
