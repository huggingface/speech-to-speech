from __future__ import annotations

from concurrent.futures import Future
from threading import Event, Lock
from time import monotonic, sleep

import pytest

from speech_to_speech.STT.endpoint_admission import (
    AdmissionRejected,
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


class UncooperativeOperation(ControlledOperation):
    def cancel(self, reason: str) -> None:
        del reason
        self.cancelled.set()


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


def test_superseded_pending_progressive_is_never_materialized():
    starts: list[str] = []
    materialized: list[str] = []
    lock = Lock()
    active = ControlledOperation("active", starts, lock)
    superseded = ControlledOperation("superseded", starts, lock)
    latest = ControlledOperation("latest", starts, lock)
    controller = EndpointAdmissionController(
        "test",
        EndpointAdmissionSettings(max_concurrency=1, max_queue_size=2, progressive_min_interval_s=0),
    )

    def request(name: str, operation: ControlledOperation) -> TranscriptionAdmissionRequest[str]:
        def operation_factory() -> ControlledOperation:
            materialized.append(name)
            return operation

        return TranscriptionAdmissionRequest(
            request_id=name,
            owner_id="pipeline-1",
            turn_id="turn-1",
            turn_revision=0,
            mode="progressive",
            operation_factory=operation_factory,
        )

    active_future = controller.submit(request("active", active))
    _wait_until(lambda: starts == ["active"])
    superseded_future = controller.submit(request("superseded", superseded))
    latest_future = controller.submit(request("latest", latest))

    with pytest.raises(TranscriptionCancelled) as cancelled:
        superseded_future.result(timeout=1)
    assert cancelled.value.reason == "superseded"
    assert materialized == ["active"]

    active.release.set()
    assert active_future.result(timeout=1) == "active"
    _wait_until(lambda: starts == ["active", "latest"])
    latest.release.set()
    assert latest_future.result(timeout=1) == "latest"
    assert materialized == ["active", "latest"]
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


def test_full_queue_rejects_unrelated_final_instead_of_superseding_it():
    starts: list[str] = []
    lock = Lock()
    active = ControlledOperation("active", starts, lock)
    queued = ControlledOperation("queued", starts, lock)
    rejected = ControlledOperation("rejected", starts, lock)
    controller = EndpointAdmissionController(
        "test",
        EndpointAdmissionSettings(max_concurrency=1, max_queue_size=1, progressive_min_interval_s=0),
    )

    active_future = controller.submit(_request("active", "final", active, owner="owner-1", turn="turn-1"))
    _wait_until(lambda: starts == ["active"])
    queued_future = controller.submit(_request("queued", "final", queued, owner="owner-2", turn="turn-2"))
    with controller._condition:
        assert controller._next_wait_s_locked() == controller._RELEVANCE_POLL_S
    rejected_future = controller.submit(_request("rejected", "final", rejected, owner="owner-3", turn="turn-3"))

    with pytest.raises(AdmissionRejected, match="queue is full"):
        rejected_future.result(timeout=1)
    assert not rejected.cancelled.is_set()

    active.release.set()
    assert active_future.result(timeout=1) == "active"
    _wait_until(lambda: starts == ["active", "queued"])
    queued.release.set()
    assert queued_future.result(timeout=1) == "queued"
    controller.close()


def test_new_session_owner_dispatches_while_cancelled_old_transport_is_still_active():
    starts: list[str] = []
    lock = Lock()
    old = UncooperativeOperation("old", starts, lock)
    new = ControlledOperation("new", starts, lock)
    controller = EndpointAdmissionController(
        "test",
        EndpointAdmissionSettings(max_concurrency=2, max_queue_size=2, progressive_min_interval_s=0),
    )

    old_future = controller.submit(_request("old", "final", old, owner="session-1", turn="turn-1"))
    _wait_until(lambda: starts == ["old"])
    assert controller.cancel(CancelTranscription(owner_id="session-1", reason="session_end")) == 1
    with pytest.raises(TranscriptionCancelled):
        old_future.result(timeout=1)

    new_future = controller.submit(_request("new", "final", new, owner="session-2", turn="turn-1"))
    _wait_until(lambda: starts == ["old", "new"])
    new.release.set()
    assert new_future.result(timeout=1) == "new"

    old.release.set()
    _wait_until(lambda: controller.active_count == 0)
    controller.close()


def test_key_blocked_pending_request_uses_relevance_poll_wait():
    starts: list[str] = []
    lock = Lock()
    active = ControlledOperation("active", starts, lock)
    pending = ControlledOperation("pending", starts, lock)
    controller = EndpointAdmissionController(
        "test",
        EndpointAdmissionSettings(max_concurrency=2, max_queue_size=2, progressive_min_interval_s=0),
    )

    active_future = controller.submit(_request("active", "progressive", active))
    _wait_until(lambda: starts == ["active"])
    pending_future = controller.submit(_request("pending", "progressive", pending))
    _wait_until(lambda: controller.pending_count == 1)

    with controller._condition:
        assert controller._next_wait_s_locked() == controller._RELEVANCE_POLL_S

    active.release.set()
    assert active_future.result(timeout=1) == "active"
    _wait_until(lambda: starts == ["active", "pending"])
    pending.release.set()
    assert pending_future.result(timeout=1) == "pending"
    assert ("pipeline-1", "turn-1") not in controller._last_progressive_dispatch
    controller.close()
