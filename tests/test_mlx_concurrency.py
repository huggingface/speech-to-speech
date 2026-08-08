from __future__ import annotations

import threading
from time import monotonic

from speech_to_speech.utils.mlx_concurrency import _FairConcurrencyGate


def _wait_for_waiter_count(gate: _FairConcurrencyGate, expected: int) -> None:
    deadline = monotonic() + 1.0
    while monotonic() < deadline:
        with gate._condition:
            if len(gate._waiters) == expected:
                return
        threading.Event().wait(0.001)
    raise AssertionError(f"Expected {expected} queued MLX operations")


def test_gate_allows_two_operations_and_queues_the_third() -> None:
    gate = _FairConcurrencyGate(2)
    release = threading.Event()
    two_entered = threading.Event()
    entered: list[int] = []
    entered_lock = threading.Lock()

    def worker(index: int) -> None:
        assert gate.acquire(timeout=1.0)
        with entered_lock:
            entered.append(index)
            if len(entered) == 2:
                two_entered.set()
        assert release.wait(timeout=1.0)
        gate.release()

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(3)]
    for thread in threads:
        thread.start()

    assert two_entered.wait(timeout=1.0)
    with entered_lock:
        assert len(entered) == 2

    release.set()
    for thread in threads:
        thread.join(timeout=1.0)
        assert not thread.is_alive()
    assert sorted(entered) == [0, 1, 2]


def test_gate_serves_queued_operations_in_arrival_order() -> None:
    gate = _FairConcurrencyGate(1)
    assert gate.acquire()
    entered: list[int] = []

    def worker(index: int) -> None:
        assert gate.acquire(timeout=1.0)
        entered.append(index)
        gate.release()

    threads = []
    for index in range(3):
        thread = threading.Thread(target=worker, args=(index,))
        thread.start()
        threads.append(thread)
        _wait_for_waiter_count(gate, index + 1)

    gate.release()
    for thread in threads:
        thread.join(timeout=1.0)
        assert not thread.is_alive()
    assert entered == [0, 1, 2]


def test_gate_timeout_removes_waiter() -> None:
    gate = _FairConcurrencyGate(1)
    assert gate.acquire()
    acquired: list[bool] = []

    thread = threading.Thread(target=lambda: acquired.append(gate.acquire(timeout=0.01)))
    thread.start()
    thread.join(timeout=1.0)

    assert acquired == [False]
    with gate._condition:
        assert not gate._waiters
    gate.release()


def test_gate_is_reentrant_for_nested_handler_calls() -> None:
    gate = _FairConcurrencyGate(1)

    assert gate.acquire()
    assert gate.acquire(timeout=0.0)
    gate.release()
    gate.release()

    acquired: list[bool] = []

    def worker() -> None:
        acquired.append(gate.acquire(timeout=0.1))
        if acquired[-1]:
            gate.release()

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=1.0)
    assert acquired == [True]
