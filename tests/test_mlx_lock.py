"""The MLX lock lets a final transcription in between the steps of a long hold."""

import threading
import time

import pytest

from speech_to_speech.utils.mlx_lock import (
    MLXLockContext,
    acquire_mlx_lock,
    hand_over_mlx_lock,
    release_mlx_lock,
    suspend_mlx_lock,
)


def _waiter(order: list[str], name: str, priority: bool, started: threading.Event) -> threading.Thread:
    def run() -> None:
        started.set()
        with MLXLockContext(handler_name=name, timeout=5.0, priority=priority):
            order.append(name)
            time.sleep(0.05)

    thread = threading.Thread(target=run)
    thread.start()
    return thread


def _hold_in_steps(order: list[str], steps: int, waiter_queued: threading.Event) -> None:
    """A reply being synthesized: hold the lock and hand it over between steps."""
    with MLXLockContext(handler_name="TTS", timeout=5.0):
        for step in range(steps):
            order.append(f"tts{step}")
            if step == 0:
                waiter_queued.wait(1.0)
                time.sleep(0.05)  # the waiter is now blocked on the lock
            hand_over_mlx_lock("TTS")


def test_a_priority_waiter_runs_between_two_steps_of_a_long_hold():
    order: list[str] = []
    queued = threading.Event()
    holder = threading.Thread(target=_hold_in_steps, args=(order, 3, queued))
    holder.start()
    while not order:
        time.sleep(0.001)
    final = _waiter(order, "stt-final", True, queued)
    holder.join(5)
    final.join(5)
    assert order == ["tts0", "stt-final", "tts1", "tts2"]


def test_an_ordinary_waiter_waits_for_the_whole_hold():
    """A local LLM may hold the lock for seconds: it must not stall a reply mid-sentence."""
    order: list[str] = []
    queued = threading.Event()
    holder = threading.Thread(target=_hold_in_steps, args=(order, 3, queued))
    holder.start()
    while not order:
        time.sleep(0.001)
    llm = _waiter(order, "llm", False, queued)
    holder.join(5)
    llm.join(5)
    assert order == ["tts0", "tts1", "tts2", "llm"]


@pytest.mark.parametrize("final_first", [False, True])
def test_no_hand_over_while_an_ordinary_waiter_is_queued(final_first):
    """A release wakes every queued thread, so handing over to a final would let a queued LLM in too."""
    order: list[str] = []
    queued = threading.Event()
    holder = threading.Thread(target=_hold_in_steps, args=(order, 3, queued))
    holder.start()
    while not order:
        time.sleep(0.001)
    first, second = (("stt-final", True), ("llm", False)) if final_first else (("llm", False), ("stt-final", True))
    first_started = threading.Event()
    waiters = [_waiter(order, *first, first_started)]
    first_started.wait(1.0)
    time.sleep(0.01)  # the first waiter is blocked on the lock before the second starts
    waiters.append(_waiter(order, *second, queued))
    holder.join(5)
    for waiter in waiters:
        waiter.join(5)
    assert order[:3] == ["tts0", "tts1", "tts2"]
    assert sorted(order[3:]) == ["llm", "stt-final"]


def test_hand_over_without_waiters_or_without_the_lock_does_nothing():
    assert hand_over_mlx_lock("TTS") is False
    with MLXLockContext(handler_name="TTS"):
        assert hand_over_mlx_lock("TTS") is False
        with MLXLockContext(handler_name="TTS"):
            # Held re-entrantly: releasing once would not free it.
            assert hand_over_mlx_lock("TTS") is False


def test_suspend_mlx_lock_releases_and_restores_reentrant_owner():
    with suspend_mlx_lock():  # no-op when unheld
        pass

    def try_acquire(timeout: float) -> bool:
        got: list[bool] = []

        def run() -> None:
            ok = acquire_mlx_lock(timeout=timeout, handler_name="Other")
            got.append(ok)
            if ok:
                release_mlx_lock(handler_name="Other")

        t = threading.Thread(target=run)
        t.start()
        t.join(1.0)
        return got == [True]

    with MLXLockContext(handler_name="TTS"), MLXLockContext(handler_name="TTS"):
        with suspend_mlx_lock():
            assert try_acquire(0.1) is True
        assert try_acquire(0.02) is False
