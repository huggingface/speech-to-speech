"""The MLX lock lets a final transcription in between the steps of a long hold."""

import threading
import time

from speech_to_speech.utils.mlx_lock import MLXLockContext, hand_over_mlx_lock


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


def test_hand_over_without_waiters_or_without_the_lock_does_nothing():
    assert hand_over_mlx_lock("TTS") is False
    with MLXLockContext(handler_name="TTS"):
        assert hand_over_mlx_lock("TTS") is False
        with MLXLockContext(handler_name="TTS"):
            # Held re-entrantly: releasing once would not free it.
            assert hand_over_mlx_lock("TTS") is False
