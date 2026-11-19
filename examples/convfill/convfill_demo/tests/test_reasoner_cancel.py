"""ConvFillSystem must abort the Reasoner stream when a turn is cancelled."""

from __future__ import annotations

import queue
import threading
import time

from convfill_demo.runtime.run_convfill import ConvFillSystem


class _SlowStreamingBackend:
    """Mimics a provider stream: one sentence every 20 ms, records teardown."""

    def __init__(self, sentences=40):
        self.sentences = sentences
        self.closed = threading.Event()
        self.yielded = 0

    def backend_infer(self):
        try:
            for i in range(self.sentences):
                time.sleep(0.02)
                self.yielded += 1
                yield f"Fact number {i}."
        finally:
            # Mirrors exiting ``with client.messages.stream(...)``.
            self.closed.set()


def _bare_system(backend):
    system = ConvFillSystem.__new__(ConvFillSystem)
    system.thought_queue = queue.Queue()
    system.tts_queue = queue.Queue()
    system.done_event = threading.Event()
    system.cancel_event = threading.Event()
    system.convfill_backend = backend
    system.on_thought = None
    system._frontend_inflight = 0
    system._frontend_inflight_lock = threading.Lock()
    system.audio_in_flight_fn = lambda: 1  # never emit <sil> in this test
    return system


def test_cancel_turn_closes_the_reasoner_stream():
    backend = _SlowStreamingBackend()
    system = _bare_system(backend)

    thread = threading.Thread(target=system.run_backend, daemon=True)
    thread.start()

    # Let a few sentences through, then cancel mid-stream.
    deadline = time.time() + 2.0
    while system.thought_queue.qsize() < 3 and time.time() < deadline:
        time.sleep(0.005)
    assert system.thought_queue.qsize() >= 3
    system.cancel_turn()

    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert backend.closed.wait(timeout=1.0), "provider stream was not closed after cancel"
    assert backend.yielded < backend.sentences, "stream was drained to completion despite cancel"


def test_uncancelled_turn_drains_and_closes_the_stream():
    backend = _SlowStreamingBackend(sentences=5)
    system = _bare_system(backend)
    system.run_backend()

    assert backend.closed.is_set()
    assert backend.yielded == 5
    assert system.done_event.is_set()
    drained = []
    while not system.thought_queue.empty():
        drained.append(system.thought_queue.get_nowait())
    assert drained == [f"Fact number {i}." for i in range(5)]
