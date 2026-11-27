from __future__ import annotations

from queue import Empty, Queue
from threading import Event, Thread
from time import sleep
from typing import Iterator, Literal

import numpy as np

from speech_to_speech.pipeline.messages import PIPELINE_END, PartialTranscription, Transcription, VADAudio
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.pipeline.turn_latency import TurnLatencyStore
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler
from tests.turns import reopen


class RecordingSTTHandler(BaseSTTHandler):
    def setup(
        self,
        speculative_turns: SpeculativeTurnTracker | None = None,
        mark_stale_during_process: bool = False,
        final_revision_settle_s: float = 0.0,
    ) -> None:
        self.speculative_turns = speculative_turns
        self.mark_stale_during_process = mark_stale_during_process
        self.final_revision_settle_s = final_revision_settle_s
        self.processed: list[tuple[str | None, int | None]] = []

    def process(self, vad_audio: VADAudio) -> Iterator[Transcription]:
        self.processed.append((vad_audio.turn_id, vad_audio.turn_revision))
        if self.mark_stale_during_process and vad_audio.turn_id is not None and vad_audio.turn_revision is not None:
            assert self.speculative_turns is not None
            reopen(self.speculative_turns, vad_audio.turn_id, vad_audio.turn_revision)
        yield Transcription(
            text="hello",
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
        )


def _vad_audio(
    turn_id: str = "turn_1",
    revision: int = 0,
    mode: Literal["progressive", "final"] | None = None,
    processing_delay_s: float = 0.0,
) -> VADAudio:
    return VADAudio(
        audio=np.zeros(512, dtype=np.float32),
        mode=mode,
        turn_id=turn_id,
        turn_revision=revision,
        processing_delay_s=processing_delay_s,
    )


def _handler(
    tracker: SpeculativeTurnTracker,
    queue_in: Queue,
    queue_out: Queue,
    *,
    mark_stale_during_process: bool = False,
    final_revision_settle_s: float = 0.0,
) -> RecordingSTTHandler:
    return RecordingSTTHandler(
        Event(),
        queue_in=queue_in,
        queue_out=queue_out,
        setup_kwargs={
            "speculative_turns": tracker,
            "mark_stale_during_process": mark_stale_during_process,
            "final_revision_settle_s": final_revision_settle_s,
        },
    )


def test_final_input_gate_records_vad_settle_on_the_pending_turn():
    tracker = SpeculativeTurnTracker()
    tracker.start_turn()
    queue_in = Queue()
    queue_out = Queue()
    handler = _handler(tracker, queue_in, queue_out, final_revision_settle_s=0.05)
    store = TurnLatencyStore()
    handler.turn_latency_store = store

    assert handler.should_process_input(_vad_audio(mode="progressive"))
    assert store.pending_for_turn("turn_1", 0) is None

    # A gate that returned immediately stays out of the line entirely.
    quiet_turns = SpeculativeTurnTracker()
    quiet_turn_id, quiet_revision = quiet_turns.start_turn()
    quiet = _handler(quiet_turns, Queue(), Queue())
    quiet_store = quiet.turn_latency_store = TurnLatencyStore()
    assert quiet.should_process_input(_vad_audio(turn_id=quiet_turn_id, revision=quiet_revision, mode="final"))
    assert quiet_store.pending_for_turn(quiet_turn_id, quiet_revision) is None

    assert handler.should_process_input(_vad_audio(mode="final"))
    pending = store.pending_for_turn("turn_1", 0)
    assert pending is not None
    assert pending.vad_settle_s is not None
    assert pending.vad_settle_s >= 0.05


def test_superseded_revision_drops_its_pending_latency_slot():
    """A progressive update opens a pending slot; if the revision is then
    superseded no transcription will ever consume it."""
    tracker = SpeculativeTurnTracker()
    queue_in = Queue()
    queue_out = Queue()
    handler = _handler(tracker, queue_in, queue_out)
    store = TurnLatencyStore()
    handler.turn_latency_store = store
    store.get_or_create_for_turn("turn_1", 0).record_mlx_lock_wait(0.01, "ParakeetSTT-Progressive")

    tracker.start_turn()
    reopen(tracker)
    assert not handler.should_process_input(_vad_audio(revision=0, mode="final"))

    assert store.pending_for_turn("turn_1", 0) is None


def test_stt_handler_drops_stale_queued_audio_without_processing():
    tracker = SpeculativeTurnTracker()
    tracker.start_turn()
    reopen(tracker)
    queue_in = Queue()
    queue_out = Queue()
    handler = _handler(tracker, queue_in, queue_out)

    queue_in.put(_vad_audio(revision=0))
    queue_in.put(PIPELINE_END)

    handler.run()

    assert handler.processed == []
    assert queue_out.get_nowait() == PIPELINE_END


def test_stt_handler_bulk_drops_stale_queued_audio():
    tracker = SpeculativeTurnTracker()
    tracker.start_turn()
    reopen(tracker)
    queue_in = Queue()
    queue_out = Queue()
    handler = _handler(tracker, queue_in, queue_out)

    for _ in range(5):
        queue_in.put(_vad_audio(revision=0))
    queue_in.put(_vad_audio(revision=1))
    queue_in.put(PIPELINE_END)

    handler.run()

    assert handler.processed == [("turn_1", 1)]
    output = queue_out.get_nowait()
    assert isinstance(output, Transcription)
    assert output.turn_revision == 1
    assert queue_out.get_nowait() == PIPELINE_END


def test_stt_handler_waits_for_pending_reopen_before_processing():
    tracker = SpeculativeTurnTracker()
    tracker.start_turn()
    candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)
    queue_in = Queue()
    queue_out = Queue()
    handler = _handler(tracker, queue_in, queue_out)

    queue_in.put(_vad_audio(revision=0))
    queue_in.put(PIPELINE_END)
    thread = Thread(target=handler.run)
    thread.start()

    sleep(0.05)
    assert handler.processed == []
    assert queue_out.empty()

    assert tracker.confirm_reopen_candidate("turn_1", 0, candidate_revision)
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert handler.processed == []
    assert queue_out.get_nowait() == PIPELINE_END


def test_stt_handler_waits_for_final_revision_stability_window():
    tracker = SpeculativeTurnTracker()
    tracker.start_turn()
    queue_in = Queue()
    queue_out = Queue()
    handler = _handler(tracker, queue_in, queue_out, final_revision_settle_s=0.2)

    queue_in.put(_vad_audio(revision=0, mode="final"))
    queue_in.put(PIPELINE_END)
    thread = Thread(target=handler.run)
    thread.start()

    sleep(0.05)
    candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)
    assert tracker.confirm_reopen_candidate("turn_1", 0, candidate_revision)
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert handler.processed == []
    assert queue_out.get_nowait() == PIPELINE_END


def test_stt_handler_uses_per_endpoint_processing_delay():
    tracker = SpeculativeTurnTracker()
    tracker.start_turn()
    queue_in = Queue()
    queue_out = Queue()
    handler = _handler(tracker, queue_in, queue_out)

    queue_in.put(_vad_audio(revision=0, mode="final", processing_delay_s=0.2))
    queue_in.put(PIPELINE_END)
    thread = Thread(target=handler.run)
    thread.start()

    sleep(0.05)
    candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)
    assert tracker.confirm_reopen_candidate("turn_1", 0, candidate_revision)
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert handler.processed == []
    assert queue_out.get_nowait() == PIPELINE_END


def test_stale_final_input_discards_pending_vad_measurement():
    revisions = SpeculativeTurnTracker()
    revisions.start_turn()
    reopen(revisions)
    handler = _handler(revisions, Queue(), Queue())
    store = TurnLatencyStore()
    handler.turn_latency_store = store
    store.get_or_create_for_turn("turn_1", 0).vad_decision_s = 0.3

    assert not handler.should_process_input(_vad_audio(revision=0, mode="final"))
    assert store._pending_turn == {}


def test_stt_handler_drops_output_that_became_stale_during_processing():
    tracker = SpeculativeTurnTracker()
    tracker.start_turn()
    queue_in = Queue()
    queue_out = Queue()
    handler = _handler(tracker, queue_in, queue_out, mark_stale_during_process=True)

    queue_in.put(_vad_audio(revision=0))
    queue_in.put(PIPELINE_END)

    handler.run()

    assert handler.processed == [("turn_1", 0)]
    assert queue_out.get_nowait() == PIPELINE_END
    try:
        queue_out.get_nowait()
    except Empty:
        pass
    else:
        raise AssertionError("stale transcription output was emitted")


def test_stt_handler_drops_progressive_input_after_final_emit():
    tracker = SpeculativeTurnTracker()
    tracker.start_turn()
    queue_in = Queue()
    queue_out = Queue()
    handler = _handler(tracker, queue_in, queue_out)

    handler.before_emit_output(Transcription(text="done", turn_id="turn_1", turn_revision=0))

    assert not handler.should_process_input(_vad_audio(revision=0, mode="progressive"))


def test_stt_handler_drops_partial_output_after_final_emit():
    tracker = SpeculativeTurnTracker()
    tracker.start_turn()
    queue_in = Queue()
    queue_out = Queue()
    handler = _handler(tracker, queue_in, queue_out)

    handler.before_emit_output(Transcription(text="done", turn_id="turn_1", turn_revision=0))

    assert not handler.should_emit_output(PartialTranscription(text="partial", turn_id="turn_1", turn_revision=0))


def test_stt_handler_bulk_drops_queued_progressives_after_final_emit():
    tracker = SpeculativeTurnTracker()
    tracker.start_turn()
    queue_in = Queue()
    queue_out = Queue()
    handler = _handler(tracker, queue_in, queue_out)

    handler.before_emit_output(Transcription(text="done", turn_id="turn_1", turn_revision=0))
    for _ in range(3):
        queue_in.put(_vad_audio(revision=0, mode="progressive"))
    queue_in.put(_vad_audio(turn_id="turn_2", revision=0, mode="progressive"))
    tracker.start_turn()

    assert not handler.should_process_input(_vad_audio(revision=0, mode="progressive"))
    remaining = queue_in.get_nowait()

    assert isinstance(remaining, VADAudio)
    assert remaining.turn_id == "turn_2"
    assert queue_in.empty()


def test_stt_handler_drops_progressive_when_final_for_same_revision_is_queued():
    tracker = SpeculativeTurnTracker()
    tracker.start_turn()
    queue_in = Queue()
    queue_out = Queue()
    handler = _handler(tracker, queue_in, queue_out)

    queue_in.put(_vad_audio(revision=0, mode="final"))

    assert not handler.should_process_input(_vad_audio(revision=0, mode="progressive"))


def test_stt_handler_keeps_progressive_when_final_for_different_turn_is_queued():
    tracker = SpeculativeTurnTracker()
    tracker.start_turn()
    queue_in = Queue()
    queue_out = Queue()
    handler = _handler(tracker, queue_in, queue_out)

    queue_in.put(_vad_audio(turn_id="turn_2", revision=0, mode="final"))

    assert handler.should_process_input(_vad_audio(revision=0, mode="progressive"))


def test_stt_handler_bulk_drops_progressives_queued_before_matching_final():
    tracker = SpeculativeTurnTracker()
    tracker.start_turn()
    queue_in = Queue()
    queue_out = Queue()
    handler = _handler(tracker, queue_in, queue_out)

    queue_in.put(_vad_audio(revision=0, mode="progressive"))
    queue_in.put(_vad_audio(revision=0, mode="final"))
    queue_in.put(_vad_audio(turn_id="turn_2", revision=0, mode="progressive"))
    tracker.start_turn()

    assert handler._drop_stale_queued_inputs() == 2
    remaining = queue_in.get_nowait()

    assert isinstance(remaining, VADAudio)
    assert remaining.turn_id == "turn_2"
    assert queue_in.empty()


def test_rejected_stale_partial_discards_only_its_pending_latency():
    revisions = SpeculativeTurnTracker()
    revisions.start_turn()
    reopen(revisions)
    handler = _handler(revisions, Queue(), Queue())
    store = handler.turn_latency_store = TurnLatencyStore()
    store.get_or_create_for_turn("turn_1", 0)
    current = store.get_or_create_for_turn("turn_1", 1)

    assert not handler.should_emit_output(PartialTranscription(text="old", turn_id="turn_1", turn_revision=0))
    assert store._pending_turn == {("turn_1", 1): current}


def test_bulk_progressive_cleanup_preserves_same_revision_queued_final_latency():
    revisions = SpeculativeTurnTracker()
    revisions.start_turn()
    reopen(revisions)
    queue = Queue()
    handler = _handler(revisions, queue, Queue())
    store = handler.turn_latency_store = TurnLatencyStore()
    store.get_or_create_for_turn("turn_1", 0)
    current = store.get_or_create_for_turn("turn_1", 1)
    current.record_mlx_lock_wait(0.02, "ParakeetSTT-Progressive")
    queue.put(_vad_audio(revision=0, mode="progressive"))
    queue.put(_vad_audio(revision=1, mode="progressive"))
    final = _vad_audio(revision=1, mode="final")
    queue.put(final)

    assert handler._drop_stale_queued_inputs() == 2
    assert queue.get_nowait() is final
    assert queue.empty()
    assert store._pending_turn == {("turn_1", 1): current}
    assert current.mlx_lock_wait_s == 0.02


def test_stt_does_not_restart_expired_tracker_processing_delay(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("speech_to_speech.pipeline.speculative_turns.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("speech_to_speech.STT.base_stt_handler.monotonic", lambda: clock[0])
    tracker = SpeculativeTurnTracker()
    tracker.start_turn()
    tracker.segment_finalized(1000, output_hold_ms=2000, processing_delay_ms=600)
    clock[0] = 10.7
    handler = _handler(tracker, Queue(), Queue())
    item = _vad_audio(mode="final", processing_delay_s=0.6)
    result = []
    thread = Thread(target=lambda: result.append(handler.should_process_input(item)))
    thread.start()
    thread.join(timeout=0.2)
    completed_without_new_wait = not thread.is_alive()
    tracker.start_turn()
    thread.join(timeout=1.0)
    assert completed_without_new_wait
    assert result == [True]


class _SignalingSTTHandler(BaseSTTHandler):
    def setup(
        self,
        speculative_turns: SpeculativeTurnTracker | None = None,
        started: Event | None = None,
        finished: Event | None = None,
        final_revision_settle_s: float = 0.0,
    ) -> None:
        self.speculative_turns = speculative_turns
        self.started = started or Event()
        self.finished = finished or Event()
        self.final_revision_settle_s = final_revision_settle_s
        self.processed: list[tuple[str | None, int | None]] = []

    def process(self, vad_audio: VADAudio) -> Iterator[Transcription]:
        self.processed.append((vad_audio.turn_id, vad_audio.turn_revision))
        self.started.set()
        yield Transcription(
            text="hello",
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
        )
        self.finished.set()


def test_stt_runs_in_parallel_with_pending_smart_turn_and_emits_on_complete():
    tracker = SpeculativeTurnTracker()
    tracker.start_turn()
    tracker.segment_finalized(1000, pending_smart_turn=True, smart_turn_hold_s=2.0)

    started = Event()
    finished = Event()
    queue_in = Queue()
    queue_out = Queue()
    handler = _SignalingSTTHandler(
        Event(),
        queue_in=queue_in,
        queue_out=queue_out,
        setup_kwargs={"speculative_turns": tracker, "started": started, "finished": finished},
    )

    queue_in.put(_vad_audio(revision=0, mode="final"))
    queue_in.put(PIPELINE_END)
    thread = Thread(target=handler.run)
    thread.start()

    # STT begins transcribing immediately while Smart Turn is still pending,
    # but holds emission of the final Transcription until Smart Turn resolves.
    assert started.wait(1.0)
    sleep(0.03)
    assert queue_out.empty()

    # Resolving Smart Turn as complete (0ms processing delay, 800ms reopen grace)
    # releases the final Transcription immediately without waiting for reopen grace.
    assert tracker.resolve_smart_turn("turn_1", 0, output_hold_ms=800, processing_delay_ms=0)
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert handler.processed == [("turn_1", 0)]
    emitted = queue_out.get_nowait()
    assert isinstance(emitted, Transcription)
    assert emitted.text == "hello"
    assert queue_out.get_nowait() == PIPELINE_END


def test_stt_enforces_incomplete_delay_after_parallel_transcription_and_drops_on_reopen():
    tracker = SpeculativeTurnTracker()
    tracker.start_turn()
    tracker.segment_finalized(1000, pending_smart_turn=True, smart_turn_hold_s=2.0)

    started = Event()
    queue_in = Queue()
    queue_out = Queue()
    handler = _SignalingSTTHandler(
        Event(),
        queue_in=queue_in,
        queue_out=queue_out,
        setup_kwargs={"speculative_turns": tracker, "started": started},
    )

    queue_in.put(_vad_audio(revision=0, mode="final"))
    queue_in.put(PIPELINE_END)
    thread = Thread(target=handler.run)
    thread.start()

    assert started.wait(1.0)
    assert tracker.resolve_smart_turn("turn_1", 0, output_hold_ms=2000, processing_delay_ms=300)

    # Transcription must remain held during the incomplete processing delay.
    sleep(0.04)
    assert queue_out.empty()

    # User resumes speaking before incomplete delay expires -> dropped cleanly.
    candidate = tracker.begin_reopen_candidate("turn_1", 0)
    assert tracker.confirm_reopen_candidate("turn_1", 0, candidate)
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert queue_out.get_nowait() == PIPELINE_END


def test_stt_emits_after_incomplete_delay_expires_and_records_vad_settle():
    tracker = SpeculativeTurnTracker()
    tracker.start_turn()
    tracker.segment_finalized(1000, pending_smart_turn=True, smart_turn_hold_s=2.0)

    started = Event()
    queue_in = Queue()
    queue_out = Queue()
    handler = _SignalingSTTHandler(
        Event(),
        queue_in=queue_in,
        queue_out=queue_out,
        setup_kwargs={"speculative_turns": tracker, "started": started},
    )
    store = TurnLatencyStore()
    handler.turn_latency_store = store

    queue_in.put(_vad_audio(revision=0, mode="final"))
    queue_in.put(PIPELINE_END)
    thread = Thread(target=handler.run)
    thread.start()

    assert started.wait(1.0)
    assert tracker.resolve_smart_turn("turn_1", 0, output_hold_ms=500, processing_delay_ms=70)
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    emitted = queue_out.get_nowait()
    assert isinstance(emitted, Transcription)
    assert queue_out.get_nowait() == PIPELINE_END
    pending = store.pending_for_turn("turn_1", 0)
    assert pending is not None
    assert pending.vad_settle_s is not None
    assert pending.vad_settle_s >= 0.05


def test_stt_starts_immediately_when_incomplete_smart_turn_finishes_before_dequeue():
    tracker = SpeculativeTurnTracker()
    tracker.start_turn()
    tracker.segment_finalized(1000, pending_smart_turn=True, smart_turn_hold_s=2.0)
    # Smart Turn wins the race with an incomplete 200ms processing delay BEFORE STT
    # dequeues the final audio.
    assert tracker.resolve_smart_turn("turn_1", 0, output_hold_ms=2000, processing_delay_ms=200)

    started = Event()
    queue_in = Queue()
    queue_out = Queue()
    handler = _SignalingSTTHandler(
        Event(),
        queue_in=queue_in,
        queue_out=queue_out,
        setup_kwargs={"speculative_turns": tracker, "started": started, "final_revision_settle_s": 0.0},
    )

    queue_in.put(_vad_audio(revision=0, mode="final"))
    queue_in.put(PIPELINE_END)
    thread = Thread(target=handler.run)
    thread.start()

    # Transcription must start immediately (not wait 200ms for input admission),
    # while final transcript emission stays held for the incomplete delay.
    assert started.wait(0.08)
    sleep(0.04)
    assert queue_out.empty()

    thread.join(timeout=1.0)
    assert not thread.is_alive()
    emitted = queue_out.get_nowait()
    assert isinstance(emitted, Transcription)
    assert queue_out.get_nowait() == PIPELINE_END


def test_stt_preserves_input_revision_settle_with_resolved_async_smart_turn():
    tracker = SpeculativeTurnTracker()
    tracker.start_turn()
    tracker.segment_finalized(1000, pending_smart_turn=True, smart_turn_hold_s=2.0)
    assert tracker.resolve_smart_turn("turn_1", 0, output_hold_ms=2000, processing_delay_ms=300)

    started = Event()
    queue_in = Queue()
    queue_out = Queue()
    handler = _SignalingSTTHandler(
        Event(),
        queue_in=queue_in,
        queue_out=queue_out,
        setup_kwargs={"speculative_turns": tracker, "started": started, "final_revision_settle_s": 0.08},
    )

    queue_in.put(_vad_audio(revision=0, mode="final"))
    queue_in.put(PIPELINE_END)
    thread = Thread(target=handler.run)
    thread.start()

    # Input revision settling still holds transcription start for 80ms,
    # then transcription begins before the 300ms incomplete delay finishes.
    assert not started.wait(0.03)
    assert started.wait(0.15)
    assert queue_out.empty()

    # Reopening during the remaining incomplete delay still drops the final transcript.
    reopen(tracker, "turn_1", 0)
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert queue_out.get_nowait() == PIPELINE_END
