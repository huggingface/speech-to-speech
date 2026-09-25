from contextlib import contextmanager
from threading import Event, Thread

import numpy as np
import pytest

from speech_to_speech.diarization.streaming import SpeakerSegment
from speech_to_speech.diarization.worker import DiarizationWorker
from speech_to_speech.pipeline.messages import Transcription, VADAudio
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler


class RecordingDiarizer:
    sample_rate = 20

    def __init__(self):
        self.calls = []
        self.reset_count = 0
        self.flush_count = 0
        self.entered = Event()
        self.release = Event()
        self.release.set()
        self.fail = False
        self.reset()

    def reset(self):
        self.reset_count += 1
        self.samples = 0

    def push(self, audio, *, sample_rate):
        self.entered.set()
        assert self.release.wait(2)
        if self.fail:
            raise RuntimeError("inference failed")
        self.calls.append(audio.copy())
        self.samples += len(audio)
        return []

    @property
    def processed_seconds(self):
        return self.samples / self.sample_rate

    @property
    def active_segments(self):
        return (SpeakerSegment(0, 0, self.processed_seconds),) if self.samples else ()

    def finish_utterance(self):
        self.flush_count += 1
        result = [SpeakerSegment(0, 0, self.samples / self.sample_rate)] if self.samples else []
        self.samples = 0
        return result


@contextmanager
def running(worker):
    thread = Thread(target=worker.run)
    thread.start()
    try:
        yield
    finally:
        worker.diarizer.release.set()
        worker.stop_event.set()
        thread.join(2)
        assert not thread.is_alive()


def wait_result(pending):
    for future, _, _ in pending.parts:
        future.result(timeout=2)
    return pending.resolve()


def test_worker_flushes_tail_and_maps_skipped_silence_without_replaying_preroll():
    model = RecordingDiarizer()
    worker = DiarizationWorker(model, Event())
    with running(worker):
        worker.append_audio(np.arange(8, dtype=np.float32), 0)
        first = worker.finalize(0, 8)
        assert wait_result(first).complete
        # Repeated pre-roll is trimmed rather than scored twice.
        worker.append_audio(np.arange(6, 14, dtype=np.float32), 6)
        second = worker.finalize(6, 14)
        assert wait_result(second).complete
        np.testing.assert_array_equal(np.concatenate(model.calls), np.arange(14))
        # A long silence never goes through the model, nor shifts timestamps.
        worker.append_audio(np.arange(4, dtype=np.float32), 100)
        third = worker.finalize(100, 104)
        result = wait_result(third)
        assert result.complete
        assert [(i.start, i.end) for i in result.intervals] == [(0, pytest.approx(0.2))]
        assert sum(len(c) for c in model.calls) == 18
        assert model.flush_count == 3


def test_slow_model_does_not_block_producer_or_stt_and_resolves_later():
    model = RecordingDiarizer()
    model.release.clear()
    worker = DiarizationWorker(model, Event())
    with running(worker):
        worker.append_audio(np.ones(4, dtype=np.float32), 0)
        assert model.entered.wait(1)
        queued = Event()
        holder = []

        def produce():
            holder.append(worker.finalize(0, 4))
            queued.set()

        producer = Thread(target=produce)
        producer.start()
        assert queued.wait(0.5)  # Model is still deliberately blocked.
        producer.join()
        source = VADAudio(audio=np.ones(4), speaker_pending=holder[0])
        stt = object.__new__(BaseSTTHandler)
        result = stt.output_for_queue(Transcription(text="hello"), source)
        assert not result.speaker_attribution.complete
        assert result.text == "hello"
        assert not model.release.is_set()
        model.release.set()
        assert wait_result(holder[0]).complete
        # Already-emitted metadata is a snapshot, not mutated by late inference.
        assert not result.speaker_attribution.complete


def test_overflow_invalidates_session_without_blocking_or_relabelling():
    worker = DiarizationWorker(RecordingDiarizer(), Event(), max_queue_items=2)
    worker.append_audio(np.ones(8, dtype=np.float32), 0)
    assert worker.queue.qsize() == 2
    old = worker.finalize(0, 8)
    assert not old.resolve().available
    worker.reset_session()
    assert worker.queue.empty()
    with running(worker):
        worker.append_audio(np.ones(2, dtype=np.float32), 0)
        new = worker.finalize(0, 2)
        assert wait_result(new).available
        assert not old.resolve().available


def test_reset_during_inference_never_publishes_old_session_labels():
    model = RecordingDiarizer()
    model.release.clear()
    worker = DiarizationWorker(model, Event())
    with running(worker):
        worker.append_audio(np.ones(2, dtype=np.float32), 0)
        assert model.entered.wait(1)
        old = worker.finalize(0, 2)
        worker.reset_session()
        assert not old.resolve().available
        worker.append_audio(np.ones(4, dtype=np.float32), 0)
        new = worker.finalize(0, 4)
        model.release.set()
        result = wait_result(new)
        assert result.complete
        assert [(i.start, i.end) for i in result.intervals] == [(0, 0.2)]
        assert model.reset_count >= 3
        assert not old.resolve().available


def test_model_failure_invalidates_pending_and_can_recover_next_session():
    model = RecordingDiarizer()
    model.fail = True
    worker = DiarizationWorker(model, Event())
    with running(worker):
        worker.append_audio(np.ones(2, dtype=np.float32), 0)
        pending = worker.finalize(0, 2)
        assert pending.parts[0][1].invalid.wait(1)
        assert not pending.resolve().available
        model.fail = False
        worker.reset_session()
        worker.append_audio(np.ones(2, dtype=np.float32), 0)
        assert wait_result(worker.finalize(0, 2)).complete


def test_reopened_prefix_can_finish_after_later_revision_is_queued():
    worker = DiarizationWorker(RecordingDiarizer(), Event())
    worker.append_audio(np.ones(2, dtype=np.float32), 0)
    prefix = worker.finalize(0, 2)
    worker.append_audio(np.ones(4, dtype=np.float32), 20)
    current = worker.finalize(20, 24).with_prefix(prefix, offset=0.1)
    assert not current.resolve().complete
    with running(worker):
        result = wait_result(current)
        assert result.complete
        assert [(i.start, i.end) for i in result.intervals] == [(0, 0.1), (0.1, pytest.approx(0.3))]


def test_discarded_candidate_does_not_leak_intervals_into_next_utterance():
    worker = DiarizationWorker(RecordingDiarizer(), Event())
    with running(worker):
        worker.append_audio(np.ones(2, dtype=np.float32), 0)
        worker.discard_utterance()
        worker.append_audio(np.ones(2, dtype=np.float32), 40)
        result = wait_result(worker.finalize(40, 42))
        assert [(i.start, i.end) for i in result.intervals] == [(0, pytest.approx(0.1))]


def test_invalid_queue_bound_is_rejected():
    with pytest.raises(ValueError):
        DiarizationWorker(RecordingDiarizer(), Event(), max_queue_items=0)


def test_inflight_tail_uses_ready_prefix_without_waiting_or_mutating_sent_metadata():
    model = RecordingDiarizer()
    flushing, release_flush = Event(), Event()
    original_flush = model.finish_utterance

    def flush():
        flushing.set()
        assert release_flush.wait(2)
        return original_flush()

    model.finish_utterance = flush
    worker = DiarizationWorker(model, Event())
    with running(worker):
        try:
            worker.append_audio(np.ones(4, dtype=np.float32), 0)
            pending = worker.finalize(0, 4)
            assert flushing.wait(1)
            partial = pending.resolve()
            assert not partial.complete
            assert {i.speaker for i in partial.intervals} == {0}
            release_flush.set()
            assert wait_result(pending).complete
            assert not partial.complete
        finally:
            release_flush.set()


def test_worker_logs_actual_processing(caplog):
    import logging

    worker = DiarizationWorker(RecordingDiarizer(), Event())
    with caplog.at_level(logging.INFO), running(worker):
        worker.append_audio(np.ones(4), 0)
        wait_result(worker.finalize(0, 4))
    assert "Diarization worker running" in caplog.text
    assert "first audio chunk processed" in caplog.text
    assert "audio=0.20s speakers=speaker_0 complete=True" in caplog.text
    assert "durations=[speaker_0=0.200s]" in caplog.text
    assert "Diarization worker stopped" in caplog.text


def test_worker_timings_separate_processing_from_queue_wait(monkeypatch, caplog):
    import logging

    import speech_to_speech.diarization.worker as worker_module

    clock = [0.0]
    monkeypatch.setattr(worker_module, "perf_counter", lambda: clock[0])

    class TimedDiarizer(RecordingDiarizer):
        def push(self, audio, *, sample_rate):
            clock[0] += 0.02
            return super().push(audio, sample_rate=sample_rate)

        def finish_utterance(self):
            clock[0] += 0.03
            return super().finish_utterance()

    worker = DiarizationWorker(TimedDiarizer(), Event())
    worker.append_audio(np.ones(4), 0)  # Two chunks, 20 ms processing each.
    pending = worker.finalize(0, 4)
    clock[0] = 0.5  # Queue waiting must not count as processing time.
    with caplog.at_level(logging.INFO), running(worker):
        wait_result(pending)
    assert "processing=0.070s flush=0.030s finalize_queue=0.540s finalize_latency=0.570s" in caplog.text
