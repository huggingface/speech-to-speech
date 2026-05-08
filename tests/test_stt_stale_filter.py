from __future__ import annotations

from queue import Empty, Queue
from threading import Event, Thread
from time import sleep
from typing import Iterator

import numpy as np

from speech_to_speech.pipeline.messages import PIPELINE_END, Transcription, VADAudio
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler


class RecordingSTTHandler(BaseSTTHandler):
    def setup(
        self,
        speculative_turns: SpeculativeTurnTracker | None = None,
        mark_stale_during_process: bool = False,
    ) -> None:
        self.speculative_turns = speculative_turns
        self.mark_stale_during_process = mark_stale_during_process
        self.processed: list[tuple[str | None, int | None]] = []

    def process(self, vad_audio: VADAudio) -> Iterator[Transcription]:
        self.processed.append((vad_audio.turn_id, vad_audio.turn_revision))
        if self.mark_stale_during_process and vad_audio.turn_id is not None and vad_audio.turn_revision is not None:
            self.speculative_turns.observe(vad_audio.turn_id, vad_audio.turn_revision + 1)
        yield Transcription(
            text="hello",
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
        )


def _vad_audio(turn_id: str = "turn_1", revision: int = 0) -> VADAudio:
    return VADAudio(audio=np.zeros(512, dtype=np.float32), turn_id=turn_id, turn_revision=revision)


def _handler(
    tracker: SpeculativeTurnTracker,
    queue_in: Queue,
    queue_out: Queue,
    *,
    mark_stale_during_process: bool = False,
) -> RecordingSTTHandler:
    return RecordingSTTHandler(
        Event(),
        queue_in=queue_in,
        queue_out=queue_out,
        setup_kwargs={
            "speculative_turns": tracker,
            "mark_stale_during_process": mark_stale_during_process,
        },
    )


def test_stt_handler_drops_stale_queued_audio_without_processing():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 1)
    queue_in = Queue()
    queue_out = Queue()
    handler = _handler(tracker, queue_in, queue_out)

    queue_in.put(_vad_audio(revision=0))
    queue_in.put(PIPELINE_END)

    handler.run()

    assert handler.processed == []
    assert queue_out.get_nowait() == PIPELINE_END


def test_stt_handler_waits_for_pending_reopen_before_processing():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
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


def test_stt_handler_drops_output_that_became_stale_during_processing():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
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
