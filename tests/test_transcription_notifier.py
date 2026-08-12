import logging
from queue import Queue
from threading import Event

from speech_to_speech.pipeline.events import (
    PartialTranscriptionEvent,
    TranscriptionCompletedEvent,
    TranscriptionFailedEvent,
)
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, TranscriptionFailure
from speech_to_speech.STT.transcription_notifier import TranscriptionNotifier


def _notifier(
    text_output_queue: Queue | None = None,
    should_listen: Event | None = None,
) -> TranscriptionNotifier:
    notifier = object.__new__(TranscriptionNotifier)
    notifier.setup(text_output_queue=text_output_queue, should_listen=should_listen)
    return notifier


def test_empty_final_transcription_still_emits_completion_after_partial():
    text_output_queue = Queue()
    notifier = _notifier(text_output_queue=text_output_queue)

    assert list(notifier.process(PartialTranscription(text="Yeah."))) == []
    assert list(notifier.process(Transcription(text="", language_code="en", speech_stopped_at_s=123.0))) == []

    partial = text_output_queue.get_nowait()
    completed = text_output_queue.get_nowait()

    assert isinstance(partial, PartialTranscriptionEvent)
    assert partial.delta == "Yeah."
    assert isinstance(completed, TranscriptionCompletedEvent)
    assert completed.transcript == ""
    assert completed.language_code == "en"
    assert completed.speech_stopped_at_s == 123.0
    assert text_output_queue.empty()


def test_transcription_failure_is_forwarded_as_terminal_event():
    text_output_queue = Queue()
    should_listen = Event()
    notifier = _notifier(text_output_queue=text_output_queue, should_listen=should_listen)

    assert (
        list(
            notifier.process(
                TranscriptionFailure(
                    message="failed",
                    code="backend_error",
                    turn_id="turn_1",
                    turn_revision=2,
                )
            )
        )
        == []
    )

    failed = text_output_queue.get_nowait()
    assert isinstance(failed, TranscriptionFailedEvent)
    assert failed.message == "failed"
    assert failed.code == "backend_error"
    assert (failed.turn_id, failed.turn_revision) == ("turn_1", 2)
    assert should_listen.is_set()


def test_non_empty_final_transcription_logs_full_text_at_info(caplog):
    notifier = _notifier()
    transcript = "hello " * 30

    with caplog.at_level(logging.INFO, logger="speech_to_speech.STT.transcription_notifier"):
        assert list(notifier.process(Transcription(text=transcript, language_code="en"))) == []

    assert "Transcription completed (language=en): " + transcript in caplog.text


def test_empty_final_transcription_reenables_listening_without_runtime_config():
    should_listen = Event()
    notifier = _notifier(should_listen=should_listen)

    assert list(notifier.process(Transcription(text="", language_code="en"))) == []

    assert should_listen.is_set()
