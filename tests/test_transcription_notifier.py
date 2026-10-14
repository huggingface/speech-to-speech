import logging
from queue import Queue
from threading import Event

from speech_to_speech.pipeline.events import (
    PartialTranscriptionEvent,
    TranscriptionCompletedEvent,
    TranscriptionFailedEvent,
)
from speech_to_speech.pipeline.messages import (
    PartialTranscription,
    Transcription,
    TranscriptionFailure,
)
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


def test_non_empty_final_transcription_logs_metadata_without_content(caplog):
    """Logs the completion, language and length -- never the transcript itself.

    Operational logs outlive the conversation, so content is deliberately omitted
    (see tests/test_transcript_log_hygiene.py).
    """
    notifier = _notifier()
    transcript = "hello " * 30

    with caplog.at_level(logging.INFO, logger="speech_to_speech.STT.transcription_notifier"):
        assert list(notifier.process(Transcription(text=transcript, language_code="en"))) == []

    assert "Transcription completed" in caplog.text
    assert "language=en" in caplog.text
    assert f"chars={len(transcript)}" in caplog.text
    assert transcript not in caplog.text
    assert "hello hello" not in caplog.text


def test_empty_final_transcription_reenables_listening_without_runtime_config():
    should_listen = Event()
    notifier = _notifier(should_listen=should_listen)

    assert list(notifier.process(Transcription(text="", language_code="en"))) == []

    assert should_listen.is_set()


def test_transcription_failure_defers_listening_change_to_realtime_owner():
    text_output_queue = Queue()
    should_listen = Event()
    notifier = _notifier(text_output_queue=text_output_queue, should_listen=should_listen)

    result = list(
        notifier.process(
            TranscriptionFailure(
                message="transcription request timed out",
                turn_id="turn-1",
                turn_revision=2,
            )
        )
    )

    assert result == []
    assert not should_listen.is_set()
    event = text_output_queue.get_nowait()
    assert isinstance(event, TranscriptionFailedEvent)
    assert event.message == "transcription request timed out"
    assert event.turn_id == "turn-1"
    assert event.turn_revision == 2


def test_diarization_status_logs_without_transcript(caplog):
    from speech_to_speech.pipeline.speaker_metadata import SpeakerAttribution, SpeakerInterval

    notifier = _notifier()
    with caplog.at_level(logging.INFO):
        for complete, available, status in [
            (True, True, "complete"),
            (False, True, "partial"),
            (False, False, "unavailable"),
        ]:
            caplog.clear()
            list(
                notifier.process(
                    Transcription(
                        text="private conversation",
                        turn_id="turn_1",
                        turn_revision=2,
                        speaker_attribution=SpeakerAttribution(
                            intervals=[SpeakerInterval(speaker=1, start=0, end=1)],
                            complete=complete,
                            available=available,
                        ),
                    )
                )
            )
            assert f"turn=turn_1 rev=2 speakers=speaker_1 status={status}" in caplog.text
            assert "durations=[speaker_1=1.000s]" in caplog.text
            assert "private conversation" not in caplog.text


def test_no_diarization_status_when_disabled(caplog):
    with caplog.at_level(logging.INFO):
        list(_notifier().process(Transcription(text="hello")))
    assert "Diarization" not in caplog.text


def test_speaker_durations_include_brief_and_simultaneous_activity():
    from speech_to_speech.pipeline.speaker_metadata import SpeakerAttribution, SpeakerInterval

    attribution = SpeakerAttribution(
        intervals=[
            SpeakerInterval(speaker=2, start=0.5, end=0.51),
            SpeakerInterval(speaker=1, start=0, end=1),
            SpeakerInterval(speaker=1, start=1.5, end=2),
        ]
    )
    assert attribution.durations_for_log() == "speaker_1=1.500s,speaker_2=0.010s"
    assert SpeakerAttribution().durations_for_log() == "none"
