"""Protocol tests for speculative turn revisions on the Realtime wire.

The speculative pipeline can pause, resume and revise one logical user turn
several times. A generic Realtime client cannot follow any of that: it keys
history by ``item_id``, treats
``conversation.item.input_audio_transcription.completed`` as final, and has no
event for retracting a stop or replacing a transcript.

Every test here therefore drives a scenario through the service and checks
only what such a client would see, then compares that history against the chat
the LLM is given.
"""

from queue import Queue
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.pipeline.events import (
    AssistantOutputEvent,
    PartialTranscriptionEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TranscriptionCompletedEvent,
    TranscriptionFailedEvent,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

from .realtime_contract import (
    RealtimeClientHistory,
    assert_input_lifecycle_contract,
    assert_openai_schema,
)

_INPUT_EVENTS = (
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
    "input_audio_buffer.committed",
    "conversation.item.created",
    "conversation.item.input_audio_transcription.delta",
    "conversation.item.input_audio_transcription.completed",
    "conversation.item.input_audio_transcription.failed",
)


class _Session:
    """Drives one connection the way the VAD, STT and TTS handlers do."""

    def __init__(self, runtime_config, should_listen) -> None:
        self.tracker = SpeculativeTurnTracker()
        self.text_prompt_queue: Queue = Queue()
        self.service = RealtimeService(
            text_prompt_queue=self.text_prompt_queue,
            should_listen=should_listen,
            speculative_turns=self.tracker,
        )
        self.conn_id = self.service.register()
        self.service._state(self.conn_id).runtime_config = runtime_config
        self.runtime_config = runtime_config
        self.events: list[Any] = []

    def close(self) -> None:
        self.service.unregister(self.conn_id)

    def dispatch(self, event) -> list[Any]:
        emitted = self.service.dispatch_pipeline_event(self.conn_id, event)
        self.events.extend(emitted)
        return emitted

    # ── VAD ──────────────────────────────────────

    def start_turn(self, turn_id: str, *, audio_start_ms: int = 0) -> list[Any]:
        self.tracker.observe(turn_id, 0)
        return self.dispatch(SpeechStartedEvent(turn_id=turn_id, turn_revision=0, audio_start_ms=audio_start_ms))

    def resume_turn(self, turn_id: str, revision: int, *, audio_start_ms: int = 0) -> list[Any]:
        """Reopen a turn exactly as ``VADHandler._reopen_current_turn`` does."""
        candidate = self.tracker.begin_reopen_candidate(turn_id, revision)
        assert candidate == revision + 1
        assert self.tracker.confirm_reopen_candidate(turn_id, revision, candidate)
        return self.dispatch(
            SpeechStartedEvent(
                turn_id=turn_id,
                turn_revision=candidate,
                reopened=True,
                audio_start_ms=audio_start_ms,
            )
        )

    def stop_speech(self, turn_id: str, revision: int, *, duration_s: float, audio_end_ms: int) -> list[Any]:
        return self.dispatch(
            SpeechStoppedEvent(
                turn_id=turn_id,
                turn_revision=revision,
                duration_s=duration_s,
                audio_end_ms=audio_end_ms,
            )
        )

    # ── STT ──────────────────────────────────────

    def hypothesis(self, turn_id: str, revision: int, transcript: str) -> list[Any]:
        return self.dispatch(PartialTranscriptionEvent(delta=transcript, turn_id=turn_id, turn_revision=revision))

    def final(self, turn_id: str, revision: int, transcript: str) -> list[Any]:
        return self.dispatch(
            TranscriptionCompletedEvent(transcript=transcript, turn_id=turn_id, turn_revision=revision)
        )

    # ── Assistant ────────────────────────────────

    def answer(self, turn_id: str, revision: int, text: str = "Sure.") -> list[Any]:
        """Accept assistant output, which commits the turn it answers."""
        return self.dispatch(AssistantOutputEvent(text=text, turn_id=turn_id, turn_revision=revision))

    # ── Views ────────────────────────────────────

    @property
    def input_events(self) -> list[Any]:
        return [event for event in self.events if event.type in _INPUT_EVENTS]

    def client_history(self) -> RealtimeClientHistory:
        return RealtimeClientHistory().apply(self.events)

    def chat_user_turns(self) -> list[str]:
        return [
            item.content[0].text for item in self.runtime_config.chat.buffer if getattr(item, "role", None) == "user"
        ]


@pytest.fixture
def session(runtime_config, should_listen):
    active = _Session(runtime_config, should_listen)
    yield active
    active.close()


def _item_ids(events: list[Any]) -> set[str]:
    return {getattr(event, "item_id", None) or event.item.id for event in events}


def test_paused_utterance_is_one_committed_item(session):
    """One utterance with two Smart Turn pauses reaches the client once.

    The captured failure in the issue is a run of ``speech_stopped`` events on
    one item, each followed by more speech. Only the stop that ends the
    committed turn may reach the client.
    """
    session.start_turn("turn_1")
    session.hypothesis("turn_1", 0, "The quick brown fox")
    assert session.stop_speech("turn_1", 0, duration_s=1.0, audio_end_ms=1000) == []

    session.resume_turn("turn_1", 0, audio_start_ms=1200)
    session.hypothesis("turn_1", 1, "The quick brown fox jumps over")
    assert session.stop_speech("turn_1", 1, duration_s=2.0, audio_end_ms=2000) == []

    session.resume_turn("turn_1", 1, audio_start_ms=2200)
    session.hypothesis("turn_1", 2, "The quick brown fox jumps over the lazy dog")
    assert session.stop_speech("turn_1", 2, duration_s=3.0, audio_end_ms=3000) == []

    # The final transcript feeds the LLM, but the user item stays private
    # until the server commits to answering it.
    assert session.final("turn_1", 2, "The quick brown fox jumps over the lazy dog") == []
    committed = session.answer("turn_1", 2)

    assert_openai_schema(session.events)
    assert_input_lifecycle_contract(session.events)

    stops = [event for event in session.events if event.type == "input_audio_buffer.speech_stopped"]
    completions = [
        event for event in session.events if event.type == "conversation.item.input_audio_transcription.completed"
    ]
    assert len(stops) == 1
    assert stops[0].audio_end_ms == 3000
    assert len(completions) == 1
    assert len([event for event in session.input_events if event.type == "input_audio_buffer.speech_started"]) == 1
    assert len(_item_ids(session.input_events)) == 1
    # The user item precedes the assistant output that committed it.
    assert [event.type for event in committed[:4]] == [
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        "conversation.item.created",
        "conversation.item.input_audio_transcription.completed",
    ]

    history = session.client_history()
    assert history.user_turns == ["The quick brown fox jumps over the lazy dog"]
    assert history.user_turns == session.chat_user_turns()


def test_speech_after_a_final_transcript_does_not_split_the_turn(session):
    """A pause long enough for a final STT still ends as one user item.

    This is the ``item_A`` / ``item_B`` / ``item_C`` divergence from the issue:
    the server replaced the earlier message in its own chat, while the client
    kept every cumulative completion as a separate user turn.
    """
    session.start_turn("turn_1")
    session.hypothesis("turn_1", 0, "The quick brown fox")
    session.stop_speech("turn_1", 0, duration_s=1.0, audio_end_ms=1000)
    assert session.final("turn_1", 0, "The quick brown fox") == []

    # The pause was long enough to transcribe, and then the user carried on.
    session.resume_turn("turn_1", 0, audio_start_ms=2500)
    # Output generated from the superseded revision is refused.
    assert session.answer("turn_1", 0, "About the fox?") == []
    session.hypothesis("turn_1", 1, "The quick brown fox jumps over the lazy dog")
    session.stop_speech("turn_1", 1, duration_s=3.0, audio_end_ms=3000)
    assert session.final("turn_1", 1, "The quick brown fox jumps over the lazy dog") == []
    session.answer("turn_1", 1)

    assert_openai_schema(session.events)
    assert_input_lifecycle_contract(session.events)

    history = session.client_history()
    assert history.completed_items == [event.item_id for event in session.input_events[:1]]
    assert history.user_turns == ["The quick brown fox jumps over the lazy dog"]
    assert history.user_turns == session.chat_user_turns()
    assert len(_item_ids(session.input_events)) == 1
    # Nothing was published while the turn could still be revised, so the
    # superseded transcript never reached the client.
    assert "The quick brown fox" not in history.transcripts.values()


def test_speech_after_a_committed_item_starts_a_new_item(session):
    """Once an item is committed, later speech is a genuine new user item."""
    session.start_turn("turn_1")
    session.hypothesis("turn_1", 0, "What is the weather")
    session.stop_speech("turn_1", 0, duration_s=1.0, audio_end_ms=1000)
    session.final("turn_1", 0, "What is the weather")
    session.answer("turn_1", 0, "Sunny.")

    # A committed turn cannot reopen, so the VAD opens a new one.
    assert session.tracker.begin_reopen_candidate("turn_1", 0) is None
    session.start_turn("turn_2", audio_start_ms=4000)
    session.hypothesis("turn_2", 0, "And tomorrow")
    session.stop_speech("turn_2", 0, duration_s=1.0, audio_end_ms=5000)
    session.final("turn_2", 0, "And tomorrow")
    session.answer("turn_2", 0, "Rain.")

    assert_openai_schema(session.events)
    assert_input_lifecycle_contract(session.events)

    history = session.client_history()
    assert len(history.completed_items) == 2
    assert len(set(history.completed_items)) == 2
    assert history.user_turns == ["What is the weather", "And tomorrow"]
    assert history.user_turns == session.chat_user_turns()


def test_unanswered_turn_is_published_when_the_next_turn_starts(session):
    """A turn nothing answered still ends before the next one opens.

    The client's history and the chat both keep the unanswered turn, so the
    two stay aligned even when no assistant output ever commits it.
    """
    session.start_turn("turn_1")
    session.hypothesis("turn_1", 0, "Hold on")
    session.stop_speech("turn_1", 0, duration_s=1.0, audio_end_ms=1000)
    assert session.final("turn_1", 0, "Hold on") == []

    published = session.start_turn("turn_2", audio_start_ms=9000)

    assert [event.type for event in published[:4]] == [
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        "conversation.item.created",
        "conversation.item.input_audio_transcription.completed",
    ]
    assert published[4].type == "input_audio_buffer.speech_started"
    assert published[0].item_id == published[1].item_id != published[4].item_id

    session.hypothesis("turn_2", 0, "Never mind")
    session.stop_speech("turn_2", 0, duration_s=1.0, audio_end_ms=10000)
    session.final("turn_2", 0, "Never mind")
    session.answer("turn_2", 0)

    assert_openai_schema(session.events)
    assert_input_lifecycle_contract(session.events)

    history = session.client_history()
    assert history.user_turns == ["Hold on", "Never mind"]
    assert history.user_turns == session.chat_user_turns()


def test_held_input_state_is_released_once_published(session):
    """Publishing a user item ends its routing and transcript bookkeeping."""
    session.start_turn("turn_1")
    session.hypothesis("turn_1", 0, "Release me")
    session.stop_speech("turn_1", 0, duration_s=1.0, audio_end_ms=1000)
    session.final("turn_1", 0, "Release me")

    state = session.service._state(session.conn_id)
    assert state.pending_input_terminals
    assert state.input_items

    session.answer("turn_1", 0)

    assert state.pending_input_terminals == {}
    assert state.input_items == {}
    assert state.input_item_by_turn_revision == {}
    assert state.current_input_item_id is None


def test_failed_turn_closes_without_another_utterance(session, should_listen):
    started = session.start_turn("turn_1")
    session.stop_speech("turn_1", 0, duration_s=1.0, audio_end_ms=1000)
    should_listen.clear()

    events = session.dispatch(TranscriptionFailedEvent(message="STT failed", turn_id="turn_1", turn_revision=0))

    assert [event.type for event in events] == [
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        "conversation.item.created",
        "conversation.item.input_audio_transcription.failed",
    ]
    assert _item_ids(events) == {started[0].item_id}
    assert should_listen.is_set()
    assert session.text_prompt_queue.empty()
    assert session.tracker.begin_reopen_candidate("turn_1", 0) is None
    state = session.service._state(session.conn_id)
    assert state.pending_input_terminals == state.input_items == state.input_item_by_turn_revision == {}
    assert state.current_input_item_id is None

    next_started = session.start_turn("turn_2", audio_start_ms=1100)
    assert next_started[0].item_id != started[0].item_id
    assert_openai_schema(session.events)
    assert_input_lifecycle_contract(session.events)


def test_failure_waits_only_for_reopen_grace(session, monkeypatch):
    now = 100.0
    monkeypatch.setattr("speech_to_speech.pipeline.speculative_turns.time", SimpleNamespace(monotonic=lambda: now))
    session.start_turn("turn_1")
    session.stop_speech("turn_1", 0, duration_s=1.0, audio_end_ms=1000)
    session.tracker.start_reopen_grace("turn_1", 0, grace_s=0.8)

    assert session.dispatch(TranscriptionFailedEvent(message="STT failed", turn_id="turn_1", turn_revision=0)) == []
    assert not session.tracker.is_committed("turn_1", 0)
    now += 0.9
    events = session.service.audio.resolve_input_terminals(session.conn_id)
    assert [event.type for event in events] == [
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        "conversation.item.created",
        "conversation.item.input_audio_transcription.failed",
    ]
    assert session.service.audio.resolve_input_terminals(session.conn_id) == []


def test_empty_transcript_closes_after_reopen_grace(session, monkeypatch):
    now = 100.0
    monkeypatch.setattr("speech_to_speech.pipeline.speculative_turns.time", SimpleNamespace(monotonic=lambda: now))
    started = session.start_turn("turn_1")
    session.stop_speech("turn_1", 0, duration_s=0.3, audio_end_ms=300)
    session.tracker.start_reopen_grace("turn_1", 0, grace_s=0.8)

    # No response is queued for an empty transcript, so no output will commit it.
    assert session.final("turn_1", 0, "") == []
    assert session.text_prompt_queue.empty()
    now += 0.9
    events = session.service.audio.resolve_input_terminals(session.conn_id)

    assert [event.type for event in events] == [
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        "conversation.item.created",
        "conversation.item.input_audio_transcription.completed",
    ]
    assert _item_ids(events) == {started[0].item_id}
    state = session.service._state(session.conn_id)
    assert state.pending_input_terminals == state.input_items == {}


def test_turn_cancelled_before_output_is_published(session):
    session.start_turn("turn_1")
    session.stop_speech("turn_1", 0, duration_s=1.0, audio_end_ms=1000)
    assert session.final("turn_1", 0, "Hello") == []

    session.service.handle_response_cancel(session.conn_id)
    session.events.extend(session.service.audio.resolve_input_terminals(session.conn_id))

    assert session.client_history().user_turns == session.chat_user_turns() == ["Hello"]
    assert session.tracker.begin_reopen_candidate("turn_1", 0) is None
    assert_openai_schema(session.events)
    assert_input_lifecycle_contract(session.events)


@pytest.mark.parametrize("resume", [False, True])
def test_failure_respects_pending_reopen(session, resume):
    session.start_turn("turn_1")
    session.stop_speech("turn_1", 0, duration_s=1.0, audio_end_ms=1000)
    candidate = session.tracker.begin_reopen_candidate("turn_1", 0)
    assert candidate == 1

    assert session.dispatch(TranscriptionFailedEvent(message="STT failed", turn_id="turn_1", turn_revision=0)) == []
    if resume:
        session.resume_turn("turn_1", 0, audio_start_ms=1200)
        session.stop_speech("turn_1", 1, duration_s=2.0, audio_end_ms=2000)
        session.final("turn_1", 1, "Try again")
        session.answer("turn_1", 1)
        assert session.client_history().failed_items == []
        assert session.client_history().user_turns == session.chat_user_turns() == ["Try again"]
    else:
        session.tracker.cancel_reopen_candidate("turn_1", candidate)
        session.events.extend(session.service.audio.resolve_input_terminals(session.conn_id))
        assert len(session.client_history().failed_items) == 1
        assert session.tracker.begin_reopen_candidate("turn_1", 0) is None
    assert_openai_schema(session.events)
    assert_input_lifecycle_contract(session.events)


def test_failed_revision_removes_superseded_chat_text(session):
    session.start_turn("turn_1")
    session.stop_speech("turn_1", 0, duration_s=1.0, audio_end_ms=1000)
    session.final("turn_1", 0, "An unfinished request")
    session.resume_turn("turn_1", 0, audio_start_ms=1200)
    session.stop_speech("turn_1", 1, duration_s=2.0, audio_end_ms=2000)

    session.dispatch(TranscriptionFailedEvent(message="STT failed", turn_id="turn_1", turn_revision=1))

    assert session.answer("turn_1", 0, "An outdated answer") == []
    assert session.client_history().user_turns == session.chat_user_turns() == []
    assert len(session.client_history().failed_items) == 1
    assert session.service._state(session.conn_id).response_usage.audio_duration_s == 0
    assert_input_lifecycle_contract(session.events)


def test_late_transcript_does_not_recreate_a_committed_item(session):
    first = session.start_turn("turn_1")[0].item_id
    session.stop_speech("turn_1", 0, duration_s=1.0, audio_end_ms=1000)
    # The next turn settles the old audio before its transcription arrives.
    published = session.start_turn("turn_2", audio_start_ms=9000)
    assert [event.type for event in published[:3]] == [
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        "conversation.item.created",
    ]
    assert published[1].previous_item_id is None
    assert published[2].previous_item_id is None
    session.final("turn_1", 0, "First")
    session.answer("turn_1", 0)
    session.stop_speech("turn_2", 0, duration_s=1.5, audio_end_ms=10500)
    session.final("turn_2", 0, "Second")
    previous = session.service._state(session.conn_id).last_item_id
    committed = session.answer("turn_2", 0)
    assert committed[1].previous_item_id == previous
    assert committed[2].previous_item_id == previous
    assert (
        len([event for event in session.events if event.type == "conversation.item.created" and event.item.id == first])
        == 1
    )
    assert session.client_history().user_turns == session.chat_user_turns() == ["First", "Second"]
    assert_input_lifecycle_contract(session.events)
    assert_openai_schema(session.events)


def test_direct_audio_commits_a_user_item_without_transcription(session):
    from speech_to_speech.pipeline.events import AudioInputCompletedEvent

    started = session.start_turn("turn_1")[0]
    session.stop_speech("turn_1", 0, duration_s=1.0, audio_end_ms=1000)
    session.dispatch(
        AudioInputCompletedEvent(
            audio=np.zeros(16000, dtype=np.float32),
            audio_sample_rate=16000,
            audio_duration_s=1.0,
            turn_id="turn_1",
            turn_revision=0,
        )
    )
    committed = session.answer("turn_1", 0)
    assert [event.type for event in committed[:3]] == [
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        "conversation.item.created",
    ]
    assert committed[2].item.id == started.item_id
    assert not session.client_history().completed_items
    assert session.service._state(session.conn_id).input_items == {}
    assert_input_lifecycle_contract(session.events)
    assert_openai_schema(session.events)
