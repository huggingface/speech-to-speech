"""Barge-in that waits for words: coughs and backchannels do not cancel a reply (#878).

The router tests drive one WebSocket session the way the VAD, STT, LLM and TTS
threads do, and check only what the client receives and what the model is
given next.
"""

import threading
import time
from queue import Empty

import numpy as np
import pytest
from starlette.testclient import TestClient

from speech_to_speech.api.openai_realtime.false_interruption import (
    FalseInterruptionConfig,
    PendingInterruption,
    Verdict,
    parse_false_interruption,
)
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.tool_followup import TOOL_FOLLOWUP_METADATA_KEY
from speech_to_speech.api.openai_realtime.transports import WebSocketTransport
from speech_to_speech.api.openai_realtime.websocket_router import create_app
from speech_to_speech.LLM.language_model import BaseLanguageModelHandler
from speech_to_speech.pipeline.events import (
    AssistantOutputEvent,
    AssistantResponseDoneEvent,
    PartialTranscriptionEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TranscriptionCompletedEvent,
    TranscriptionFailedEvent,
)
from speech_to_speech.pipeline.history import ResponseHistory
from speech_to_speech.pipeline.messages import (
    AUDIO_RESPONSE_DONE,
    AssistantTextPart,
    AssistantToolCallPart,
    AudioOutput,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker, TurnPhase
from tests.openai_realtime.conftest import make_pipeline_unit

RATE = 16_000
_INPUT_EVENTS = {
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
    "input_audio_buffer.committed",
    "conversation.item.input_audio_transcription.delta",
    "conversation.item.input_audio_transcription.completed",
    "conversation.item.input_audio_transcription.failed",
}

# ── Words that count ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("transcript", "words"),
    [
        ("", 0),
        ("Uh", 0),
        ("Uh.", 0),
        ("Mm-hmm.", 0),
        ("mm hmm", 0),
        ("Uh-huh, yeah right.", 0),
        ("Yeah.", 0),
        ("Okay.", 0),
        ("Yeah, I see.", 0),
        ("Ahem.", 0),
        ("A ham.", 0),
        ("[coughs]", 0),
        ("(laughing) Hmm.", 0),
        ("Thank you.", 0),
        ("No.", 1),
        ("Uh, what?", 1),
        ("Okay, stop.", 1),
        ("Yeah, but why?", 2),
        ("Wait, wait, can you say that again?", 7),
        ("什么意思", 4),
    ],
)
def test_fillers_and_backchannels_do_not_count(transcript, words):
    assert FalseInterruptionConfig().word_count(transcript) == words


def test_ignore_words_can_be_replaced():
    config = FalseInterruptionConfig(ignore_words=("vale",))
    assert config.word_count("Vale.") == 0
    assert config.word_count("Uh") == 1
    assert FalseInterruptionConfig(ignore_words=()).word_count("Yeah.") == 1


def test_session_field_values():
    assert parse_false_interruption(None) is None
    assert parse_false_interruption(False) is None
    assert parse_false_interruption(True) == FalseInterruptionConfig()
    assert parse_false_interruption({"playback": "pause", "min_words": 2}) == FalseInterruptionConfig(
        playback="pause", min_words=2
    )
    for bad in ({"min_words": 0}, {"playback": "mute"}, {"unknown": 1}, "yes", 3):
        with pytest.raises(ValueError, match="false_interruption"):
            parse_false_interruption(bad)


def test_one_hypothesis_with_a_word_is_not_enough():
    """An STT can hallucinate a word into a cough; the next hypothesis must agree."""
    pending = PendingInterruption.begin(
        SpeechStartedEvent(turn_id="turn_2", turn_revision=0), FalseInterruptionConfig()
    )
    assert pending.observe(PartialTranscriptionEvent(delta="Hey", turn_id="turn_2", turn_revision=0)) is Verdict.HOLD
    assert pending.observe(PartialTranscriptionEvent(delta="Uh", turn_id="turn_2", turn_revision=0)) is Verdict.HOLD
    assert pending.observe(PartialTranscriptionEvent(delta="Wait", turn_id="turn_2", turn_revision=0)) is Verdict.HOLD
    assert (
        pending.observe(PartialTranscriptionEvent(delta="Wait, wait", turn_id="turn_2", turn_revision=0))
        is Verdict.CONFIRM
    )


def test_events_of_a_replaced_revision_are_dropped():
    pending = PendingInterruption.begin(
        SpeechStartedEvent(turn_id="turn_2", turn_revision=0), FalseInterruptionConfig()
    )
    started = SpeechStartedEvent(turn_id="turn_2", turn_revision=1, reopened=True)
    assert pending.observe(started) is Verdict.HOLD
    late = TranscriptionCompletedEvent(transcript="", turn_id="turn_2", turn_revision=0)
    assert pending.observe(late) is Verdict.DROP
    assert late not in pending.events
    assert pending.revisions == {0, 1}


# ── Session configuration ────────────────────────────────────────────


def test_session_update_validates_and_echoes_the_field():
    service = RealtimeService()
    conn_id = service.register()
    update = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "audio": {"input": {"turn_detection": {"type": "server_vad", "false_interruption": {"min_words": 0}}}},
        },
    }
    error = service.handle_session_update(conn_id, service.parse_client_event(update))
    assert error is not None and "min_words" in error.error.message
    assert service._state(conn_id).runtime_config.false_interruption is None

    update["session"]["audio"]["input"]["turn_detection"]["false_interruption"] = {"playback": "duck"}
    assert service.handle_session_update(conn_id, service.parse_client_event(update)) is None
    assert service._state(conn_id).runtime_config.false_interruption == FalseInterruptionConfig(playback="duck")
    echoed = service.build_session_updated(conn_id).model_dump()["session"]["audio"]["input"]["turn_detection"]
    assert echoed["false_interruption"] == {"playback": "duck"}


def test_server_default_applies_until_the_session_sets_the_field():
    service = RealtimeService(default_false_interruption=FalseInterruptionConfig(playback="pause"))
    conn_id = service.register()
    cfg = service._state(conn_id).runtime_config
    assert cfg.false_interruption == FalseInterruptionConfig(playback="pause")
    cfg.session.audio.input.turn_detection = {"type": "server_vad", "interrupt_response": True}
    assert cfg.false_interruption == FalseInterruptionConfig(playback="pause")
    cfg.session.audio.input.turn_detection = {"type": "server_vad", "false_interruption": None}
    assert cfg.false_interruption is None
    assert RuntimeConfig().false_interruption is None


# ── Turn tracker ─────────────────────────────────────────────────────


def _answering_turn(tracker):
    turn = tracker.start_turn()
    tracker.segment_finalized(1000)
    assert tracker.gate(*turn, commit=True).action.value == "accept"
    return turn


def test_provisional_turn_keeps_the_previous_turn_relevant():
    tracker = SpeculativeTurnTracker()
    first = _answering_turn(tracker)
    tracker.close(*first)  # its reply finished; a tool follow-up may still answer it
    second = tracker.start_turn()
    assert not tracker.is_latest(*first)

    assert tracker.mark_provisional(second[0])
    assert tracker.is_latest(*first)
    assert tracker.current_is_provisional()

    assert tracker.discard_provisional(second[0])
    assert tracker.current_turn() == first
    assert tracker.phase is TurnPhase.CLOSED
    assert not tracker.current_is_provisional()
    assert not tracker.is_latest(*second)


def test_dismissed_turn_restores_a_reply_that_finished_meanwhile():
    tracker = SpeculativeTurnTracker()
    first = _answering_turn(tracker)
    second = tracker.start_turn()
    assert tracker.mark_provisional(second[0])
    assert tracker.phase is TurnPhase.LISTENING

    tracker.close(*first)  # the reply ends while the cough is being transcribed
    assert tracker.is_latest(*first)
    tracker.discard_provisional(second[0])

    assert tracker.current_turn() == first
    assert tracker.phase is TurnPhase.CLOSED
    # The next speech starts a new turn instead of reopening the cough.
    assert tracker.speech_started(5000) == ("turn_3", 0, False)


def test_confirmed_turn_supersedes_the_previous_one():
    tracker = SpeculativeTurnTracker()
    first = _answering_turn(tracker)
    tracker.close(*first)
    second = tracker.start_turn()
    tracker.mark_provisional(second[0])

    tracker.confirm_provisional(second[0])

    assert not tracker.is_latest(*first)
    assert tracker.current_turn() == second
    assert not tracker.discard_provisional(second[0])


def test_a_newer_turn_keeps_the_turn_before_the_provisional_one():
    tracker = SpeculativeTurnTracker()
    first = _answering_turn(tracker)
    second = tracker.start_turn()
    tracker.mark_provisional(second[0])
    third = tracker.start_turn()
    assert not tracker.discard_provisional(second[0])

    assert tracker.mark_provisional(third[0])
    assert tracker.discard_provisional(third[0])
    assert tracker.current_turn() == first


# ── Router ───────────────────────────────────────────────────────────


class _Call:
    """One WebSocket session, with the pipeline threads played by the test."""

    def __init__(self, ws, unit):
        self.ws = ws
        self.unit = unit
        self.service = unit.service
        self.tracker = unit.service.speculative_turns
        self.conn_id = next(iter(self.service._conns))
        self.received: list[dict] = []
        self.audio_ms = 0

    # Client side

    def configure(self, false_interruption=True, *, extensions=None):
        session = {
            "type": "realtime",
            "audio": {
                "input": {
                    "turn_detection": {
                        "type": "server_vad",
                        "interrupt_response": True,
                        "false_interruption": false_interruption,
                    }
                }
            },
        }
        if extensions is not None:
            session["extensions"] = extensions
        self.ws.send_json({"type": "session.update", "session": session})
        self.receive_until("session.updated")

    def receive_until(self, event_type, predicate=lambda event: True) -> list[dict]:
        events = []
        while True:
            event = self.ws.receive_json()
            events.append(event)
            self.received.append(event)
            if event["type"] == "error":
                raise AssertionError(event)
            if event["type"] == event_type and predicate(event):
                return events

    def probe(self, key) -> list[dict]:
        """Everything sent before one more audio chunk of the reply *key* arrives."""
        self.audio(key, 0.1)
        return self.receive_until("response.output_audio.delta")

    # VAD and STT

    def speech(self, *, reopen=False):
        if reopen:
            turn_id, revision, reopened = self.tracker.speech_started(self.audio_ms)
            assert reopened
        else:
            turn_id, revision = self.tracker.start_turn()
        self.unit.text_output_queue.put(
            SpeechStartedEvent(turn_id=turn_id, turn_revision=revision, audio_start_ms=self.audio_ms, reopened=reopen)
        )
        return turn_id, revision

    def stop(self, turn, duration_s=0.4):
        self.audio_ms += int(duration_s * 1000)
        self.tracker.segment_finalized(self.audio_ms)
        self.unit.text_output_queue.put(
            SpeechStoppedEvent(
                turn_id=turn[0], turn_revision=turn[1], duration_s=duration_s, audio_end_ms=self.audio_ms
            )
        )

    def partial(self, turn, text):
        self.unit.text_output_queue.put(PartialTranscriptionEvent(delta=text, turn_id=turn[0], turn_revision=turn[1]))

    def final(self, turn, text):
        self.unit.text_output_queue.put(
            TranscriptionCompletedEvent(transcript=text, turn_id=turn[0], turn_revision=turn[1])
        )

    def user_says(self, text) -> tuple[tuple[str, int], str]:
        turn = self.speech()
        self.stop(turn, 1.0)
        self.final(turn, text)
        request = self.service.text_prompt_queue.get(timeout=2)
        return turn, request.response_key

    # LLM and TTS

    def reply(self, turn, key, text, seconds=2.0, parts=None):
        generation = self.unit.cancel_scope.generation
        parts = parts or [AssistantTextPart(text=text)]
        chat = self.service._state(self.conn_id).runtime_config.chat
        history = ResponseHistory.capture(
            chat,
            BaseLanguageModelHandler._ordered_output_items(parts, wants_audio=True),
            after_item_id=chat.history_anchor_id(),
        )
        for part in parts:
            self.unit.output_queue.put(
                AssistantOutputEvent(
                    parts=[part],
                    turn_id=turn[0] if turn else None,
                    turn_revision=turn[1] if turn else None,
                    response_key=key,
                    cancel_generation=generation,
                    history=history,
                )
            )
        self.audio(key, seconds)

    def audio(self, key, seconds):
        generation = self.unit.cancel_scope.generation
        for _ in range(round(seconds * 10)):
            self.unit.output_queue.put(
                AudioOutput(audio=np.zeros(RATE // 10, dtype=np.int16), response_key=key, cancel_generation=generation)
            )

    def finish(self, key) -> dict:
        generation = self.unit.cancel_scope.generation
        self.unit.output_queue.put(AssistantResponseDoneEvent(response_key=key, cancel_generation=generation))
        self.unit.output_queue.put(
            AudioOutput(audio=AUDIO_RESPONSE_DONE, response_key=key, cancel_generation=generation)
        )
        return self.receive_until("response.done")[-1]

    # Views

    def user_messages(self):
        return [
            item.content[0].text
            for item in self.service._state(self.conn_id).runtime_config.chat.buffer
            if getattr(item, "role", None) == "user"
        ]


@pytest.fixture
def call():
    tracker = SpeculativeTurnTracker()
    unit = make_pipeline_unit(service=RealtimeService(speculative_turns=tracker))
    unit.service.text_prompt_queue = unit.text_prompt_queue
    unit.service.should_listen = unit.should_listen
    app = create_app(pool=[unit], stop_event=threading.Event())
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()  # session.created
            yield _Call(ws, unit)


def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


def _pending(call):
    return call.service._state(call.conn_id).pending_interruption


def _coach_is_reading(call):
    """The user answered; the reply that reads the next question is playing."""
    turn, key = call.user_says("The answer is softmax.")
    call.reply(turn, key, "Correct! Question two: what does the scaling factor in attention prevent?")
    call.receive_until("response.output_audio.delta")
    return turn, key


@pytest.mark.parametrize("noise", ["", "Uh.", "Mm-hmm.", "Yeah.", "[coughs]"])
def test_cough_or_backchannel_does_not_cancel_the_reply(call, noise):
    call.configure()
    turn, key = _coach_is_reading(call)
    generation = call.unit.cancel_scope.generation

    start = len(call.received)
    cough = call.speech()
    call.probe(key)
    call.stop(cough, 0.35)
    call.partial(cough, noise)
    call.final(cough, noise)
    call.probe(key)
    done = call.finish(key)

    assert [event["type"] for event in call.received[start:] if event["type"] in _INPUT_EVENTS] == []
    assert done["response"]["status"] == "completed"
    assert call.unit.cancel_scope.generation == generation
    assert call.user_messages() == ["The answer is softmax."]
    assert call.service.text_prompt_queue.empty()
    assert call.tracker.current_turn() == turn
    assert call.service._state(call.conn_id).pending_interruption is None


def test_real_interruption_is_committed_on_the_second_matching_hypothesis(call):
    call.configure()
    _, key = _coach_is_reading(call)
    generation = call.unit.cancel_scope.generation

    interruption = call.speech()
    call.partial(interruption, "Wait")
    _wait_for(lambda: _pending(call) is not None and _pending(call).previous_words == 1)
    assert call.unit.cancel_scope.generation == generation  # one hypothesis is not enough
    call.partial(interruption, "Wait, wait, can")
    events = call.receive_until("input_audio_buffer.speech_started")

    done = next(event for event in events if event["type"] == "response.done")
    assert (done["response"]["status"], done["response"]["status_details"]["reason"]) == ("cancelled", "turn_detected")
    assert call.unit.cancel_scope.generation == generation + 1
    assert call.tracker.current_turn() == interruption

    call.stop(interruption, 2.0)
    call.final(interruption, "Wait, wait, can you say that again?")
    request = call.service.text_prompt_queue.get(timeout=2)
    assert (request.turn_id, request.turn_revision) == interruption
    assert call.user_messages() == ["The answer is softmax.", "Wait, wait, can you say that again?"]

    # The user item reaches the client as usual, ahead of the reply to it.
    call.reply(interruption, request.response_key, "Of course.")
    events = call.receive_until("response.output_audio.delta")
    types = [event["type"] for event in events]
    transcript = types.index("conversation.item.input_audio_transcription.completed")
    assert types.index("input_audio_buffer.speech_stopped") < transcript < types.index("response.created")
    completed = events[transcript]
    assert completed["transcript"] == "Wait, wait, can you say that again?"


def test_a_short_real_interruption_is_committed_by_its_final_transcript(call):
    call.configure()
    _, key = _coach_is_reading(call)

    interruption = call.speech()
    call.stop(interruption, 0.4)
    call.final(interruption, "No.")
    events = call.receive_until("input_audio_buffer.speech_started")

    types = [event["type"] for event in events]
    assert types.index("response.done") < types.index("input_audio_buffer.speech_started")
    assert call.user_messages()[-1] == "No."
    assert call.service.text_prompt_queue.get(timeout=2).turn_id == interruption[0]


def test_reply_that_ended_is_still_protected_while_its_audio_plays(call):
    """The reply is fully generated, but the client is still playing it."""
    call.configure()
    turn, key = _coach_is_reading(call)
    call.finish(key)

    cough = call.speech()
    call.stop(cough)
    call.final(cough, "")
    _wait_for(lambda: call.tracker.current_turn() == turn)
    # The next question is a normal turn; nothing about the cough reached the client.
    turn_3, _ = call.user_says("What was the question?")
    call.receive_until("input_audio_buffer.speech_started", lambda event: event["audio_start_ms"] == 1400)
    started = [event["audio_start_ms"] for event in call.received if event["type"].endswith("speech_started")]
    assert started == [0, 1400]
    assert call.user_messages() == ["The answer is softmax.", "What was the question?"]
    assert turn_3 == ("turn_3", 0)


def test_short_answer_as_the_reply_ends_is_a_turn(call):
    """'Yes.' to the question the coach is just finishing must not be thrown away."""
    call.configure()
    turn, key = call.user_says("Let's start.")
    call.reply(turn, key, "Ready for the first question?", seconds=0.5)
    call.finish(key)

    answer = call.speech()
    _wait_for(lambda: _pending(call) is not None)
    call.stop(answer, 0.3)
    call.final(answer, "Yes.")

    request = call.service.text_prompt_queue.get(timeout=2)
    assert (request.turn_id, request.turn_revision) == answer
    assert call.user_messages() == ["Let's start.", "Yes."]
    call.receive_until("input_audio_buffer.speech_started", lambda event: event["audio_start_ms"] == 1000)


def test_backchannel_with_seconds_of_reply_left_is_ignored(call):
    call.configure()
    turn, key = call.user_says("Explain attention.")
    call.reply(turn, key, "Attention weighs every token against every other token.", seconds=4.0)
    call.finish(key)

    backchannel = call.speech()
    call.stop(backchannel, 0.3)
    call.final(backchannel, "Yeah.")
    _wait_for(lambda: call.tracker.current_turn() == turn)

    assert call.user_messages() == ["Explain attention."]
    assert call.service.text_prompt_queue.empty()


def test_short_answer_during_a_text_only_side_response_is_a_turn(call):
    """Out-of-band text responses (e.g. a client's background extraction) are not speech to protect."""
    call.configure()
    turn, key = call.user_says("Let's start.")
    call.reply(turn, key, "Ready for the first question?", seconds=0.5)
    call.finish(key)
    call.ws.send_json({"type": "response.create", "response": {"conversation": "none", "output_modalities": ["text"]}})
    call.receive_until("response.created")
    call.service.text_prompt_queue.get(timeout=2)

    answer = call.speech()
    _wait_for(lambda: _pending(call) is not None)
    call.stop(answer, 0.3)
    call.final(answer, "Yes.")

    request = call.service.text_prompt_queue.get(timeout=2)
    assert (request.turn_id, request.turn_revision) == answer
    assert call.user_messages() == ["Let's start.", "Yes."]


def test_speech_while_the_assistant_is_silent_is_a_normal_turn(call):
    call.configure()
    call.speech()
    assert call.receive_until("input_audio_buffer.speech_started")
    assert call.service._state(call.conn_id).pending_interruption is None


def test_continuing_an_unanswered_turn_cancels_its_stale_reply_at_once(call):
    """A reopened turn is the user still talking, not a possible cough."""
    call.configure()
    turn, key = call.user_says("Is it the")
    assert call.service._state(call.conn_id).response_pending
    generation = call.unit.cancel_scope.generation

    call.speech(reopen=True)

    _wait_for(lambda: call.unit.cancel_scope.generation == generation + 1)
    assert _pending(call) is None
    assert call.service.text_prompt_queue.empty()


def test_without_the_option_speech_cancels_at_once(call):
    call.configure(None)
    _, key = _coach_is_reading(call)
    call.speech()
    events = call.receive_until("input_audio_buffer.speech_started")
    assert any(event["type"] == "response.done" and event["response"]["status"] == "cancelled" for event in events)


def test_no_transcript_resumes_after_the_timeout(call):
    call.configure({"timeout_ms": 50}, extensions=["speech_to_speech.interruption"])
    turn, key = _coach_is_reading(call)

    noise = call.speech()
    call.receive_until("speech_to_speech.interruption.pending")
    call.stop(noise)  # VAD discarded it; no transcript follows
    resolved = call.receive_until("speech_to_speech.interruption.resolved")[-1]

    assert resolved["interrupted"] is False
    assert call.tracker.current_turn() == turn
    assert call.finish(key)["response"]["status"] == "completed"


def test_failed_transcription_is_not_an_interruption(call):
    call.configure()
    turn, key = _coach_is_reading(call)
    start = len(call.received)
    noise = call.speech()
    call.stop(noise)
    call.unit.text_output_queue.put(
        TranscriptionFailedEvent(message="decoder error", turn_id=noise[0], turn_revision=noise[1])
    )
    _wait_for(lambda: call.tracker.current_turn() == turn)
    call.finish(key)
    assert all(event["type"] not in _INPUT_EVENTS for event in call.received[start:])


def test_opt_in_events_announce_the_pending_decision(call):
    call.configure({"playback": "pause"}, extensions=["speech_to_speech.interruption"])
    _, key = _coach_is_reading(call)

    cough = call.speech()
    pending = call.receive_until("speech_to_speech.interruption.pending")[-1]
    assert (pending["playback"], pending["audio_start_ms"]) == ("pause", call.audio_ms)
    call.stop(cough)
    call.final(cough, "Uh.")
    resolved = call.receive_until("speech_to_speech.interruption.resolved")[-1]
    assert (resolved["interrupted"], resolved["transcript"]) == (False, "Uh.")

    real = call.speech()
    call.receive_until("speech_to_speech.interruption.pending")
    call.stop(real)
    call.final(real, "Stop.")
    events = call.receive_until("input_audio_buffer.speech_started")
    resolved = next(event for event in events if event["type"] == "speech_to_speech.interruption.resolved")
    assert resolved["interrupted"] is True


def test_tool_follow_up_during_a_cough_is_answered(call):
    """The coach called a tool and the client answers it while the user coughs."""
    from openai.types.responses import ResponseFunctionToolCall

    call.configure()
    turn, key = call.user_says("Next question please.")
    tool = ResponseFunctionToolCall(
        type="function_call", id="fc_q", call_id="call_q", name="next_question", arguments="{}"
    )
    call.reply(
        turn,
        key,
        "",
        parts=[AssistantTextPart(text="Sure, one moment."), AssistantToolCallPart(tool=tool)],
    )
    call.finish(key)

    cough = call.speech()
    _wait_for(lambda: _pending(call) is not None)
    call.ws.send_json(
        {
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": "call_q", "output": "What is a token?"},
        }
    )
    call.receive_until("conversation.item.created")
    call.ws.send_json({"type": "response.create", "response": {"metadata": {TOOL_FOLLOWUP_METADATA_KEY: '["call_q"]'}}})
    call.receive_until("response.created")
    request = call.service.text_prompt_queue.get(timeout=2)
    assert (request.turn_id, request.turn_revision) == turn

    follow_up = (request.turn_id, request.turn_revision)
    call.reply(follow_up, request.response_key, "Question three: what is a token?")
    events = call.receive_until("response.output_audio_transcript.delta")
    assert events[-1]["delta"] == "Question three: what is a token?"

    call.stop(cough)
    call.final(cough, "Ahem.")
    assert call.finish(request.response_key)["response"]["status"] == "completed"
    assert call.tracker.current_turn() == turn


# ── Transports ───────────────────────────────────────────────────────


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_websocket_playback_estimate_stops_while_the_client_is_paused():
    clock = _Clock()
    transport = WebSocketTransport(websocket=None, clock=clock)
    transport._playback_end = clock.now + 2.0
    assert transport.has_unplayed_audio()

    transport.set_playback("pause")
    clock.now += 5.0
    assert transport.has_unplayed_audio()
    transport.set_playback("play")
    clock.now += 1.5
    assert transport.discard_pending_audio() == pytest.approx(500.0)
    assert not transport.has_unplayed_audio()


def test_websocket_discard_while_paused_counts_the_paused_audio_as_unplayed():
    clock = _Clock()
    transport = WebSocketTransport(websocket=None, clock=clock)
    transport._playback_end = clock.now + 2.0
    transport.set_playback("pause")
    clock.now += 3.0
    assert transport.discard_pending_audio() == pytest.approx(2000.0)
    transport.set_playback("play")
    assert not transport.has_unplayed_audio()


def test_send_loop_wakes_for_the_transcript_deadline(call):
    call.configure({"timeout_ms": 300})
    _coach_is_reading(call)
    noise = call.speech()
    _wait_for(lambda: _pending(call) is not None)
    call.stop(noise)
    _wait_for(lambda: _pending(call) is None)
    with pytest.raises(Empty):
        call.service.text_prompt_queue.get_nowait()
