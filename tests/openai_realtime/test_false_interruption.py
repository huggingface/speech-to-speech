"""Barge-in that waits for words: coughs and backchannels do not cancel a reply (#878)."""

import threading
import time

import numpy as np
import pytest
from starlette.testclient import TestClient

from speech_to_speech.api.openai_realtime.false_interruption import (
    FalseInterruptionConfig,
    PendingInterruption,
    Verdict,
    parse_false_interruption,
    resume_rewind_samples,
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

# ── Word counting, config, rewind, pending state ─────────────────────


@pytest.mark.parametrize(
    ("transcript", "words"),
    [
        ("", 0),
        ("Uh.", 0),
        ("Mm-hmm.", 0),
        ("Uh-huh, yeah right.", 0),
        ("Yeah, I see.", 0),
        ("Ahem.", 0),
        ("[coughs] (laughing) Hmm.", 0),
        ("Hmmmm, Uhhh, Mhmm, Yeahhh.", 0),
        ("Haha! Hehehe.", 0),
        # Direct answers, STT mishearings, and real words count.
        ("Yes.", 1),
        ("Sure.", 1),
        ("No.", 1),
        ("A ham.", 2),
        ("He. Hi. See. Good.", 4),
        ("Okay, stop.", 1),
        ("Haha, no.", 1),
        ("Wait, wait, can you say that again?", 7),
        ("什么意思", 4),
    ],
)
def test_word_counting_ignores_fillers_and_backchannels(transcript, words):
    assert FalseInterruptionConfig().word_count(transcript) == words


def test_custom_fillers_and_backchannels_per_language():
    french = FalseInterruptionConfig(fillers=("euh",), backchannels=("ouais", "d'accord"))
    assert french.word_count("Euh, ouais, d'accord.") == 0
    assert french.word_count("Uh, yeah.") == 2
    assert FalseInterruptionConfig(backchannels=()).word_count("Mm-hmm, yeah.") == 1
    assert FalseInterruptionConfig(fillers=()).word_count("Uh, yeah.") == 1


def test_session_config_validation_and_server_default():
    assert parse_false_interruption(None) is None
    assert parse_false_interruption(False) is None
    assert parse_false_interruption(True) == FalseInterruptionConfig()
    assert parse_false_interruption({"min_words": 2, "timeout_ms": 500}) == FalseInterruptionConfig(
        min_words=2, timeout_ms=500
    )
    for bad in ({"min_words": 0}, {"timeout_ms": 0}, {"playback": "pause"}, {"unknown": 1}, "yes"):
        with pytest.raises(ValueError, match="false_interruption"):
            parse_false_interruption(bad)

    service = RealtimeService(default_false_interruption=FalseInterruptionConfig())
    conn_id = service.register()
    cfg = service._state(conn_id).runtime_config
    assert cfg.false_interruption == FalseInterruptionConfig()

    def apply(td):
        ev = service.parse_client_event(
            {"type": "session.update", "session": {"type": "realtime", "audio": {"input": {"turn_detection": td}}}}
        )
        return service.handle_session_update(conn_id, ev)

    err = apply({"type": "server_vad", "false_interruption": {"min_words": 0}})
    assert err is not None and "min_words" in err.error.message
    assert apply({"type": "server_vad", "threshold": 0.6}) is None
    assert cfg.false_interruption == FalseInterruptionConfig()
    assert apply({"type": "server_vad", "false_interruption": {"min_words": 2}}) is None
    assert cfg.false_interruption == FalseInterruptionConfig(min_words=2)
    echoed = service.build_session_updated(conn_id).model_dump()["session"]["audio"]["input"]["turn_detection"]
    assert echoed["false_interruption"] == {"min_words": 2}
    # Explicitly clearing turn_detection disables the server default instead of re-enabling it.
    assert apply(None) is None and cfg.false_interruption is None
    assert RuntimeConfig().false_interruption is None


def _tone(ms, value=8000, rate=48_000):
    return (value * np.sin(2 * np.pi * 200 * np.arange(rate * ms // 1000) / rate)).astype(np.int16)


def _silence(ms, rate=48_000):
    return np.zeros(rate * ms // 1000, dtype=np.int16)


@pytest.mark.parametrize(
    ("played", "replayed_ms"),
    [
        (np.concatenate([_tone(300), _silence(80), _tone(200)]), 210),  # mid-word -> replay + 10 ms gap
        (np.concatenate([_silence(100), _tone(500)]), 510),  # gap at very start of 600 ms window
        (np.concatenate([_tone(300), _silence(80)]), 0),  # paused between words
        (_tone(700), 0),  # continuous tone
        (np.concatenate([_tone(300), _silence(30), _tone(200)]), 0),  # short consonant dip
        (np.concatenate([_tone(300), _silence(80), _tone(650)]), 0),  # gap outside 600 ms window
        (np.zeros(0, dtype=np.int16), 0),  # empty
    ],
)
def test_resume_replays_from_the_last_gap_between_words(played, replayed_ms):
    assert resume_rewind_samples(played, 48_000) == 48 * replayed_ms


def test_pending_interruption_partials_and_reopened_revisions():
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

    pending = PendingInterruption.begin(
        SpeechStartedEvent(turn_id="turn_2", turn_revision=0), FalseInterruptionConfig()
    )
    assert pending.observe(SpeechStartedEvent(turn_id="turn_2", turn_revision=1, reopened=True)) is Verdict.HOLD
    empty_old = TranscriptionCompletedEvent(transcript="", turn_id="turn_2", turn_revision=0)
    assert pending.observe(empty_old) is Verdict.DROP and empty_old not in pending.events
    assert pending.revisions == {0, 1}
    words_old = TranscriptionCompletedEvent(transcript="Wait.", turn_id="turn_2", turn_revision=0)
    assert pending.observe(words_old) is Verdict.CONFIRM and pending.transcript == "Wait."


# ── Speculative turn tracker ─────────────────────────────────────────


def _answering_turn(tracker):
    turn = tracker.start_turn()
    tracker.segment_finalized(1000)
    assert tracker.gate(*turn, commit=True).action.value == "accept"
    return turn


def test_provisional_turn_lifecycle_and_reopened_revision_guard():
    tracker = SpeculativeTurnTracker()
    first = _answering_turn(tracker)
    tracker.close(*first)  # closed reply whose tool follow-up may still generate

    # Starting provisional atomically keeps the closed predecessor relevant (no cross-thread window).
    second_id, _, reopened = tracker.speech_started(2000, provisional=True)
    assert not reopened and tracker.current_is_provisional() and tracker.is_latest(*first)

    # A second provisional turn preserves the original predecessor; discarding restores it.
    third_id, third_rev = tracker.start_turn(provisional=True)
    assert tracker.is_latest(*first) and not tracker.discard_provisional(second_id)
    assert tracker.discard_provisional(third_id, third_rev)
    assert tracker.current_turn() == first and tracker.phase is TurnPhase.CLOSED
    assert not tracker.current_is_provisional() and not tracker.is_latest(third_id, third_rev)

    # Dismissing rev 0 after VAD reopened rev 1 must leave rev 1 provisional and current.
    fourth_id, _ = tracker.start_turn(provisional=True)
    tracker.segment_finalized(2000)
    assert tracker.speech_started(2300) == (fourth_id, 1, True)
    assert not tracker.discard_provisional(fourth_id, 0)
    assert tracker.current_turn() == (fourth_id, 1) and tracker.is_provisional(fourth_id)
    assert tracker.discard_provisional(fourth_id, 1) and tracker.current_turn() == first

    # Confirming supersedes the predecessor.
    fifth = tracker.start_turn(provisional=True)
    tracker.confirm_provisional(fifth[0])
    assert not tracker.is_latest(*first) and tracker.current_turn() == fifth


def test_turn_answered_during_provisional_turn_is_committed_and_closed():
    """The reply to turn 1 starts and closes while a cough is held as turn 2."""
    tracker = SpeculativeTurnTracker()
    first = tracker.start_turn()
    tracker.segment_finalized(1000)
    second, _ = tracker.start_turn(provisional=True)

    assert tracker.gate(*first, commit=True).action.value == "accept" and tracker.is_committed(*first)
    tracker.close(*first)
    assert tracker.discard_provisional(second, 0)
    assert tracker.current_turn() == first and tracker.phase is TurnPhase.CLOSED
    assert tracker.is_committed(*first) and not tracker.can_reopen(3000)
    assert tracker.speech_started(3000)[2] is False


# ── Router end-to-end ────────────────────────────────────────────────


class _Call:
    """One WebSocket session, with the pipeline threads played by the test."""

    def __init__(self, ws, unit):
        self.ws, self.unit, self.service = ws, unit, unit.service
        self.tracker = unit.service.speculative_turns
        self.conn_id = next(iter(self.service._conns))
        self.received: list[dict] = []
        self.audio_ms = 0

    def configure(
        self, false_interruption=True, *, extensions=("speech_to_speech.interruption",), interrupt_response=True
    ):
        td = {"type": "server_vad", "interrupt_response": interrupt_response, "false_interruption": false_interruption}
        session = {"type": "realtime", "audio": {"input": {"turn_detection": td}}, "extensions": list(extensions)}
        self.ws.send_json({"type": "session.update", "session": session})
        self.receive_until("session.updated")

    def receive_until(self, event_type, predicate=lambda _e: True) -> list[dict]:
        events = []
        while True:
            event = self.ws.receive_json()
            events.append(event)
            self.received.append(event)
            if event["type"] == "error":
                raise AssertionError(event)
            if event["type"] == event_type and predicate(event):
                return events

    def speech(self, *, reopen=False, provisional=False):
        if reopen:
            tid, rev, reopened = self.tracker.speech_started(self.audio_ms)
            assert reopened
        else:
            tid, rev = self.tracker.start_turn(provisional=provisional)
        self.unit.text_output_queue.put(
            SpeechStartedEvent(turn_id=tid, turn_revision=rev, audio_start_ms=self.audio_ms, reopened=reopen)
        )
        return tid, rev

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
        return turn, self.service.text_prompt_queue.get(timeout=2).response_key

    def reply(self, turn, key, text, seconds=2.0, parts=None):
        gen = self.unit.cancel_scope.generation
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
                    cancel_generation=gen,
                    history=history,
                )
            )
        self.audio(key, seconds)

    def audio(self, key, seconds):
        gen = self.unit.cancel_scope.generation
        for _ in range(round(seconds * 10)):
            self.unit.output_queue.put(
                AudioOutput(audio=np.zeros(RATE // 10, dtype=np.int16), response_key=key, cancel_generation=gen)
            )

    def finish(self, key) -> dict:
        gen = self.unit.cancel_scope.generation
        self.unit.output_queue.put(AssistantResponseDoneEvent(response_key=key, cancel_generation=gen))
        self.unit.output_queue.put(AudioOutput(audio=AUDIO_RESPONSE_DONE, response_key=key, cancel_generation=gen))
        return self.receive_until("response.done")[-1]

    def user_messages(self):
        return [
            item.content[0].text
            for item in self.service._state(self.conn_id).runtime_config.chat.buffer
            if getattr(item, "role", None) == "user"
        ]


@pytest.fixture
def call():
    unit = make_pipeline_unit(service=RealtimeService(speculative_turns=SpeculativeTurnTracker()))
    unit.service.text_prompt_queue = unit.text_prompt_queue
    unit.service.should_listen = unit.should_listen
    with TestClient(create_app(pool=[unit], stop_event=threading.Event())) as client:
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
    turn, key = call.user_says("The answer is softmax.")
    call.reply(turn, key, "Correct! Question two: what does the scaling factor in attention prevent?")
    call.receive_until("response.output_audio.delta")
    return turn, key


@pytest.mark.parametrize("noise", ["", "Yeah."])
def test_cough_or_backchannel_does_not_cancel_the_reply(call, noise):
    call.configure()
    turn, key = _coach_is_reading(call)
    gen = call.unit.cancel_scope.generation

    start = len(call.received)
    cough = call.speech(provisional=True)
    call.audio(key, 0.1)
    call.receive_until("response.output_audio.delta")
    call.stop(cough, 0.35)
    call.partial(cough, noise)
    call.final(cough, noise)
    call.audio(key, 0.1)
    call.receive_until("response.output_audio.delta")
    done = call.finish(key)

    assert [e["type"] for e in call.received[start:] if e["type"] in _INPUT_EVENTS] == []
    assert done["response"]["status"] == "completed" and call.unit.cancel_scope.generation == gen
    assert call.user_messages() == ["The answer is softmax."] and call.service.text_prompt_queue.empty()
    assert call.tracker.current_turn() == turn and _pending(call) is None


def test_real_interruption_commits_on_two_partials_and_answers(call):
    call.configure()
    _coach_is_reading(call)
    gen = call.unit.cancel_scope.generation

    interruption = call.speech(provisional=True)
    call.partial(interruption, "Wait")
    _wait_for(lambda: _pending(call) is not None and _pending(call).previous_words == 1)
    assert call.unit.cancel_scope.generation == gen  # one hypothesis is not enough
    call.partial(interruption, "Wait, wait, can")
    events = call.receive_until("input_audio_buffer.speech_started")

    done = next(e for e in events if e["type"] == "response.done")
    assert (done["response"]["status"], done["response"]["status_details"]["reason"]) == ("cancelled", "turn_detected")
    assert call.unit.cancel_scope.generation == gen + 1 and call.tracker.current_turn() == interruption

    call.stop(interruption, 2.0)
    call.final(interruption, "Wait, wait, can you say that again?")
    req = call.service.text_prompt_queue.get(timeout=2)
    assert (req.turn_id, req.turn_revision) == interruption
    call.reply(interruption, req.response_key, "Of course.")
    types = [e["type"] for e in call.receive_until("response.output_audio.delta")]
    assert (
        types.index("input_audio_buffer.speech_stopped")
        < types.index("conversation.item.input_audio_transcription.completed")
        < types.index("response.created")
    )


def test_short_final_commits_and_reopened_rev1_survives_rev0_filler(call):
    """A short final ('No.') commits; rev 0 ('Uh.') racing rev 1 ('Uh. Wait, stop!') does not drop rev 1."""
    call.configure(True, extensions=["speech_to_speech.interruption"])
    _coach_is_reading(call)

    t2 = call.speech(provisional=True)
    call.receive_until("speech_to_speech.interruption.pending")
    call.stop(t2, 0.35)
    # VAD reopens rev 1 on the tracker BEFORE rev 0's transcript is processed.
    assert call.tracker.speech_started(call.audio_ms) == (t2[0], 1, True)
    call.final(t2, "Uh.")
    call.unit.text_output_queue.put(
        SpeechStartedEvent(turn_id=t2[0], turn_revision=1, audio_start_ms=call.audio_ms, reopened=True)
    )
    call.stop((t2[0], 1), 0.8)
    call.final((t2[0], 1), "Uh. Wait, stop!")

    events = call.receive_until("input_audio_buffer.speech_started")
    done = next(e for e in events if e["type"] == "response.done")
    assert (done["response"]["status"], done["response"]["status_details"]["reason"]) == ("cancelled", "turn_detected")
    req = call.service.text_prompt_queue.get(timeout=2)
    assert (req.turn_id, req.turn_revision) == (t2[0], 1) and call.user_messages()[-1] == "Uh. Wait, stop!"

    # Short single-word final commits without partials.
    call.reply((t2[0], 1), req.response_key, "Let me explain again.")
    call.receive_until("response.output_audio.delta")
    short = call.speech(provisional=True)
    call.receive_until("speech_to_speech.interruption.pending")
    call.stop(short, 0.3)
    call.final(short, "No.")
    events = call.receive_until("input_audio_buffer.speech_started")
    assert any(e["type"] == "speech_to_speech.interruption.resolved" and e["interrupted"] is True for e in events)
    assert call.service.text_prompt_queue.get(timeout=2).turn_id == short[0]


def test_reply_ending_and_out_of_band_rules(call):
    """'Yeah.' is ignored with 4 s left, but commits with <=1 s left even during a text-only OOB response."""
    call.configure()
    turn, key = call.user_says("Explain attention.")
    call.reply(turn, key, "Attention weighs every token against every other token.", seconds=4.0)
    call.finish(key)

    backchannel = call.speech(provisional=True)
    call.stop(backchannel, 0.3)
    call.final(backchannel, "Yeah.")
    _wait_for(lambda: call.tracker.current_turn() == turn)
    assert call.service.text_prompt_queue.empty()

    call.unit.session.transport.discard_pending_audio()
    turn2, key2 = call.user_says("Ready?")
    call.reply(turn2, key2, "Ready for the first question?", seconds=0.5)
    call.finish(key2)
    call.ws.send_json({"type": "response.create", "response": {"conversation": "none", "output_modalities": ["text"]}})
    call.receive_until("response.created")
    call.service.text_prompt_queue.get(timeout=2)

    answer = call.speech(provisional=True)
    _wait_for(lambda: _pending(call) is not None)
    call.stop(answer, 0.3)
    call.final(answer, "Yeah.")
    req = call.service.text_prompt_queue.get(timeout=2)
    assert (req.turn_id, req.turn_revision) == answer and call.user_messages()[-1] == "Yeah."


def test_silent_disabled_unpausable_and_reopened_unanswered_turn_do_not_hold(call):
    call.configure()
    t1 = call.speech(provisional=True)
    call.receive_until("input_audio_buffer.speech_started")
    assert _pending(call) is None and not call.tracker.is_provisional(t1[0])

    # Reopening the user's own unanswered turn cancels its stale reply immediately.
    call.stop(t1, 0.8)
    call.final(t1, "Is it the")
    call.service.text_prompt_queue.get(timeout=2)
    gen = call.unit.cancel_scope.generation
    call.speech(reopen=True)
    _wait_for(lambda: call.unit.cancel_scope.generation == gen + 1)
    assert _pending(call) is None

    # Option off: speech cancels an active reply immediately.
    call.configure(None)
    _coach_is_reading(call)
    call.speech()
    events = call.receive_until("input_audio_buffer.speech_started")
    assert any(e["type"] == "response.done" and e["response"]["status"] == "cancelled" for e in events)

    # Option on, but a WebSocket client that did not opt in to the events cannot
    # pause its player: it keeps the immediate barge-in instead of talking over the user.
    call.configure(True, extensions=[])
    _coach_is_reading(call)
    gen = call.unit.cancel_scope.generation
    unpausable = call.speech(provisional=True)
    _wait_for(lambda: call.unit.cancel_scope.generation == gen + 1)
    events = call.receive_until("input_audio_buffer.speech_started")
    assert any(e["type"] == "response.done" and e["response"]["status"] == "cancelled" for e in events)
    assert not any(e["type"].startswith("speech_to_speech.interruption") for e in events)
    assert _pending(call) is None and not call.tracker.is_provisional(unpausable[0])


def test_timeout_failed_transcription_and_pause_events(call, monkeypatch):
    states: list[bool] = []
    original = WebSocketTransport.set_paused
    monkeypatch.setattr(WebSocketTransport, "set_paused", lambda self, p: (states.append(p), original(self, p))[1])
    call.configure({"timeout_ms": 50}, extensions=["speech_to_speech.interruption"])
    turn, key = _coach_is_reading(call)

    noise = call.speech(provisional=True)
    assert call.receive_until("speech_to_speech.interruption.pending")[-1]["audio_start_ms"] == call.audio_ms
    call.stop(noise)
    assert call.receive_until("speech_to_speech.interruption.resolved")[-1]["interrupted"] is False
    assert call.tracker.current_turn() == turn

    start = len(call.received)
    fail = call.speech(provisional=True)
    call.receive_until("speech_to_speech.interruption.pending")
    call.stop(fail)
    call.unit.text_output_queue.put(TranscriptionFailedEvent(message="err", turn_id=fail[0], turn_revision=fail[1]))
    assert call.receive_until("speech_to_speech.interruption.resolved")[-1]["interrupted"] is False
    assert call.finish(key)["response"]["status"] == "completed"
    assert all(e["type"] not in _INPUT_EVENTS for e in call.received[start:])
    assert states == [True, False, True, False]


def test_cough_before_first_output_keeps_input_lifecycle_first(call):
    call.configure()
    turn, key = call.user_says("What does attention do?")
    cough = call.speech(provisional=True)
    _wait_for(lambda: _pending(call) is not None)

    call.reply(turn, key, "It weighs the parts of the input.")
    call.receive_until("response.output_audio.delta")
    types = [e["type"] for e in call.received]
    for ev in (
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        "conversation.item.input_audio_transcription.completed",
    ):
        assert types.index(ev) < types.index("response.created"), ev

    call.finish(key)
    call.stop(cough)
    call.final(cough, "")
    _wait_for(lambda: _pending(call) is None)
    assert call.tracker.current_turn() == turn and call.tracker.phase is TurnPhase.CLOSED


def test_interruption_after_pause_cuts_history_at_pause(call):
    now = [100.0]
    call.unit.session.transport._clock = lambda: now[0]
    call.configure(True, extensions=["speech_to_speech.interruption"])
    _, key = _coach_is_reading(call)
    # Wait for all 2.0 s of reply audio deltas to be sent before advancing the clock.
    _wait_for(lambda: call.unit.session.transport.unplayed_audio_ms() >= 1900.0)
    now[0] += 0.4

    real = call.speech(provisional=True)
    call.receive_until("speech_to_speech.interruption.pending")
    call.audio(key, 1.0)  # sent while paused: never heard
    _wait_for(lambda: call.unit.session.transport.unplayed_audio_ms() >= 2500.0)
    now[0] += 0.5
    call.stop(real)
    call.final(real, "Stop.")
    call.receive_until("input_audio_buffer.speech_started")

    played_s = list(call.service._state(call.conn_id).assistant_audio_cuts.values())[-1] / RATE
    assert played_s == pytest.approx(0.4, abs=0.05)


def test_response_cancel_during_hold_releases_before_next_response(call):
    call.configure(True, extensions=["speech_to_speech.interruption"])
    _coach_is_reading(call)
    held = call.speech(provisional=True)
    call.receive_until("speech_to_speech.interruption.pending")
    call.ws.send_json({"type": "response.cancel"})
    _wait_for(lambda: _pending(call) is None)
    events = call.receive_until("input_audio_buffer.speech_started")
    assert any(e["type"] == "speech_to_speech.interruption.resolved" and e["interrupted"] is True for e in events)

    call.ws.send_json({"type": "response.create"})
    key = call.service.text_prompt_queue.get(timeout=2).response_key
    call.reply(None, key, "Here is another way to put it.")
    created = call.receive_until("response.created")[-1]["response"]["id"]
    call.stop(held)
    call.final(held, "Wait.")
    done = call.finish(key)
    assert (done["response"]["id"], done["response"]["status"]) == (created, "completed")


def test_session_updates_during_hold(call):
    # 1. Removing the interruption extension mid-hold still delivers .resolved so playback is not stuck paused.
    call.configure(True, extensions=["speech_to_speech.interruption"])
    _, key = _coach_is_reading(call)
    cough = call.speech(provisional=True)
    call.receive_until("speech_to_speech.interruption.pending")
    call.configure(True, extensions=[])
    call.stop(cough)
    call.final(cough, "")
    assert call.receive_until("speech_to_speech.interruption.resolved")[-1]["interrupted"] is False

    # 2. Setting interrupt_response=false mid-hold releases speech without cancelling and tells client to resume.
    call.configure(True, extensions=["speech_to_speech.interruption"])
    call.speech(provisional=True)
    call.receive_until("speech_to_speech.interruption.pending")
    call.configure(True, extensions=["speech_to_speech.interruption"], interrupt_response=False)
    events = call.receive_until("input_audio_buffer.speech_started")
    assert any(e["type"] == "speech_to_speech.interruption.resolved" and e["interrupted"] is False for e in events)
    assert not any(e["type"] == "response.done" for e in events)
    assert call.finish(key)["response"]["status"] == "completed"

    # 3. Turning false_interruption off mid-hold releases and cancels the active reply.
    call.configure(True, extensions=["speech_to_speech.interruption"], interrupt_response=True)
    _coach_is_reading(call)
    call.speech(provisional=True)
    call.receive_until("speech_to_speech.interruption.pending")
    call.configure(False, extensions=["speech_to_speech.interruption"])
    events = call.receive_until("input_audio_buffer.speech_started")
    assert any(e["type"] == "response.done" and e["response"]["status"] == "cancelled" for e in events)


def test_tool_follow_up_during_a_cough_is_answered(call):
    from openai.types.responses import ResponseFunctionToolCall

    call.configure()
    turn, key = call.user_says("Next question please.")
    tool = ResponseFunctionToolCall(type="function_call", id="fc_q", call_id="call_q", name="next_q", arguments="{}")
    call.reply(turn, key, "", parts=[AssistantTextPart(text="Sure, one moment."), AssistantToolCallPart(tool=tool)])
    call.finish(key)

    cough = call.speech(provisional=True)
    # Turn 1 remains valid immediately, before the router even processes SpeechStartedEvent.
    assert call.tracker.is_latest(*turn)
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
    req = call.service.text_prompt_queue.get(timeout=2)
    assert (req.turn_id, req.turn_revision) == turn

    call.reply((req.turn_id, req.turn_revision), req.response_key, "Question three: what is a token?")
    assert (
        call.receive_until("response.output_audio_transcript.delta")[-1]["delta"] == "Question three: what is a token?"
    )
    call.stop(cough)
    call.final(cough, "Ahem.")
    assert call.finish(req.response_key)["response"]["status"] == "completed" and call.tracker.current_turn() == turn


def test_a_failed_replay_ends_the_hold(call, monkeypatch):
    """The send loop logs a failing event and goes on; the hold must not outlive it."""
    from speech_to_speech.api.openai_realtime import websocket_router

    original = websocket_router._deliver_text_event
    armed, failed = [], []

    async def fail_once(unit, session, transport, session_id, text_msg, parked_index=None):
        if armed and not failed and isinstance(text_msg, SpeechStartedEvent):
            failed.append(True)
            raise RuntimeError("replay failed")
        await original(unit, session, transport, session_id, text_msg, parked_index)

    monkeypatch.setattr(websocket_router, "_deliver_text_event", fail_once)
    call.configure(True, extensions=["speech_to_speech.interruption"])
    _coach_is_reading(call)
    armed.append(True)

    real = call.speech(provisional=True)
    call.receive_until("speech_to_speech.interruption.pending")
    call.stop(real)
    call.final(real, "Stop.")
    _wait_for(lambda: bool(failed) and _pending(call) is None)
    assert call.unit.session.transport._paused_at is None
    call.speech(provisional=True)
    call.receive_until("speech_to_speech.interruption.pending")


def test_websocket_transport_pause_clock():
    now = 100.0
    transport = WebSocketTransport(websocket=None, clock=lambda: now)
    transport._playback_end = now + 2.0
    transport.set_paused(True)
    now += 5.0
    assert transport.unplayed_audio_ms() == pytest.approx(2000.0)
    transport.set_paused(False)
    now += 1.5
    assert transport.discard_pending_audio() == pytest.approx(500.0)

    # Discard while paused counts paused audio as unplayed and ends the pause for the next reply.
    transport._playback_end = now + 2.0
    transport.set_paused(True)
    now += 3.0
    assert transport.discard_pending_audio() == pytest.approx(2000.0)
    transport._playback_end = now + 1.0
    now += 0.6
    assert transport.unplayed_audio_ms() == pytest.approx(400.0)
