"""Tests for server-side PCM audio-replay false-interruption recovery."""

from __future__ import annotations

import asyncio
import base64
import time
from contextlib import contextmanager
from threading import Event as ThreadingEvent
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from openai.types.realtime import RealtimeConversationItemFunctionCall, RealtimeConversationItemFunctionCallOutput
from openai.types.responses import ResponseFunctionToolCall
from starlette.websockets import WebSocketState

from speech_to_speech.api.openai_realtime import websocket_router as router_module
from speech_to_speech.api.openai_realtime.false_interruption import (
    MAX_RETAINED_AUDIO_SAMPLES,
    PIPELINE_SAMPLE_RATE,
    AssistantAudioBuffer,
    apply_fade_in,
    count_interruption_words,
    resume_rewind_samples,
)
from speech_to_speech.api.openai_realtime.pipeline_unit import SessionState
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.transports import WebSocketTransport
from speech_to_speech.LLM.chat import Chat, make_assistant_message, make_user_message
from speech_to_speech.pipeline.events import (
    AssistantOutputEvent,
    AssistantResponseDoneEvent,
    AssistantToolCallReadyEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TranscriptionCompletedEvent,
    TranscriptionFailedEvent,
)
from speech_to_speech.pipeline.history import ResponseHistory
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, AssistantToolCallPart, AudioOutput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from tests.openai_realtime.conftest import make_pipeline_unit

RATE = PIPELINE_SAMPLE_RATE


def _tone(samples: int, amplitude: int = 8000) -> bytes:
    return np.full(samples, amplitude, dtype=np.int16).tobytes()


def _silence(samples: int) -> bytes:
    return np.zeros(samples, dtype=np.int16).tobytes()


def _make_unit(*, min_words: int = 1, timeout_ms: int = 2000, speculative: bool = False):
    unit = make_pipeline_unit()
    unit.service = RealtimeService(
        text_prompt_queue=unit.text_prompt_queue,
        should_listen=unit.should_listen,
        speculative_turns=SpeculativeTurnTracker() if speculative else None,
        min_interruption_words=min_words,
        false_interruption_timeout_ms=timeout_ms,
    )
    return unit


def _wait_for(predicate, *, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not predicate():
        time.sleep(0.01)
    assert predicate()


async def _await_for(predicate, *, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not predicate():
        await asyncio.sleep(0.01)
    assert predicate()


def _drain_until(ws, event_type: str, *, timeout_s: float = 2.0) -> list[dict]:
    events: list[dict] = []
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ev = ws.receive_json()
        events.append(ev)
        if ev["type"] == event_type:
            return events
    raise AssertionError(f"Timed out waiting for {event_type}; received {[e['type'] for e in events]}")


@contextmanager
def _ws_session(*, min_words: int = 1, timeout_ms: int = 2000, speculative: bool = False):
    unit = _make_unit(min_words=min_words, timeout_ms=timeout_ms, speculative=speculative)
    stop_event = ThreadingEvent()
    app = router_module.create_app(pool=[unit], stop_event=stop_event)
    try:
        with TestClient(app) as client, client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()  # session.created
            assert unit.session is not None
            yield unit, ws, unit.session.session_id
    finally:
        stop_event.set()


def _added_id(events: list[dict]) -> str:
    return next(ev["item"]["id"] for ev in events if ev["type"] == "response.output_item.added")


def _replayed_s(events: list[dict]) -> float:
    pcm = b"".join(base64.b64decode(ev["delta"]) for ev in events if ev["type"] == "response.output_audio.delta")
    return len(pcm) / 2 / RATE


def _queue_reply(unit, sid: str, rk: str, segments: list[tuple[str, bytes]], *, turn=None) -> str:
    chat = unit.service._state(sid).runtime_config.chat
    full = " ".join(text for text, _ in segments)
    hist = ResponseHistory.capture(
        chat, [make_assistant_message(full)], after_item_id=chat.history_anchor_id(), complete=True
    )
    t_id, t_rev = turn or (None, None)
    for text, pcm in segments:
        unit.output_queue.put(
            AssistantOutputEvent(text=text, response_key=rk, turn_id=t_id, turn_revision=t_rev, history=hist)
        )
        unit.output_queue.put(AudioOutput(audio=pcm, response_key=rk))
    unit.output_queue.put(AssistantResponseDoneEvent(response_key=rk, turn_id=t_id, turn_revision=t_rev, history=hist))
    unit.output_queue.put(AudioOutput(audio=AUDIO_RESPONSE_DONE, response_key=rk))
    return full


def _start_stream(unit, ws, rk: str, text: str, samples: int = RATE * 2, *, turn=None, seq=None, history=None) -> int:
    gen = unit.cancel_scope.generation
    t_id, t_rev = turn or (None, None)
    unit.output_queue.put(
        AssistantOutputEvent(
            text=text,
            response_key=rk,
            turn_id=t_id,
            turn_revision=t_rev,
            cancel_generation=gen,
            output_sequence=seq,
            history=history,
        )
    )
    unit.output_queue.put(AudioOutput(audio=_tone(samples), response_key=rk, cancel_generation=gen))
    _drain_until(ws, "response.output_audio.delta")
    return gen


def _finish_stream(unit, rk: str, gen: int, *, turn=None, status="completed", reason=None, history=None) -> None:
    t_id, t_rev = turn or (None, None)
    unit.output_queue.put(
        AssistantResponseDoneEvent(
            status=status,
            reason=reason,
            response_key=rk,
            turn_id=t_id,
            turn_revision=t_rev,
            cancel_generation=gen,
            history=history,
        )
    )
    unit.output_queue.put(AudioOutput(audio=AUDIO_RESPONSE_DONE, response_key=rk, cancel_generation=gen))


def _truncate(ws, item_id: str, audio_end_ms: int) -> None:
    ws.send_json(
        {"type": "conversation.item.truncate", "item_id": item_id, "content_index": 0, "audio_end_ms": audio_end_ms}
    )


def _start_cough(
    unit,
    ws,
    turn_id: str,
    *,
    rev: int = 0,
    item_id: str | None = None,
    audio_end_ms: int | None = None,
    total_played_ms: int | None = None,
) -> list[dict]:
    unit.text_output_queue.put(SpeechStartedEvent(turn_id=turn_id, turn_revision=rev))
    evs = _drain_until(ws, "input_audio_buffer.speech_started")
    if item_id is not None and (audio_end_ms is not None or total_played_ms is not None):
        if audio_end_ms is None:
            buf = unit.session.pending_false_interruption.audio_buffer
            assert total_played_ms is not None
            audio_end_ms = total_played_ms - buf.wire_start_sample * 1000 // RATE
        _truncate(ws, item_id, audio_end_ms)
        _wait_for(lambda: unit.session.pending_false_interruption.client_played_samples is not None)
    return evs


def _finish_cough(
    unit,
    ws,
    turn_id: str,
    *,
    transcript: str = "",
    rev: int = 0,
    duration_s: float = 0.2,
    drain: str | None = "response.done",
) -> list[dict]:
    unit.text_output_queue.put(SpeechStoppedEvent(turn_id=turn_id, turn_revision=rev, duration_s=duration_s))
    unit.text_output_queue.put(TranscriptionCompletedEvent(transcript=transcript, turn_id=turn_id, turn_revision=rev))
    return _drain_until(ws, drain) if drain else []


def _cough_replay(
    unit, ws, turn_id: str, item_id: str, *, audio_end_ms: int | None = None, total_played_ms: int | None = None
):
    _start_cough(unit, ws, turn_id, item_id=item_id, audio_end_ms=audio_end_ms, total_played_ms=total_played_ms)
    resumed = _finish_cough(unit, ws, turn_id)
    return _added_id(resumed), resumed


def _retrieve(ws, item_id: str, *, expect_error: bool = False) -> dict:
    ws.send_json({"type": "conversation.item.retrieve", "item_id": item_id})
    return _drain_until(ws, "error" if expect_error else "conversation.item.retrieved")[-1][
        "error" if expect_error else "item"
    ]


def _delete(ws, item_id: str) -> None:
    ws.send_json({"type": "conversation.item.delete", "item_id": item_id})
    assert _drain_until(ws, "conversation.item.deleted")[-1]["item_id"] == item_id


def _td_update(**td) -> dict:
    return {
        "type": "session.update",
        "session": {"type": "realtime", "audio": {"input": {"turn_detection": {"type": "server_vad", **td}}}},
    }


# ── Unit tests: word counting, rewind, fade-in, buffer, chat serializers ───


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", 0),
        ("   \t\n ", 0),
        ("... ?!", 0),
        ("yeah", 1),
        ("don't", 1),
        ("wait a second", 3),
        ("c'est bon", 2),
        ("¡Sí!", 1),
        ("مرحبا بك", 2),
        ("嗯", 1),
        ("等一下", 3),
        ("ちょっと", 4),
    ],
)
def test_count_interruption_words_is_multilingual(text: str, expected: int) -> None:
    assert count_interruption_words(text) == expected


def test_resume_rewind_snaps_to_word_gap_and_start() -> None:
    assert resume_rewind_samples(_tone(RATE), int(0.4 * RATE)) == 0
    pcm = _tone(RATE) + _silence(int(0.1 * RATE)) + _tone(int(0.3 * RATE))
    assert RATE <= resume_rewind_samples(pcm, int(1.4 * RATE)) <= int(1.1 * RATE)
    assert resume_rewind_samples(_tone(RATE) + _silence(int(0.4 * RATE)), int(1.4 * RATE)) >= int(1.35 * RATE)

    faded = np.frombuffer(apply_fade_in(_tone(1000, amplitude=10000)), dtype=np.int16)
    assert faded[0] == 0 and 0 < faded[80] < 10000 and faded[200] == 10000


def test_audio_buffer_tracks_segments_unsynthesized_tail_and_cap() -> None:
    buf = AssistantAudioBuffer(response_key="rk1", root_item_id="item_1", current_item_id="item_1")
    buf.append_text("Hello world.")
    buf.append_pcm(_tone(RATE))
    buf.append_text("Second sentence.")
    buf.append_pcm(_tone(RATE))
    buf.append_text("Pending third.")

    remaining = buf.remaining_segments_from(int(1.5 * RATE))
    assert [text for text, _ in remaining] == ["sentence.", "Pending third."]
    assert len(remaining[0][1]) == int(0.5 * RATE) * 2 and remaining[1][1] == b""

    buf.append_pcm(_silence(MAX_RETAINED_AUDIO_SAMPLES))
    assert len(buf.pcm) // 2 == MAX_RETAINED_AUDIO_SAMPLES and buf.pcm_base_sample == 2 * RATE


def test_chat_serializers_merge_consecutive_assistant_items_and_deduplicate_rewind() -> None:
    chat = Chat(10)
    chat.add_item(make_user_message("What is the capital of France?"))
    chat.add_heard_assistant_message("item_1", "The capital of France is", status="incomplete")
    chat.add_heard_assistant_message("item_2", "France is Paris.", status="completed", replay_source=("item_1", 3))
    assert chat.to_transformers_chat()[1]["content"] == "The capital of France is Paris."
    resp = chat.to_responses_api_chat()
    assert [m["role"] for m in resp] == ["user", "assistant"] and resp[1]["status"] == "completed"
    assert resp[1]["content"][0]["text"] == "The capital of France is Paris."

    chat.add_ordered_function_call(
        RealtimeConversationItemFunctionCall(
            id="fc_1", type="function_call", call_id="call_1", name="lookup", arguments="{}"
        )
    )
    chat.add_item(
        RealtimeConversationItemFunctionCallOutput(
            id="fco_1", type="function_call_output", call_id="call_1", output='{"ok": true}'
        )
    )
    tf = chat.to_transformers_chat()
    assert [m["role"] for m in tf] == ["user", "assistant", "tool"]
    assert (
        tf[1]["content"] == "The capital of France is Paris." and tf[1]["tool_calls"][0]["function"]["name"] == "lookup"
    )

    # Overlap tracks the preceding item's current cut: shortening past the rewind drops overlap to 0,
    # keeping a genuine repeated digit intact in both serializers; re-extending restores the overlap.
    c_cut = Chat(10)
    c_cut.add_item(make_user_message("Read the code."))
    c_cut.add_heard_assistant_message("r1", "The code is seven seven", status="incomplete")
    c_cut.add_heard_assistant_message("r2", "seven nine.", status="completed", replay_source=("r1", 4))
    assert c_cut.to_transformers_chat()[1]["content"] == "The code is seven seven nine."
    assert c_cut.to_responses_api_chat()[1]["content"][0]["text"] == "The code is seven seven nine."
    c_cut.truncate_assistant_message("r1", "The code is seven")
    assert c_cut.to_transformers_chat()[1]["content"] == "The code is seven seven nine."
    assert c_cut.to_responses_api_chat()[1]["content"][0]["text"] == "The code is seven seven nine."
    c_cut.truncate_assistant_message("r1", "The code is seven seven")
    assert c_cut.to_transformers_chat()[1]["content"] == "The code is seven seven nine."
    assert c_cut.to_responses_api_chat()[1]["content"][0]["text"] == "The code is seven seven nine."

    # Non-replay consecutive completed messages also keep legitimate repeated words.
    c2 = Chat(10)
    c2.add_item(make_user_message("Say it."))
    c2.add_heard_assistant_message("a1", "Hello.", status="completed")
    c2.add_heard_assistant_message("a2", "Hello.", status="completed")
    assert c2.to_transformers_chat()[1]["content"] == "Hello. Hello."


# ── End-to-end Realtime router tests ───────────────────────────────────────


def test_completed_playback_replay_retrieve_and_delete() -> None:
    with _ws_session(min_words=1) as (unit, ws, sid):
        chat = unit.service._state(sid).runtime_config.chat
        chat.add_item(make_user_message("Explain photosynthesis."))
        full = _queue_reply(
            unit,
            sid,
            "rk_short",
            [
                ("Plants convert sunlight into energy.", _tone(RATE * 2) + _silence(int(0.1 * RATE))),
                ("Chlorophyll absorbs light in the leaves.", _tone(RATE * 2)),
            ],
        )
        item_1 = _added_id(_drain_until(ws, "response.done"))

        _start_cough(unit, ws, "turn_2", item_id=item_1, audio_end_ms=2000)
        before = _retrieve(ws, item_1)
        assert before["status"] == "incomplete" and before["content"][0]["transcript"] == "Plants convert sunlight into"

        resumed = _finish_cough(unit, ws, "turn_2", duration_s=0.3)
        types = [ev["type"] for ev in resumed]
        assert types.index("conversation.item.input_audio_transcription.completed") < types.index("response.created")
        assert 2.05 <= _replayed_s(resumed) <= 2.75
        item_2 = _added_id(resumed)
        assert item_2 != item_1

        assert [(m.role, m.status) for m in chat.buffer] == [
            ("user", None),
            ("assistant", "incomplete"),
            ("assistant", "completed"),
        ]
        assert chat.to_transformers_chat()[1]["content"] == full
        resp = chat.to_responses_api_chat()
        assert [m["role"] for m in resp] == ["user", "assistant"] and resp[1]["status"] == "completed"
        assert resp[1]["content"][0]["text"] == full

        got_1, got_2 = _retrieve(ws, item_1), _retrieve(ws, item_2)
        assert got_1["status"] == "incomplete" and got_1["content"][0]["transcript"] == "Plants convert sunlight into"
        assert got_2["status"] == "completed" and "Chlorophyll" in got_2["content"][0]["transcript"]

        _delete(ws, item_2)
        assert [(m.role, m.status, m.content[0].text) for m in chat.buffer] == [
            ("user", None, "Explain photosynthesis."),
            ("assistant", "incomplete", "Plants convert sunlight into"),
        ]
        assert _retrieve(ws, item_2, expect_error=True)["type"] == "conversation_item_not_found"
        assert _retrieve(ws, item_1)["status"] == "incomplete"


def test_cough_during_long_generating_turn_buffers_and_streams_seamlessly() -> None:
    with _ws_session(min_words=1, speculative=True) as (unit, ws, sid):
        tracker = unit.service.speculative_turns
        assert tracker is not None
        chat = unit.service._state(sid).runtime_config.chat
        chat.add_item(make_user_message("Tell me a story."))

        t1, r1 = tracker.start_turn()
        tracker.segment_finalized(1000, output_hold_ms=0)
        s1, s2, s3 = "Once upon a time.", "A robot learned to listen.", "And finished every sentence."
        hist1 = ResponseHistory.capture(chat, [make_assistant_message(s1)], after_item_id=chat.history_anchor_id())
        gen = _start_stream(unit, ws, "rk_long", s1, turn=(t1, r1), seq=0, history=hist1)

        t2, r2 = tracker.start_turn()
        cancel_evs = _start_cough(unit, ws, t2, rev=r2)
        assert any(ev["type"] == "response.done" and ev["response"]["status"] == "cancelled" for ev in cancel_evs)
        assert unit.cancel_scope.generation == gen and not unit.cancel_scope.discarding and tracker.is_latest(t1, r1)

        # Sentence 2 buffers privately while STT runs.
        hist2 = ResponseHistory.capture(
            chat, [make_assistant_message(f"{s1} {s2}")], after_item_id=chat.history_anchor_id()
        )
        unit.output_queue.put(
            AssistantOutputEvent(
                text=s2,
                response_key="rk_long",
                turn_id=t1,
                turn_revision=r1,
                cancel_generation=gen,
                output_sequence=1,
                history=hist2,
            )
        )
        unit.output_queue.put(AudioOutput(audio=_tone(RATE), cancel_generation=gen, response_key="rk_long"))
        _wait_for(lambda: unit.session.pending_false_interruption.audio_buffer.total_samples == RATE * 3)

        # Empty transcript replays s1 tail + s2 and stays open for live s3.
        tracker.segment_finalized(2000, output_hold_ms=800)
        _finish_cough(unit, ws, t2, rev=r2, duration_s=0.25, drain="response.output_audio.delta")
        assert unit.service._state(sid).in_response

        full = f"{s1} {s2} {s3}"
        hist3 = ResponseHistory.capture(
            chat, [make_assistant_message(full)], after_item_id=chat.history_anchor_id(), complete=True
        )
        unit.output_queue.put(
            AssistantOutputEvent(
                text=s3,
                response_key="rk_long",
                turn_id=t1,
                turn_revision=r1,
                cancel_generation=gen,
                output_sequence=2,
                history=hist3,
            )
        )
        unit.output_queue.put(AudioOutput(audio=_tone(RATE), cancel_generation=gen, response_key="rk_long"))
        _finish_stream(unit, "rk_long", gen, turn=(t1, r1), history=hist3)

        tail = _drain_until(ws, "response.done")
        assert s3 in "".join(ev["delta"] for ev in tail if ev["type"] == "response.output_audio_transcript.delta")
        assert hist3.wait_until_resolved(1.0)
        assert [(m.role, m.status, m.content[0].text) for m in chat.buffer] == [
            ("user", None, "Tell me a story."),
            ("assistant", "completed", full),
        ]


def test_real_interruption_and_reopened_speech_confirm_and_cancel_long_turn() -> None:
    # 1. Immediate 2-word interruption cancels long turn and queues prompt.
    with _ws_session(min_words=2, speculative=True) as (unit, ws, sid):
        tracker = unit.service.speculative_turns
        assert tracker is not None
        chat = unit.service._state(sid).runtime_config.chat
        chat.add_item(make_user_message("Start."))
        t1, r1 = tracker.start_turn()
        tracker.segment_finalized(500, output_hold_ms=0)
        gen = _start_stream(unit, ws, "rk1", "Long reply.", turn=(t1, r1))

        t2, r2 = tracker.start_turn()
        _start_cough(unit, ws, t2, rev=r2)
        assert unit.cancel_scope.generation == gen
        tracker.segment_finalized(1500, output_hold_ms=0)
        _finish_cough(unit, ws, t2, rev=r2, transcript="Stop please", duration_s=0.6, drain=None)
        _wait_for(lambda: not unit.text_prompt_queue.empty())
        assert unit.cancel_scope.generation == gen + 1 and unit.session.pending_false_interruption is None
        assert any(m.role == "user" and m.content[0].text == "Stop please" for m in chat.buffer)

    # 2. 1-word fragment held then reopened into 3 words confirms interruption instead of resuming.
    with _ws_session(min_words=2, speculative=True) as (unit, ws, sid):
        tracker = unit.service.speculative_turns
        assert tracker is not None
        chat = unit.service._state(sid).runtime_config.chat
        t1, r1 = tracker.start_turn()
        tracker.segment_finalized(500, output_hold_ms=0)
        gen = _start_stream(unit, ws, "rk2", "Long answer.", turn=(t1, r1))

        t2, r2 = tracker.start_turn()
        _start_cough(unit, ws, t2, rev=r2)
        tracker.segment_finalized(1000, output_hold_ms=1000)
        _finish_cough(unit, ws, t2, rev=r2, transcript="Wait", duration_s=0.3, drain=None)
        _wait_for(
            lambda: bool(
                unit.session.pending_false_interruption
                and unit.session.pending_false_interruption.short_transcript_pending
            )
        )
        assert unit.text_prompt_queue.empty() and unit.cancel_scope.generation == gen

        reopen_id, reopen_rev, reopened = tracker.speech_started(1200)
        assert reopened and reopen_id == t2
        unit.text_output_queue.put(SpeechStartedEvent(turn_id=reopen_id, turn_revision=reopen_rev, reopened=True))
        tracker.segment_finalized(1800, output_hold_ms=0)
        _finish_cough(unit, ws, reopen_id, rev=reopen_rev, transcript="Wait stop that", duration_s=0.8, drain=None)
        _wait_for(lambda: not unit.text_prompt_queue.empty())
        assert unit.cancel_scope.generation == gen + 1
        assert any(m.role == "user" and m.content[0].text == "Wait stop that" for m in chat.buffer)


def test_backchannel_stt_failure_and_late_backchannel_suppression() -> None:
    # 1. 1-word backchannel ("yeah" < min_words=2) resumes after reopen grace without queueing LLM.
    with _ws_session(min_words=2, speculative=True) as (unit, ws, sid):
        tracker = unit.service.speculative_turns
        assert tracker is not None
        chat = unit.service._state(sid).runtime_config.chat
        chat.add_item(make_user_message("Go on."))
        t1, r1 = tracker.start_turn()
        tracker.segment_finalized(500, output_hold_ms=0)
        full = _queue_reply(unit, sid, "rk_bc", [("First point. Second point.", _tone(RATE * 3))], turn=(t1, r1))
        _drain_until(ws, "response.done")

        t2, r2 = tracker.start_turn()
        _start_cough(unit, ws, t2, rev=r2)
        tracker.segment_finalized(1000, output_hold_ms=50)
        resumed = _finish_cough(unit, ws, t2, rev=r2, transcript="yeah", duration_s=0.25)
        assert any(ev["type"] == "response.output_audio.delta" for ev in resumed) and unit.text_prompt_queue.empty()
        assert [(m.role, m.status, m.content[0].text) for m in chat.buffer] == [
            ("user", None, "Go on."),
            ("assistant", "completed", full),
        ]

    # 2. TranscriptionFailedEvent immediately resumes playback.
    with _ws_session(min_words=1) as (unit, ws, sid):
        _queue_reply(unit, sid, "rk_fail", [("Keep listening.", _tone(RATE * 3))])
        _drain_until(ws, "response.done")
        _start_cough(unit, ws, "turn_f")
        unit.text_output_queue.put(SpeechStoppedEvent(turn_id="turn_f", turn_revision=0, duration_s=0.3))
        unit.text_output_queue.put(TranscriptionFailedEvent(message="stt err", turn_id="turn_f", turn_revision=0))
        assert any(ev["type"] == "response.output_audio.delta" for ev in _drain_until(ws, "response.done"))

    # 3. Late backchannel arriving after post-stop timeout recovery is suppressed from Chat & LLM.
    with _ws_session(min_words=2, timeout_ms=80) as (unit, ws, sid):
        chat = unit.service._state(sid).runtime_config.chat
        chat.add_item(make_user_message("Explain."))
        _queue_reply(unit, sid, "rk_late", [("First part. Second part.", _tone(RATE * 2))])
        _drain_until(ws, "response.done")

        _start_cough(unit, ws, "turn_late")
        unit.text_output_queue.put(SpeechStoppedEvent(turn_id="turn_late", turn_revision=0, duration_s=0.1))
        assert any(ev["type"] == "response.output_audio.delta" for ev in _drain_until(ws, "response.done"))
        unit.text_output_queue.put(TranscriptionCompletedEvent(transcript="yeah", turn_id="turn_late", turn_revision=0))
        _drain_until(ws, "conversation.item.input_audio_transcription.completed")
        assert unit.text_prompt_queue.empty() and [(m.role, m.status) for m in chat.buffer] == [
            ("user", None),
            ("assistant", "completed"),
        ]


def test_timeout_explicit_cancel_and_tool_call_recovery() -> None:
    # 1a. Continuing speech past timeout confirms real interruption and cancels generation.
    with _ws_session(min_words=1, timeout_ms=80) as (unit, ws, _sid):
        gen = _start_stream(unit, ws, "rk_to", "Streaming...")
        _start_cough(unit, ws, "turn_to")
        _wait_for(lambda: unit.session.pending_false_interruption is None)
        assert unit.cancel_scope.generation == gen + 1

    # 1b. Speech stops, then STT stalls past grace timeout -> resumes instead of leaving assistant silent.
    with _ws_session(min_words=1, timeout_ms=80) as (unit, ws, sid):
        chat = unit.service._state(sid).runtime_config.chat
        chat.add_item(make_user_message("Ask me."))
        _queue_reply(unit, sid, "rk_to_res", [("What is the capital of France?", _tone(RATE * 2))])
        _drain_until(ws, "response.done")
        _start_cough(unit, ws, "turn_to_res")
        time.sleep(0.05)
        unit.text_output_queue.put(SpeechStoppedEvent(turn_id="turn_to_res", turn_revision=0, duration_s=0.05))
        assert any(ev["type"] == "response.output_audio.delta" for ev in _drain_until(ws, "response.done"))
        assert [(m.role, m.status) for m in chat.buffer] == [("user", None), ("assistant", "completed")]

    # 1c. Grace period resets at speech_stopped so a real transcript after start+timeout still confirms.
    with _ws_session(min_words=1, timeout_ms=100) as (unit, ws, _sid):
        gen = _start_stream(unit, ws, "rk_reset", "Long answer.")
        _start_cough(unit, ws, "turn_reset")
        time.sleep(0.06)
        unit.text_output_queue.put(SpeechStoppedEvent(turn_id="turn_reset", turn_revision=0, duration_s=0.06))
        time.sleep(0.06)
        assert unit.session.pending_false_interruption is not None
        unit.text_output_queue.put(
            TranscriptionCompletedEvent(transcript="Stop please", turn_id="turn_reset", turn_revision=0)
        )
        _wait_for(lambda: unit.session.pending_false_interruption is None)
        assert unit.cancel_scope.generation == gen + 1

    # 2. Explicit client response.cancel aborts recovery so late empty STT does not replay.
    with _ws_session(min_words=1) as (unit, ws, _sid):
        gen = _start_stream(unit, ws, "rk_c", "Speaking.")
        _start_cough(unit, ws, "turn_c")
        ws.send_json({"type": "response.cancel"})
        _wait_for(lambda: unit.session.pending_false_interruption is None)
        assert unit.cancel_scope.generation == gen + 1
        evs = _finish_cough(unit, ws, "turn_c", drain="conversation.item.input_audio_transcription.completed")
        assert not any(ev["type"] == "response.created" for ev in evs)

    # 3a. Tool call during a cough (early side-channel + ordered) is held and delivered once on resume.
    with _ws_session(min_words=1, speculative=True) as (unit, ws, sid):
        chat = unit.service._state(sid).runtime_config.chat
        chat.add_item(make_user_message("Check my score."))
        turns = unit.service.speculative_turns
        assert turns is not None
        t0, r0 = turns.start_turn()
        turns.segment_finalized(500, output_hold_ms=0)
        gen = _start_stream(unit, ws, "rk_tool", "Checking your score.", samples=RATE, turn=(t0, r0), seq=0)

        t1, r1 = turns.start_turn()
        _start_cough(unit, ws, t1, rev=r1)
        tool = AssistantToolCallPart(
            tool=ResponseFunctionToolCall(
                id="fc_score", type="function_call", call_id="call_score", name="get_score", arguments='{"user":"a"}'
            )
        )
        unit.text_output_queue.put(
            AssistantToolCallReadyEvent(
                part=tool,
                output_sequence=1,
                turn_id=t0,
                turn_revision=r0,
                cancel_generation=gen,
                response_key="rk_tool",
            )
        )
        unit.output_queue.put(
            AssistantOutputEvent(
                parts=[tool],
                output_sequence=1,
                turn_id=t0,
                turn_revision=r0,
                cancel_generation=gen,
                response_key="rk_tool",
            )
        )
        _finish_stream(unit, "rk_tool", gen, turn=(t0, r0))
        _wait_for(lambda: unit.session.pending_false_interruption.audio_buffer.generation_done)
        assert unit.cancel_scope.generation == gen

        turns.segment_finalized(1000, output_hold_ms=0)
        resumed = _finish_cough(unit, ws, t1, rev=r1)
        fn_done = [ev for ev in resumed if ev["type"] == "response.function_call_arguments.done"]
        assert len(fn_done) == 1 and fn_done[0]["call_id"] == "call_score" and fn_done[0]["name"] == "get_score"

        ws.send_json(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": "call_score", "output": '{"score": 10}'},
            }
        )
        assert _drain_until(ws, "conversation.item.created")[-1]["item"]["call_id"] == "call_score"
        tf = chat.to_transformers_chat()
        assert [m["role"] for m in tf] == ["user", "assistant", "tool"] and tf[1]["content"] == "Checking your score."
        assert [m["type"] for m in chat.to_responses_api_chat()] == [
            "message",
            "message",
            "function_call",
            "function_call_output",
        ]

    # 3b. Tool call arriving during a real interruption is discarded and never delivered.
    with _ws_session(min_words=1) as (unit, ws, sid):
        chat = unit.service._state(sid).runtime_config.chat
        gen = _start_stream(unit, ws, "rk_td", "Checking...", samples=RATE, seq=0)
        _start_cough(unit, ws, "turn_td")
        drop = AssistantToolCallPart(
            tool=ResponseFunctionToolCall(
                id="fc_drop", type="function_call", call_id="call_drop", name="search", arguments="{}"
            )
        )
        unit.output_queue.put(
            AssistantOutputEvent(parts=[drop], output_sequence=1, response_key="rk_td", cancel_generation=gen)
        )
        _wait_for(lambda: len(unit.session.pending_false_interruption.audio_buffer.buffered_tool_events) == 1)

        evs = _finish_cough(
            unit,
            ws,
            "turn_td",
            transcript="Never mind",
            duration_s=0.4,
            drain="conversation.item.input_audio_transcription.completed",
        )
        assert not any(ev["type"] == "response.function_call_arguments.done" for ev in evs)
        assert (
            not any(getattr(item, "call_id", None) == "call_drop" for item in chat.buffer)
            and unit.cancel_scope.generation == gen + 1
        )


def test_multi_recovery_retrieve_delete_and_truncate_semantics() -> None:
    three = [
        ("First sentence.", _tone(RATE * 2)),
        ("Middle sentence.", _tone(RATE * 2)),
        ("Last sentence.", _tone(RATE * 2)),
    ]

    def _double_recover(unit, ws, sid, rk: str) -> tuple[str, str, str]:
        unit.service._state(sid).runtime_config.chat.add_item(make_user_message("Three sentences."))
        _queue_reply(unit, sid, rk, three)
        item_1 = _added_id(_drain_until(ws, "response.done"))
        item_2, _ = _cough_replay(unit, ws, f"{rk}_1", item_1, audio_end_ms=2000)
        item_3, _ = _cough_replay(unit, ws, f"{rk}_2", item_2, total_played_ms=4000)
        return item_1, item_2, item_3

    # 1. All 3 items stay retrievable; deleting item_3 drops only "Last sentence." with no overlap dupes.
    with _ws_session(min_words=1) as (unit, ws, sid):
        chat = unit.service._state(sid).runtime_config.chat
        item_1, item_2, item_3 = _double_recover(unit, ws, sid, "rk_dbl")
        assert len({item_1, item_2, item_3}) == 3
        for iid, status in ((item_1, "incomplete"), (item_2, "incomplete"), (item_3, "completed")):
            got = _retrieve(ws, iid)
            assert got["id"] == iid and got["status"] == status

        _delete(ws, item_3)
        assert [(m.role, m.status) for m in chat.buffer] == [
            ("user", None),
            ("assistant", "incomplete"),
            ("assistant", "incomplete"),
        ]
        assert chat.to_transformers_chat()[1]["content"] == "First sentence. Middle sentence."
        resp = chat.to_responses_api_chat()
        assert resp[1]["status"] == "incomplete" and resp[1]["content"][0]["text"] == "First sentence. Middle sentence."

    # 2. Deleting original item_1 preserves chronological order before later turns, and truncating item_3
    #    afterward must NOT restore deleted "First".
    with _ws_session(min_words=1) as (unit, ws, sid):
        chat = unit.service._state(sid).runtime_config.chat
        item_1, item_2, item_3 = _double_recover(unit, ws, sid, "rk_del")
        chat.add_item(make_user_message("Follow up."))

        _delete(ws, item_1)
        assert [(m.role, m.status) for m in chat.buffer] == [
            ("user", None),
            ("assistant", "incomplete"),
            ("assistant", "completed"),
            ("user", None),
        ]
        assert _retrieve(ws, item_1, expect_error=True)["type"] == "conversation_item_not_found"

        _truncate(ws, item_3, 1000)
        assert _retrieve(ws, item_3)["status"] == "incomplete"
        assert (
            "First" not in chat.to_transformers_chat()[1]["content"]
            and "Middle sentence." in chat.to_transformers_chat()[1]["content"]
        )
        resp_msg = chat.to_responses_api_chat()[1]
        assert resp_msg["status"] == "incomplete" and "First" not in resp_msg["content"][0]["text"]

        _delete(ws, item_2)
        _delete(ws, item_3)
        assert [m for m in chat.buffer if m.role == "assistant"] == []


def test_edge_cases_zero_audio_repeated_word_incomplete_and_end_of_replay_cough() -> None:
    # 1. Cough after AssistantOutputEvent before first PCM chunk still restores completed history.
    with _ws_session(min_words=1) as (unit, ws, sid):
        chat = unit.service._state(sid).runtime_config.chat
        chat.add_item(make_user_message("Say hi."))
        gen = unit.cancel_scope.generation
        unit.output_queue.put(AssistantOutputEvent(text="Hello there!", response_key="rk_z", cancel_generation=gen))
        _drain_until(ws, "response.output_audio_transcript.delta")

        _start_cough(unit, ws, "turn_z")
        assert [m for m in chat.buffer if m.role == "assistant"] == []
        unit.output_queue.put(AudioOutput(audio=_tone(RATE), response_key="rk_z", cancel_generation=gen))
        _finish_stream(unit, "rk_z", gen)
        _wait_for(lambda: unit.session.pending_false_interruption.audio_buffer.generation_done)

        _finish_cough(unit, ws, "turn_z")
        assert [(m.role, m.status, m.content[0].text) for m in chat.buffer] == [
            ("user", None, "Say hi."),
            ("assistant", "completed", "Hello there!"),
        ]

    # 2. Genuine repeated digit survives both silence-landed rewind AND shortening item_1 after replay.
    with _ws_session(min_words=1) as (unit, ws, sid):
        chat = unit.service._state(sid).runtime_config.chat
        chat.add_item(make_user_message("What is the code?"))
        _queue_reply(
            unit,
            sid,
            "rk_rep",
            [("The code is seven", _tone(RATE * 2)), ("seven nine.", _silence(int(0.2 * RATE)) + _tone(RATE * 2))],
        )
        item_1 = _added_id(_drain_until(ws, "response.done"))
        _cough_replay(unit, ws, "turn_rep", item_1, audio_end_ms=2050)
        assert chat.to_transformers_chat()[1]["content"] == "The code is seven seven nine."
        assert chat.to_responses_api_chat()[1]["content"][0]["text"] == "The code is seven seven nine."

    with _ws_session(min_words=1) as (unit, ws, sid):
        chat = unit.service._state(sid).runtime_config.chat
        chat.add_item(make_user_message("What is the code?"))
        _queue_reply(
            unit,
            sid,
            "rk_rep2",
            [("The code is seven", _tone(RATE * 2)), ("seven nine.", _tone(RATE * 2))],
        )
        item_1 = _added_id(_drain_until(ws, "response.done"))
        item_2, _ = _cough_replay(unit, ws, "turn_rep2", item_1, audio_end_ms=3000)
        assert chat.to_transformers_chat()[1]["content"] == "The code is seven seven nine."
        # Later playback correction shortens item_1 from 3s to 2s -> both sevens come from distinct audio.
        _truncate(ws, item_1, 2000)
        assert _retrieve(ws, item_1)["content"][0]["transcript"] == "The code is seven"
        assert _retrieve(ws, item_2)["content"][0]["transcript"] == "seven nine."
        assert chat.to_transformers_chat()[1]["content"] == "The code is seven seven nine."
        assert chat.to_responses_api_chat()[1]["content"][0]["text"] == "The code is seven seven nine."

    # 3. Replay hitting max_output_tokens (status="incomplete") stays stored and retrievable.
    with _ws_session(min_words=1) as (unit, ws, sid):
        chat = unit.service._state(sid).runtime_config.chat
        chat.add_item(make_user_message("Keep going."))
        gen = _start_stream(unit, ws, "rk_inc", "Opening line.")
        _start_cough(unit, ws, "turn_inc")
        unit.output_queue.put(
            AssistantOutputEvent(text="Partial ending.", response_key="rk_inc", cancel_generation=gen)
        )
        unit.output_queue.put(AudioOutput(audio=_tone(RATE * 2), response_key="rk_inc", cancel_generation=gen))
        _finish_stream(unit, "rk_inc", gen, status="incomplete", reason="max_output_tokens")
        _wait_for(lambda: unit.session.pending_false_interruption.audio_buffer.generation_done)

        resumed = _finish_cough(unit, ws, "turn_inc")
        assert resumed[-1]["response"]["status"] == "incomplete"
        item_2 = _added_id(resumed)
        got_2 = _retrieve(ws, item_2)
        assert (
            got_2["id"] == item_2
            and got_2["status"] == "incomplete"
            and "Partial ending." in got_2["content"][0]["transcript"]
        )
        assert "Partial ending." in chat.to_transformers_chat()[1]["content"]

    # 4. Second cough at the very end of replay leaves original item_1 intact and preserves overlap dedup.
    with _ws_session(min_words=1) as (unit, ws, sid):
        chat = unit.service._state(sid).runtime_config.chat
        chat.add_item(make_user_message("Three lines."))
        _queue_reply(
            unit,
            sid,
            "rk_end",
            [
                ("First sentence.", _tone(RATE * 2)),
                ("Second sentence.", _tone(RATE * 2)),
                ("Third sentence.", _tone(RATE * 2)),
            ],
        )
        item_1 = _added_id(_drain_until(ws, "response.done"))
        item_2, _ = _cough_replay(unit, ws, "turn_e1", item_1, audio_end_ms=2000)

        _start_cough(unit, ws, "turn_e2", item_id=item_2, total_played_ms=6000)
        _finish_cough(unit, ws, "turn_e2", drain="conversation.item.input_audio_transcription.completed")
        _wait_for(lambda: unit.session.pending_false_interruption is None)

        got_1, got_2 = _retrieve(ws, item_1), _retrieve(ws, item_2)
        assert got_1["status"] == "incomplete" and got_1["content"][0]["transcript"] == "First sentence."
        assert got_2["status"] == "completed" and "Third sentence." in got_2["content"][0]["transcript"]
        assert chat.to_transformers_chat()[1]["content"] == "First sentence. Second sentence. Third sentence."


@pytest.mark.asyncio
async def test_webrtc_and_websocket_estimate_replay_without_client_truncate() -> None:
    from speech_to_speech.api.openai_realtime.webrtc_session import WEBRTC_SAMPLE_RATE, WebRTCSession

    # 1. WebRTC buffered playback flushed on cough and replayed back into track on empty STT.
    unit = _make_unit(min_words=1)
    sid = unit.service.register()
    pc = Mock(sctp=None, close=AsyncMock())
    pc.getTransceivers.return_value = []
    rtc = WebRTCSession(pc, on_client_event=AsyncMock(), on_audio=Mock(), on_open=AsyncMock(), on_closed=Mock())
    unit.session = SessionState(transport=rtc, session_id=sid)
    chat = unit.service._state(sid).runtime_config.chat
    chat.add_item(make_user_message("Hi"))
    stop_event = ThreadingEvent()
    app = router_module.create_app(pool=[unit], stop_event=stop_event)
    async with app.router.lifespan_context(app):
        text = _queue_reply(unit, sid, "rk_rtc", [("Hello there!", _tone(RATE * 2))])
        await _await_for(
            lambda: bool(unit.session.last_assistant_audio and unit.session.last_assistant_audio.generation_done)
        )
        assert rtc._track.buffered_bytes >= WEBRTC_SAMPLE_RATE * 2 * 1.9

        unit.text_output_queue.put(SpeechStartedEvent(turn_id="turn_rtc", turn_revision=0))
        await _await_for(lambda: unit.session.pending_false_interruption is not None)
        assert rtc._track.buffered_bytes == 0

        unit.text_output_queue.put(SpeechStoppedEvent(turn_id="turn_rtc", turn_revision=0, duration_s=0.2))
        unit.text_output_queue.put(TranscriptionCompletedEvent(transcript="", turn_id="turn_rtc", turn_revision=0))
        await _await_for(rtc.has_pending_audio)
        assert rtc._track.buffered_bytes >= WEBRTC_SAMPLE_RATE * 2 * 1.9
        assert [(m.role, m.status, m.content[0].text) for m in chat.buffer] == [
            ("user", None, "Hi"),
            ("assistant", "completed", text),
        ]
        stop_event.set()
    await rtc.close()

    # 2. WebSocket client that never sends truncate still replays from server playback estimate.
    clock_now = [100.0]

    class _FakeWS:
        application_state = WebSocketState.CONNECTED
        sent: list[dict]

        def __init__(self) -> None:
            self.sent = []

        async def send_json(self, data: dict) -> None:
            self.sent.append(data)

        async def close(self) -> None:
            pass

    unit2 = _make_unit(min_words=1)
    sid2 = unit2.service.register()
    fake_ws = _FakeWS()
    unit2.session = SessionState(transport=WebSocketTransport(fake_ws, clock=lambda: clock_now[0]), session_id=sid2)
    stop_event2 = ThreadingEvent()
    app2 = router_module.create_app(pool=[unit2], stop_event=stop_event2)
    async with app2.router.lifespan_context(app2):
        _queue_reply(unit2, sid2, "rk_est", [("Part one. Part two.", _tone(RATE * 4))])
        await _await_for(
            lambda: bool(unit2.session.last_assistant_audio and unit2.session.last_assistant_audio.generation_done)
        )
        clock_now[0] += 2.0
        sent_before = len(fake_ws.sent)
        unit2.text_output_queue.put(SpeechStartedEvent(turn_id="turn_e", turn_revision=0))
        await _await_for(lambda: unit2.session.pending_false_interruption is not None)
        unit2.text_output_queue.put(SpeechStoppedEvent(turn_id="turn_e", turn_revision=0, duration_s=0.2))
        unit2.text_output_queue.put(TranscriptionCompletedEvent(transcript="", turn_id="turn_e", turn_revision=0))
        await _await_for(lambda: any(ev["type"] == "response.done" for ev in fake_ws.sent[sent_before:]))
        assert 1.95 <= _replayed_s(fake_ws.sent[sent_before:]) <= 2.65
        stop_event2.set()


def test_default_disabled_mid_reply_opt_in_and_per_session_config() -> None:
    # 1. Disabled by default: retains no audio and cancels immediately on speech start.
    with _ws_session(min_words=0) as (unit, ws, _sid):
        gen = _start_stream(unit, ws, "rk_off", "Default mode.", samples=RATE)
        assert unit.session.last_assistant_audio is None
        _start_cough(unit, ws, "turn_d")
        _wait_for(lambda: unit.cancel_scope.generation == gen + 1)
        assert unit.session.pending_false_interruption is None

    # 2. Mid-reply opt-in takes effect on the next response, not a partial half-recorded reply.
    with _ws_session(min_words=0) as (unit, ws, sid):
        gen = _start_stream(unit, ws, "rk_mid", "First two seconds.")
        ws.send_json(_td_update(min_interruption_words=1))
        _drain_until(ws, "session.updated")
        unit.output_queue.put(
            AssistantOutputEvent(text="Last two seconds.", response_key="rk_mid", cancel_generation=gen)
        )
        unit.output_queue.put(AudioOutput(audio=_tone(RATE * 2), response_key="rk_mid", cancel_generation=gen))
        _finish_stream(unit, "rk_mid", gen)
        _drain_until(ws, "response.done")
        assert unit.session.last_assistant_audio is None

        _queue_reply(unit, sid, "rk_next", [("Next full reply.", _tone(RATE * 2))])
        _drain_until(ws, "response.done")
        assert unit.session.last_assistant_audio is not None
        _start_cough(unit, ws, "turn_next")
        assert any(ev["type"] == "response.output_audio.delta" for ev in _finish_cough(unit, ws, "turn_next"))

    # 3. Per-session isolation and validation on one shared RealtimeService.
    unit = _make_unit(min_words=0, timeout_ms=2000)
    sid_a, sid_b = unit.service.register(), unit.service.register()
    ev_enable = unit.service.parse_client_event(
        _td_update(min_interruption_words=2, false_interruption_timeout_ms=1200)
    )
    assert ev_enable is not None and unit.service.handle_session_update(sid_a, ev_enable) is None
    assert (
        unit.service.session_min_interruption_words(sid_a) == 2
        and unit.service.session_min_interruption_words(sid_b) == 0
    )
    assert unit.service.session_false_interruption_timeout_s(sid_a) == pytest.approx(1.2)
    assert unit.service.session_false_interruption_timeout_s(sid_b) == pytest.approx(2.0)

    for bad in ({"min_interruption_words": -1}, {"min_interruption_words": True}, {"false_interruption_timeout_ms": 0}):
        bad_ev = unit.service.parse_client_event(_td_update(**bad))
        assert (
            bad_ev is not None
            and unit.service.handle_session_update(sid_a, bad_ev).error.type == "invalid_request_error"
        )

    # 4. Idle instructions-only update followed by opt-in enables recovery on the very first reply.
    with _ws_session(min_words=0) as (u, ws, sid):
        ws.send_json({"type": "session.update", "session": {"type": "realtime", "instructions": "Be concise."}})
        _drain_until(ws, "session.updated")
        ws.send_json(_td_update(min_interruption_words=1, false_interruption_timeout_ms=1500))
        assert (
            _drain_until(ws, "session.updated")[-1]["session"]["audio"]["input"]["turn_detection"][
                "min_interruption_words"
            ]
            == 1
        )

        _queue_reply(u, sid, "rk_optin", [("Opted in reply.", _tone(RATE * 2))])
        _drain_until(ws, "response.done")
        _start_cough(u, ws, "turn_opt")
        assert any(ev["type"] == "response.output_audio.delta" for ev in _finish_cough(u, ws, "turn_opt"))

    # 5. Disabling via session.update(min_interruption_words=0) mid-check cancels pending recovery.
    with _ws_session(min_words=1) as (u, ws, _sid):
        gen = _start_stream(u, ws, "rk_off_s", "Streaming.")
        _start_cough(u, ws, "turn_off_s")
        assert u.session.pending_false_interruption is not None
        ws.send_json(_td_update(min_interruption_words=0))
        _drain_until(ws, "session.updated")
        assert u.session.pending_false_interruption is None and u.cancel_scope.generation == gen + 1
