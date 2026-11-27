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
from speech_to_speech.LLM.chat import make_assistant_message, make_user_message
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


def _queue_completed_reply(unit, session_id: str, rk: str, segments: list[tuple[str, bytes]], *, turn=None) -> str:
    chat = unit.service._state(session_id).runtime_config.chat
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


# ── Unit tests: word counting, rewind, fade-in, buffer ─────────────────────


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
    # [0..1.0s loud] [1.0..1.1s gap] [1.1..1.4s loud], cut at 1.4s -> snaps inside [1.0..1.1s] gap.
    pcm = _tone(RATE) + _silence(int(0.1 * RATE)) + _tone(int(0.3 * RATE))
    assert RATE <= resume_rewind_samples(pcm, int(1.4 * RATE)) <= int(1.1 * RATE)
    # Trailing silence prefers the end of the silence (no dead air before resuming).
    assert resume_rewind_samples(_tone(RATE) + _silence(int(0.4 * RATE)), int(1.4 * RATE)) >= int(1.35 * RATE)

    faded = np.frombuffer(apply_fade_in(_tone(1000, amplitude=10000)), dtype=np.int16)
    assert faded[0] == 0 and 0 < faded[80] < 10000 and faded[200] == 10000


def test_audio_buffer_tracks_segments_unsynthesized_tail_and_cap() -> None:
    buf = AssistantAudioBuffer(response_key="rk1", root_item_id="item_1", current_item_id="item_1")
    buf.append_text("Hello world.")
    buf.append_pcm(_tone(RATE))
    buf.append_text("Second sentence.")
    buf.append_pcm(_tone(RATE))
    buf.append_text("Pending third.")  # No PCM yet.

    remaining = buf.remaining_segments_from(int(1.5 * RATE))
    assert [text for text, _ in remaining] == ["sentence.", "Pending third."]
    assert len(remaining[0][1]) == int(0.5 * RATE) * 2 and remaining[1][1] == b""

    buf.append_pcm(_silence(MAX_RETAINED_AUDIO_SAMPLES))
    assert len(buf.pcm) // 2 == MAX_RETAINED_AUDIO_SAMPLES and buf.pcm_base_sample == 2 * RATE


# ── End-to-end Realtime router tests ───────────────────────────────────────


def test_cough_during_completed_websocket_playback_replays_from_client_truncate() -> None:
    with _ws_session(min_words=1) as (unit, ws, sid):
        chat = unit.service._state(sid).runtime_config.chat
        chat.add_item(make_user_message("Explain photosynthesis."))
        full = _queue_completed_reply(
            unit,
            sid,
            "rk_short",
            [
                ("Plants convert sunlight into energy.", _tone(RATE * 2) + _silence(int(0.1 * RATE))),
                ("Chlorophyll absorbs light in the leaves.", _tone(RATE * 2)),
            ],
        )
        initial = _drain_until(ws, "response.done")
        item_1 = next(ev["item"]["id"] for ev in initial if ev["type"] == "response.output_item.added")

        unit.text_output_queue.put(SpeechStartedEvent(turn_id="turn_2", turn_revision=0))
        _drain_until(ws, "input_audio_buffer.speech_started")
        ws.send_json(
            {"type": "conversation.item.truncate", "item_id": item_1, "content_index": 0, "audio_end_ms": 2000}
        )
        _wait_for(
            lambda: (
                unit.session.pending_false_interruption is not None
                and unit.session.pending_false_interruption.client_played_samples == 2000 * RATE // 1000
            )
        )

        unit.text_output_queue.put(SpeechStoppedEvent(turn_id="turn_2", turn_revision=0, duration_s=0.3))
        unit.text_output_queue.put(TranscriptionCompletedEvent(transcript="", turn_id="turn_2", turn_revision=0))
        resumed = _drain_until(ws, "response.done")
        types = [ev["type"] for ev in resumed]
        assert types.index("conversation.item.input_audio_transcription.completed") < types.index("response.created")

        replayed_s = (
            len(
                b"".join(base64.b64decode(ev["delta"]) for ev in resumed if ev["type"] == "response.output_audio.delta")
            )
            / 2
            / RATE
        )
        assert 2.05 <= replayed_s <= 2.75
        # Stored history mirrors the two wire items; model serializers merge them without repeating rewound words.
        assert [(m.role, m.status) for m in chat.buffer] == [
            ("user", None),
            ("assistant", "incomplete"),
            ("assistant", "completed"),
        ]
        assert chat.to_transformers_chat() == [
            {"role": "user", "content": "Explain photosynthesis."},
            {"role": "assistant", "content": full},
        ]
        resp_items = chat.to_responses_api_chat()
        assert [item["role"] for item in resp_items] == ["user", "assistant"]
        assert resp_items[1]["status"] == "completed" and resp_items[1]["content"][0]["text"] == full


def test_cough_during_long_generating_turn_buffers_and_streams_seamlessly() -> None:
    with _ws_session(min_words=1, speculative=True) as (unit, ws, sid):
        tracker = unit.service.speculative_turns
        assert tracker is not None
        chat = unit.service._state(sid).runtime_config.chat
        chat.add_item(make_user_message("Tell me a story."))

        t1, r1 = tracker.start_turn()
        tracker.segment_finalized(1000, output_hold_ms=0)
        rk, gen = "rk_long", unit.cancel_scope.generation
        s1, s2, s3 = "Once upon a time.", "A robot learned to listen.", "And finished every sentence."
        hist1 = ResponseHistory.capture(chat, [make_assistant_message(s1)], after_item_id=chat.history_anchor_id())
        unit.output_queue.put(
            AssistantOutputEvent(
                text=s1,
                response_key=rk,
                turn_id=t1,
                turn_revision=r1,
                cancel_generation=gen,
                output_sequence=0,
                history=hist1,
            )
        )
        unit.output_queue.put(AudioOutput(audio=_tone(RATE * 2), cancel_generation=gen, response_key=rk))
        _drain_until(ws, "response.output_audio.delta")

        # Cough arrives mid-generation: wire cancels immediately, but pipeline stays alive privately.
        t2, r2 = tracker.start_turn()
        unit.text_output_queue.put(SpeechStartedEvent(turn_id=t2, turn_revision=r2))
        cancel_evs = _drain_until(ws, "input_audio_buffer.speech_started")
        assert any(ev["type"] == "response.done" and ev["response"]["status"] == "cancelled" for ev in cancel_evs)
        assert unit.cancel_scope.generation == gen and not unit.cancel_scope.discarding and tracker.is_latest(t1, r1)

        # Sentence 2 arrives from LLM/TTS while STT is still running -> buffered privately.
        hist2 = ResponseHistory.capture(
            chat, [make_assistant_message(f"{s1} {s2}")], after_item_id=chat.history_anchor_id()
        )
        unit.output_queue.put(
            AssistantOutputEvent(
                text=s2,
                response_key=rk,
                turn_id=t1,
                turn_revision=r1,
                cancel_generation=gen,
                output_sequence=1,
                history=hist2,
            )
        )
        unit.output_queue.put(AudioOutput(audio=_tone(RATE), cancel_generation=gen, response_key=rk))
        _wait_for(
            lambda: (
                unit.session.pending_false_interruption is not None
                and unit.session.pending_false_interruption.audio_buffer.total_samples == RATE * 3
            )
        )

        # Empty transcript -> replays s1 tail + s2 and stays open for live s3.
        tracker.segment_finalized(2000, output_hold_ms=800)
        unit.text_output_queue.put(SpeechStoppedEvent(turn_id=t2, turn_revision=r2, duration_s=0.25))
        unit.text_output_queue.put(TranscriptionCompletedEvent(transcript="", turn_id=t2, turn_revision=r2))
        _drain_until(ws, "response.output_audio.delta")
        assert unit.service._state(sid).in_response

        full = f"{s1} {s2} {s3}"
        hist3 = ResponseHistory.capture(
            chat, [make_assistant_message(full)], after_item_id=chat.history_anchor_id(), complete=True
        )
        unit.output_queue.put(
            AssistantOutputEvent(
                text=s3,
                response_key=rk,
                turn_id=t1,
                turn_revision=r1,
                cancel_generation=gen,
                output_sequence=2,
                history=hist3,
            )
        )
        unit.output_queue.put(AudioOutput(audio=_tone(RATE), cancel_generation=gen, response_key=rk))
        unit.output_queue.put(
            AssistantResponseDoneEvent(
                response_key=rk, turn_id=t1, turn_revision=r1, cancel_generation=gen, history=hist3
            )
        )
        unit.output_queue.put(AudioOutput(audio=AUDIO_RESPONSE_DONE, cancel_generation=gen, response_key=rk))

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
        gen = unit.cancel_scope.generation
        unit.output_queue.put(
            AssistantOutputEvent(
                text="Long reply.", response_key="rk1", turn_id=t1, turn_revision=r1, cancel_generation=gen
            )
        )
        unit.output_queue.put(AudioOutput(audio=_tone(RATE * 2), cancel_generation=gen, response_key="rk1"))
        _drain_until(ws, "response.output_audio.delta")

        t2, r2 = tracker.start_turn()
        unit.text_output_queue.put(SpeechStartedEvent(turn_id=t2, turn_revision=r2))
        _drain_until(ws, "input_audio_buffer.speech_started")
        assert unit.cancel_scope.generation == gen

        tracker.segment_finalized(1500, output_hold_ms=0)
        unit.text_output_queue.put(SpeechStoppedEvent(turn_id=t2, turn_revision=r2, duration_s=0.6))
        unit.text_output_queue.put(TranscriptionCompletedEvent(transcript="Stop please", turn_id=t2, turn_revision=r2))
        _wait_for(lambda: not unit.text_prompt_queue.empty())
        assert unit.cancel_scope.generation == gen + 1 and unit.session.pending_false_interruption is None
        assert any(m.role == "user" and m.content[0].text == "Stop please" for m in chat.buffer)

    # 2. 1-word fragment paused then reopened into 3 words confirms interruption instead of resuming.
    with _ws_session(min_words=2, speculative=True) as (unit, ws, sid):
        tracker = unit.service.speculative_turns
        assert tracker is not None
        chat = unit.service._state(sid).runtime_config.chat
        t1, r1 = tracker.start_turn()
        tracker.segment_finalized(500, output_hold_ms=0)
        gen = unit.cancel_scope.generation
        unit.output_queue.put(
            AssistantOutputEvent(
                text="Long answer.", response_key="rk2", turn_id=t1, turn_revision=r1, cancel_generation=gen
            )
        )
        unit.output_queue.put(AudioOutput(audio=_tone(RATE * 2), cancel_generation=gen, response_key="rk2"))
        _drain_until(ws, "response.output_audio.delta")

        t2, r2 = tracker.start_turn()
        unit.text_output_queue.put(SpeechStartedEvent(turn_id=t2, turn_revision=r2))
        _drain_until(ws, "input_audio_buffer.speech_started")
        tracker.segment_finalized(1000, output_hold_ms=1000)
        unit.text_output_queue.put(SpeechStoppedEvent(turn_id=t2, turn_revision=r2, duration_s=0.3))
        unit.text_output_queue.put(TranscriptionCompletedEvent(transcript="Wait", turn_id=t2, turn_revision=r2))
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
        unit.text_output_queue.put(SpeechStoppedEvent(turn_id=reopen_id, turn_revision=reopen_rev, duration_s=0.8))
        unit.text_output_queue.put(
            TranscriptionCompletedEvent(transcript="Wait stop that", turn_id=reopen_id, turn_revision=reopen_rev)
        )
        _wait_for(lambda: not unit.text_prompt_queue.empty())
        assert unit.cancel_scope.generation == gen + 1
        assert any(m.role == "user" and m.content[0].text == "Wait stop that" for m in chat.buffer)


def test_short_backchannel_and_stt_failure_resume_without_triggering_llm() -> None:
    # 1-word backchannel ("yeah") with min_interruption_words=2 resumes after reopen grace and skips LLM.
    with _ws_session(min_words=2, speculative=True) as (unit, ws, sid):
        tracker = unit.service.speculative_turns
        assert tracker is not None
        chat = unit.service._state(sid).runtime_config.chat
        chat.add_item(make_user_message("Go on."))
        t1, r1 = tracker.start_turn()
        tracker.segment_finalized(500, output_hold_ms=0)
        full = _queue_completed_reply(
            unit, sid, "rk_bc", [("First point. Second point.", _tone(RATE * 3))], turn=(t1, r1)
        )
        _drain_until(ws, "response.done")

        t2, r2 = tracker.start_turn()
        unit.text_output_queue.put(SpeechStartedEvent(turn_id=t2, turn_revision=r2))
        _drain_until(ws, "input_audio_buffer.speech_started")
        tracker.segment_finalized(1000, output_hold_ms=50)
        unit.text_output_queue.put(SpeechStoppedEvent(turn_id=t2, turn_revision=r2, duration_s=0.25))
        unit.text_output_queue.put(TranscriptionCompletedEvent(transcript="yeah", turn_id=t2, turn_revision=r2))

        resumed = _drain_until(ws, "response.done")
        assert any(ev["type"] == "response.output_audio.delta" for ev in resumed)
        assert unit.text_prompt_queue.empty()
        assert [(m.role, m.status, m.content[0].text) for m in chat.buffer] == [
            ("user", None, "Go on."),
            ("assistant", "completed", full),
        ]

    # TranscriptionFailedEvent immediately resumes playback.
    with _ws_session(min_words=1) as (unit, ws, sid):
        _queue_completed_reply(unit, sid, "rk_fail", [("Keep listening.", _tone(RATE * 3))])
        _drain_until(ws, "response.done")
        unit.text_output_queue.put(SpeechStartedEvent(turn_id="turn_f", turn_revision=0))
        _drain_until(ws, "input_audio_buffer.speech_started")
        unit.text_output_queue.put(SpeechStoppedEvent(turn_id="turn_f", turn_revision=0, duration_s=0.3))
        unit.text_output_queue.put(TranscriptionFailedEvent(message="stt err", turn_id="turn_f", turn_revision=0))
        assert any(ev["type"] == "response.output_audio.delta" for ev in _drain_until(ws, "response.done"))


def test_timeout_explicit_cancel_and_tool_call_recovery() -> None:
    # 1a. Continuing speech (never stops before timeout) confirms real interruption and cancels generation.
    with _ws_session(min_words=1, timeout_ms=80) as (unit, ws, _sid):
        gen = unit.cancel_scope.generation
        unit.output_queue.put(AssistantOutputEvent(text="Streaming...", response_key="rk_to", cancel_generation=gen))
        unit.output_queue.put(AudioOutput(audio=_tone(RATE * 2), response_key="rk_to", cancel_generation=gen))
        _drain_until(ws, "response.output_audio.delta")
        unit.text_output_queue.put(SpeechStartedEvent(turn_id="turn_to", turn_revision=0))
        _drain_until(ws, "input_audio_buffer.speech_started")
        _wait_for(lambda: unit.session.pending_false_interruption is None)
        assert unit.cancel_scope.generation == gen + 1

    # 1b. Speech stops, then STT stalls past grace timeout -> resumes instead of leaving assistant silent.
    with _ws_session(min_words=1, timeout_ms=80) as (unit, ws, sid):
        chat = unit.service._state(sid).runtime_config.chat
        chat.add_item(make_user_message("Ask me."))
        _queue_completed_reply(unit, sid, "rk_to_res", [("What is the capital of France?", _tone(RATE * 2))])
        _drain_until(ws, "response.done")

        unit.text_output_queue.put(SpeechStartedEvent(turn_id="turn_to_res", turn_revision=0))
        _drain_until(ws, "input_audio_buffer.speech_started")
        time.sleep(0.05)  # Under timeout_ms; stopping resets the grace period from speech stop.
        unit.text_output_queue.put(SpeechStoppedEvent(turn_id="turn_to_res", turn_revision=0, duration_s=0.05))
        replayed = _drain_until(ws, "response.done")
        assert any(ev["type"] == "response.output_audio.delta" for ev in replayed)
        assert [(m.role, m.status) for m in chat.buffer] == [("user", None), ("assistant", "completed")]

    # 1c. Grace period resets at speech_stopped, so a real transcript arriving after start+timeout still confirms.
    with _ws_session(min_words=1, timeout_ms=100) as (unit, ws, sid):
        gen = unit.cancel_scope.generation
        unit.output_queue.put(AssistantOutputEvent(text="Long answer.", response_key="rk_reset", cancel_generation=gen))
        unit.output_queue.put(AudioOutput(audio=_tone(RATE * 2), response_key="rk_reset", cancel_generation=gen))
        _drain_until(ws, "response.output_audio.delta")
        unit.text_output_queue.put(SpeechStartedEvent(turn_id="turn_reset", turn_revision=0))
        _drain_until(ws, "input_audio_buffer.speech_started")
        time.sleep(0.06)
        unit.text_output_queue.put(SpeechStoppedEvent(turn_id="turn_reset", turn_revision=0, duration_s=0.06))
        time.sleep(0.06)  # 120 ms since speech_started (> 100 ms), 60 ms since speech_stopped (< 100 ms).
        assert unit.session.pending_false_interruption is not None
        unit.text_output_queue.put(
            TranscriptionCompletedEvent(transcript="Stop please", turn_id="turn_reset", turn_revision=0)
        )
        _wait_for(lambda: unit.session.pending_false_interruption is None)
        assert unit.cancel_scope.generation == gen + 1

    # 2. Explicit client response.cancel aborts recovery so late empty STT does not replay.
    with _ws_session(min_words=1) as (unit, ws, _sid):
        gen = unit.cancel_scope.generation
        unit.output_queue.put(AssistantOutputEvent(text="Speaking.", response_key="rk_c", cancel_generation=gen))
        unit.output_queue.put(AudioOutput(audio=_tone(RATE * 2), response_key="rk_c", cancel_generation=gen))
        _drain_until(ws, "response.output_audio.delta")
        unit.text_output_queue.put(SpeechStartedEvent(turn_id="turn_c", turn_revision=0))
        _drain_until(ws, "input_audio_buffer.speech_started")
        ws.send_json({"type": "response.cancel"})
        _wait_for(lambda: unit.session.pending_false_interruption is None)
        assert unit.cancel_scope.generation == gen + 1
        unit.text_output_queue.put(SpeechStoppedEvent(turn_id="turn_c", turn_revision=0, duration_s=0.2))
        unit.text_output_queue.put(TranscriptionCompletedEvent(transcript="", turn_id="turn_c", turn_revision=0))
        evs = _drain_until(ws, "conversation.item.input_audio_transcription.completed")
        assert not any(ev["type"] == "response.created" for ev in evs)

    # 3a. Tool call arriving during a cough (both early side-channel and ordered) is held and delivered once on resume.
    with _ws_session(min_words=1, speculative=True) as (unit, ws, sid):
        chat = unit.service._state(sid).runtime_config.chat
        chat.add_item(make_user_message("Check my score."))
        gen = unit.cancel_scope.generation
        turns = unit.service.speculative_turns
        assert turns is not None
        t0, r0 = turns.start_turn()
        turns.segment_finalized(500, output_hold_ms=0)
        unit.output_queue.put(
            AssistantOutputEvent(
                text="Checking your score.",
                response_key="rk_tool",
                cancel_generation=gen,
                turn_id=t0,
                turn_revision=r0,
                output_sequence=0,
            )
        )
        unit.output_queue.put(AudioOutput(audio=_tone(RATE), response_key="rk_tool", cancel_generation=gen))
        _drain_until(ws, "response.output_audio.delta")

        t1, r1 = turns.start_turn()
        unit.text_output_queue.put(SpeechStartedEvent(turn_id=t1, turn_revision=r1))
        _drain_until(ws, "input_audio_buffer.speech_started")
        assert unit.session.pending_false_interruption is not None

        tool_part = AssistantToolCallPart(
            tool=ResponseFunctionToolCall(
                id="fc_score", type="function_call", call_id="call_score", name="get_score", arguments='{"user":"a"}'
            )
        )
        unit.text_output_queue.put(
            AssistantToolCallReadyEvent(
                part=tool_part,
                output_sequence=1,
                turn_id=t0,
                turn_revision=r0,
                cancel_generation=gen,
                response_key="rk_tool",
            )
        )
        unit.output_queue.put(
            AssistantOutputEvent(
                parts=[tool_part],
                output_sequence=1,
                turn_id=t0,
                turn_revision=r0,
                cancel_generation=gen,
                response_key="rk_tool",
            )
        )
        unit.output_queue.put(
            AssistantResponseDoneEvent(response_key="rk_tool", cancel_generation=gen, turn_id=t0, turn_revision=r0)
        )
        unit.output_queue.put(AudioOutput(audio=AUDIO_RESPONSE_DONE, response_key="rk_tool", cancel_generation=gen))
        _wait_for(lambda: unit.session.pending_false_interruption.audio_buffer.generation_done)
        assert unit.cancel_scope.generation == gen  # Not cancelled while STT is pending.

        turns.segment_finalized(1000, output_hold_ms=0)
        unit.text_output_queue.put(SpeechStoppedEvent(turn_id=t1, turn_revision=r1, duration_s=0.2))
        unit.text_output_queue.put(TranscriptionCompletedEvent(transcript="", turn_id=t1, turn_revision=r1))
        resumed = _drain_until(ws, "response.done")
        fn_done = [ev for ev in resumed if ev["type"] == "response.function_call_arguments.done"]
        assert len(fn_done) == 1
        assert fn_done[0]["call_id"] == "call_score" and fn_done[0]["name"] == "get_score"
        assert any(getattr(item, "call_id", None) == "call_score" for item in chat.buffer)

        # Follow-up function_call_output succeeds against the recovered call in Chat.
        ws.send_json(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": "call_score", "output": '{"score": 10}'},
            }
        )
        created = _drain_until(ws, "conversation.item.created")[-1]
        assert created["item"]["call_id"] == "call_score"
        # Serialized history folds any split assistant speech and the tool call into one valid assistant turn.
        tf_msgs = chat.to_transformers_chat()
        assert [m["role"] for m in tf_msgs] == ["user", "assistant", "tool"]
        assert tf_msgs[1]["content"] == "Checking your score."
        assert [tc["function"]["name"] for tc in tf_msgs[1]["tool_calls"]] == ["get_score"]
        resp_msgs = chat.to_responses_api_chat()
        assert [m["type"] for m in resp_msgs] == ["message", "message", "function_call", "function_call_output"]
        assert resp_msgs[1]["content"][0]["text"] == "Checking your score."

    # 3b. Tool call arriving during a real interruption is discarded and never delivered.
    with _ws_session(min_words=1) as (unit, ws, sid):
        chat = unit.service._state(sid).runtime_config.chat
        gen = unit.cancel_scope.generation
        unit.output_queue.put(
            AssistantOutputEvent(text="Checking...", response_key="rk_td", cancel_generation=gen, output_sequence=0)
        )
        unit.output_queue.put(AudioOutput(audio=_tone(RATE), response_key="rk_td", cancel_generation=gen))
        _drain_until(ws, "response.output_audio.delta")

        unit.text_output_queue.put(SpeechStartedEvent(turn_id="turn_td", turn_revision=0))
        _drain_until(ws, "input_audio_buffer.speech_started")
        tool_part = AssistantToolCallPart(
            tool=ResponseFunctionToolCall(
                id="fc_drop", type="function_call", call_id="call_drop", name="search", arguments="{}"
            )
        )
        unit.output_queue.put(
            AssistantOutputEvent(parts=[tool_part], output_sequence=1, response_key="rk_td", cancel_generation=gen)
        )
        _wait_for(lambda: len(unit.session.pending_false_interruption.audio_buffer.buffered_tool_events) == 1)

        unit.text_output_queue.put(SpeechStoppedEvent(turn_id="turn_td", turn_revision=0, duration_s=0.4))
        unit.text_output_queue.put(
            TranscriptionCompletedEvent(transcript="Never mind", turn_id="turn_td", turn_revision=0)
        )
        evs = _drain_until(ws, "conversation.item.input_audio_transcription.completed")
        assert not any(ev["type"] == "response.function_call_arguments.done" for ev in evs)
        assert not any(getattr(item, "call_id", None) == "call_drop" for item in chat.buffer)
        assert unit.cancel_scope.generation == gen + 1


def test_real_interruption_during_resumed_reply_and_zero_audio_cough_history() -> None:
    # 1. Real interruption during item_2 updates root item_1 to combined heard prefix and keeps item_2 retrievable.
    with _ws_session(min_words=1) as (unit, ws, sid):
        chat = unit.service._state(sid).runtime_config.chat
        chat.add_item(make_user_message("List three fruits."))
        _queue_completed_reply(
            unit,
            sid,
            "rk_chain",
            [
                ("Apple is sweet.", _tone(RATE * 2)),
                ("Banana is yellow.", _tone(RATE * 2)),
                ("Cherry is red.", _tone(RATE * 2)),
            ],
        )
        item_1 = next(
            ev["item"]["id"] for ev in _drain_until(ws, "response.done") if ev["type"] == "response.output_item.added"
        )

        unit.text_output_queue.put(SpeechStartedEvent(turn_id="turn_c1", turn_revision=0))
        _drain_until(ws, "input_audio_buffer.speech_started")
        ws.send_json(
            {"type": "conversation.item.truncate", "item_id": item_1, "content_index": 0, "audio_end_ms": 2000}
        )
        _wait_for(lambda: unit.session.pending_false_interruption.client_played_samples is not None)
        unit.text_output_queue.put(SpeechStoppedEvent(turn_id="turn_c1", turn_revision=0, duration_s=0.2))
        unit.text_output_queue.put(TranscriptionCompletedEvent(transcript="", turn_id="turn_c1", turn_revision=0))
        item_2 = next(
            ev["item"]["id"] for ev in _drain_until(ws, "response.done") if ev["type"] == "response.output_item.added"
        )
        assert item_2 != item_1

        ws.send_json(
            {"type": "conversation.item.truncate", "item_id": item_2, "content_index": 0, "audio_end_ms": 2000}
        )
        ws.send_json({"type": "conversation.item.retrieve", "item_id": item_2})
        retrieved = _drain_until(ws, "conversation.item.retrieved")[-1]
        assert retrieved["item"]["id"] == item_2 and retrieved["item"]["status"] == "incomplete"

        assert [(m.role, m.status) for m in chat.buffer] == [
            ("user", None),
            ("assistant", "incomplete"),
            ("assistant", "incomplete"),
        ]
        tf_msgs = chat.to_transformers_chat()
        assert [m["role"] for m in tf_msgs] == ["user", "assistant"]
        assert "Apple is sweet." in tf_msgs[1]["content"] and "Cherry" not in tf_msgs[1]["content"]
        assert tf_msgs[1]["content"].count("sweet.") == 1

    # 2. Cough after AssistantOutputEvent but before first PCM chunk still restores completed history.
    with _ws_session(min_words=1) as (unit, ws, sid):
        chat = unit.service._state(sid).runtime_config.chat
        chat.add_item(make_user_message("Say hi."))
        gen = unit.cancel_scope.generation
        unit.output_queue.put(AssistantOutputEvent(text="Hello there!", response_key="rk_z", cancel_generation=gen))
        _drain_until(ws, "response.output_audio_transcript.delta")

        unit.text_output_queue.put(SpeechStartedEvent(turn_id="turn_z", turn_revision=0))
        _drain_until(ws, "input_audio_buffer.speech_started")
        assert [m for m in chat.buffer if m.role == "assistant"] == []

        unit.output_queue.put(AudioOutput(audio=_tone(RATE), response_key="rk_z", cancel_generation=gen))
        unit.output_queue.put(AssistantResponseDoneEvent(response_key="rk_z", cancel_generation=gen))
        unit.output_queue.put(AudioOutput(audio=AUDIO_RESPONSE_DONE, response_key="rk_z", cancel_generation=gen))
        _wait_for(lambda: unit.session.pending_false_interruption.audio_buffer.generation_done)

        unit.text_output_queue.put(SpeechStoppedEvent(turn_id="turn_z", turn_revision=0, duration_s=0.2))
        unit.text_output_queue.put(TranscriptionCompletedEvent(transcript="", turn_id="turn_z", turn_revision=0))
        _drain_until(ws, "response.done")
        assert [(m.role, m.status, m.content[0].text) for m in chat.buffer] == [
            ("user", None, "Say hi."),
            ("assistant", "completed", "Hello there!"),
        ]


@pytest.mark.asyncio
async def test_webrtc_and_websocket_estimate_replay_without_client_truncate() -> None:
    from speech_to_speech.api.openai_realtime.webrtc_session import WEBRTC_SAMPLE_RATE, WebRTCSession

    # 1. WebRTC buffered playback flushed on cough and replayed back into track on empty STT.
    unit = _make_unit(min_words=1)
    sid = unit.service.register()
    pc = Mock(sctp=None)
    pc.getTransceivers.return_value = []
    pc.close = AsyncMock()
    rtc = WebRTCSession(pc, on_client_event=AsyncMock(), on_audio=Mock(), on_open=AsyncMock(), on_closed=Mock())
    unit.session = SessionState(transport=rtc, session_id=sid)
    chat = unit.service._state(sid).runtime_config.chat
    chat.add_item(make_user_message("Hi"))
    stop_event = ThreadingEvent()
    app = router_module.create_app(pool=[unit], stop_event=stop_event)
    async with app.router.lifespan_context(app):
        text = _queue_completed_reply(unit, sid, "rk_rtc", [("Hello there!", _tone(RATE * 2))])
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not (
            unit.session.last_assistant_audio and unit.session.last_assistant_audio.generation_done
        ):
            await asyncio.sleep(0.01)
        assert rtc._track.buffered_bytes >= WEBRTC_SAMPLE_RATE * 2 * 1.9

        unit.text_output_queue.put(SpeechStartedEvent(turn_id="turn_rtc", turn_revision=0))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and unit.session.pending_false_interruption is None:
            await asyncio.sleep(0.01)
        assert rtc._track.buffered_bytes == 0

        unit.text_output_queue.put(SpeechStoppedEvent(turn_id="turn_rtc", turn_revision=0, duration_s=0.2))
        unit.text_output_queue.put(TranscriptionCompletedEvent(transcript="", turn_id="turn_rtc", turn_revision=0))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not rtc.has_pending_audio():
            await asyncio.sleep(0.01)
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

        def __init__(self) -> None:
            self.sent: list[dict] = []

        async def send_json(self, data: dict) -> None:
            self.sent.append(data)

        async def close(self) -> None:
            pass

    unit2 = _make_unit(min_words=1)
    sid2 = unit2.service.register()
    fake_ws = _FakeWS()
    transport = WebSocketTransport(fake_ws, clock=lambda: clock_now[0])
    unit2.session = SessionState(transport=transport, session_id=sid2)
    stop_event2 = ThreadingEvent()
    app2 = router_module.create_app(pool=[unit2], stop_event=stop_event2)
    async with app2.router.lifespan_context(app2):
        _queue_completed_reply(unit2, sid2, "rk_est", [("Part one. Part two.", _tone(RATE * 4))])
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not (
            unit2.session.last_assistant_audio and unit2.session.last_assistant_audio.generation_done
        ):
            await asyncio.sleep(0.01)
        clock_now[0] += 2.0
        sent_before = len(fake_ws.sent)
        unit2.text_output_queue.put(SpeechStartedEvent(turn_id="turn_e", turn_revision=0))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and unit2.session.pending_false_interruption is None:
            await asyncio.sleep(0.01)
        unit2.text_output_queue.put(SpeechStoppedEvent(turn_id="turn_e", turn_revision=0, duration_s=0.2))
        unit2.text_output_queue.put(TranscriptionCompletedEvent(transcript="", turn_id="turn_e", turn_revision=0))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not any(
            ev["type"] == "response.done" for ev in fake_ws.sent[sent_before:]
        ):
            await asyncio.sleep(0.01)
        replayed_s = (
            len(
                b"".join(
                    base64.b64decode(ev["delta"])
                    for ev in fake_ws.sent[sent_before:]
                    if ev["type"] == "response.output_audio.delta"
                )
            )
            / 2
            / RATE
        )
        assert 1.95 <= replayed_s <= 2.65
        stop_event2.set()


def test_disabled_by_default_retains_no_audio_and_cancels_immediately() -> None:
    with _ws_session(min_words=0) as (unit, ws, _sid):
        gen = unit.cancel_scope.generation
        unit.output_queue.put(AssistantOutputEvent(text="Default mode.", response_key="rk_off"))
        unit.output_queue.put(AudioOutput(audio=_tone(RATE), response_key="rk_off"))
        _drain_until(ws, "response.output_audio.delta")
        assert unit.session.last_assistant_audio is None

        unit.text_output_queue.put(SpeechStartedEvent(turn_id="turn_d", turn_revision=0))
        _drain_until(ws, "input_audio_buffer.speech_started")
        _wait_for(lambda: unit.cancel_scope.generation == gen + 1)
        assert unit.session.pending_false_interruption is None


def test_wire_item_retrieve_preserves_original_cut_and_delete_works_on_replay_item() -> None:
    with _ws_session(min_words=1) as (unit, ws, sid):
        chat = unit.service._state(sid).runtime_config.chat
        chat.add_item(make_user_message("Tell me two sentences."))
        _queue_completed_reply(
            unit,
            sid,
            "rk_ret",
            [("First sentence.", _tone(RATE * 2)), ("Last sentence.", _tone(RATE * 2))],
        )
        item_1 = next(
            ev["item"]["id"] for ev in _drain_until(ws, "response.done") if ev["type"] == "response.output_item.added"
        )

        # Cough at 2s; truncate item_1 and verify its wire view before replay.
        unit.text_output_queue.put(SpeechStartedEvent(turn_id="turn_ret", turn_revision=0))
        _drain_until(ws, "input_audio_buffer.speech_started")
        ws.send_json(
            {"type": "conversation.item.truncate", "item_id": item_1, "content_index": 0, "audio_end_ms": 2000}
        )
        ws.send_json({"type": "conversation.item.retrieve", "item_id": item_1})
        before = _drain_until(ws, "conversation.item.retrieved")[-1]["item"]
        assert before["status"] == "incomplete"
        assert before["content"][0]["transcript"] == "First sentence."

        # Replay completes as item_2.
        unit.text_output_queue.put(SpeechStoppedEvent(turn_id="turn_ret", turn_revision=0, duration_s=0.2))
        unit.text_output_queue.put(TranscriptionCompletedEvent(transcript="", turn_id="turn_ret", turn_revision=0))
        item_2 = next(
            ev["item"]["id"] for ev in _drain_until(ws, "response.done") if ev["type"] == "response.output_item.added"
        )
        assert item_2 != item_1

        # Stored history mirrors the two wire items; model serializers merge them with rewound overlap deduplicated.
        assert [(m.role, m.status) for m in chat.buffer] == [
            ("user", None),
            ("assistant", "incomplete"),
            ("assistant", "completed"),
        ]
        assert chat.to_transformers_chat() == [
            {"role": "user", "content": "Tell me two sentences."},
            {"role": "assistant", "content": "First sentence. Last sentence."},
        ]
        resp_msgs = chat.to_responses_api_chat()
        assert [m["role"] for m in resp_msgs] == ["user", "assistant"]
        assert resp_msgs[1]["status"] == "completed"
        assert resp_msgs[1]["content"][0]["text"] == "First sentence. Last sentence."
        ws.send_json({"type": "conversation.item.retrieve", "item_id": item_1})
        after_1 = _drain_until(ws, "conversation.item.retrieved")[-1]["item"]
        assert after_1["status"] == "incomplete"
        assert after_1["content"][0]["transcript"] == "First sentence."

        ws.send_json({"type": "conversation.item.retrieve", "item_id": item_2})
        after_2 = _drain_until(ws, "conversation.item.retrieved")[-1]["item"]
        assert after_2["status"] == "completed"
        assert "Last sentence." in after_2["content"][0]["transcript"]

        # Deleting item_2 removes only item_2, leaving item_1's heard prefix in Chat and removing item_2 from retrieval.
        ws.send_json({"type": "conversation.item.delete", "item_id": item_2})
        deleted = _drain_until(ws, "conversation.item.deleted")[-1]
        assert deleted["item_id"] == item_2
        assert [(m.role, m.status, m.content[0].text) for m in chat.buffer] == [
            ("user", None, "Tell me two sentences."),
            ("assistant", "incomplete", "First sentence."),
        ]
        ws.send_json({"type": "conversation.item.retrieve", "item_id": item_2})
        err = _drain_until(ws, "error")[-1]
        assert err["error"]["type"] == "conversation_item_not_found"
        ws.send_json({"type": "conversation.item.retrieve", "item_id": item_1})
        still_1 = _drain_until(ws, "conversation.item.retrieved")[-1]["item"]
        assert still_1["status"] == "incomplete" and still_1["content"][0]["transcript"] == "First sentence."


def test_late_backchannel_after_timeout_recovery_is_suppressed() -> None:
    with _ws_session(min_words=2, timeout_ms=80) as (unit, ws, sid):
        chat = unit.service._state(sid).runtime_config.chat
        chat.add_item(make_user_message("Explain."))
        _queue_completed_reply(unit, sid, "rk_late", [("First part. Second part.", _tone(RATE * 2))])
        _drain_until(ws, "response.done")

        unit.text_output_queue.put(SpeechStartedEvent(turn_id="turn_late", turn_revision=0))
        _drain_until(ws, "input_audio_buffer.speech_started")
        unit.text_output_queue.put(SpeechStoppedEvent(turn_id="turn_late", turn_revision=0, duration_s=0.1))
        # Grace period expires before STT returns; reply resumes.
        replayed = _drain_until(ws, "response.done")
        assert any(ev["type"] == "response.output_audio.delta" for ev in replayed)

        # Delayed 1-word backchannel ("yeah" < min_words=2) for that same turn must not enter Chat or queue LLM.
        unit.text_output_queue.put(TranscriptionCompletedEvent(transcript="yeah", turn_id="turn_late", turn_revision=0))
        _drain_until(ws, "conversation.item.input_audio_transcription.completed")
        assert unit.text_prompt_queue.empty()
        assert [(m.role, m.status) for m in chat.buffer] == [("user", None), ("assistant", "completed")]


def test_double_recovery_retrieval_and_deletion_history_reconciliation() -> None:
    # 1. Two recoveries on the same reply: all 3 wire items stay retrievable; deleting item_3 drops only "Last sentence.".
    with _ws_session(min_words=1) as (unit, ws, sid):
        chat = unit.service._state(sid).runtime_config.chat
        chat.add_item(make_user_message("Three sentences."))
        _queue_completed_reply(
            unit,
            sid,
            "rk_dbl",
            [
                ("First sentence.", _tone(RATE * 2)),
                ("Middle sentence.", _tone(RATE * 2)),
                ("Last sentence.", _tone(RATE * 2)),
            ],
        )
        item_1 = next(
            ev["item"]["id"] for ev in _drain_until(ws, "response.done") if ev["type"] == "response.output_item.added"
        )

        # First cough at 2s -> item_2 replays from ~1.4s..6s.
        unit.text_output_queue.put(SpeechStartedEvent(turn_id="turn_d1", turn_revision=0))
        _drain_until(ws, "input_audio_buffer.speech_started")
        ws.send_json(
            {"type": "conversation.item.truncate", "item_id": item_1, "content_index": 0, "audio_end_ms": 2000}
        )
        _wait_for(lambda: unit.session.pending_false_interruption.client_played_samples is not None)
        unit.text_output_queue.put(SpeechStoppedEvent(turn_id="turn_d1", turn_revision=0, duration_s=0.2))
        unit.text_output_queue.put(TranscriptionCompletedEvent(transcript="", turn_id="turn_d1", turn_revision=0))
        item_2 = next(
            ev["item"]["id"] for ev in _drain_until(ws, "response.done") if ev["type"] == "response.output_item.added"
        )

        # Second cough during item_2 -> truncate item_2 after "Middle sentence." (total played 4s).
        unit.text_output_queue.put(SpeechStartedEvent(turn_id="turn_d2", turn_revision=0))
        _drain_until(ws, "input_audio_buffer.speech_started")
        wire_start_ms = unit.session.pending_false_interruption.audio_buffer.wire_start_sample * 1000 // RATE
        ws.send_json(
            {
                "type": "conversation.item.truncate",
                "item_id": item_2,
                "content_index": 0,
                "audio_end_ms": 4000 - wire_start_ms,
            }
        )
        _wait_for(lambda: unit.session.pending_false_interruption.client_played_samples is not None)
        unit.text_output_queue.put(SpeechStoppedEvent(turn_id="turn_d2", turn_revision=0, duration_s=0.2))
        unit.text_output_queue.put(TranscriptionCompletedEvent(transcript="", turn_id="turn_d2", turn_revision=0))
        item_3 = next(
            ev["item"]["id"] for ev in _drain_until(ws, "response.done") if ev["type"] == "response.output_item.added"
        )
        assert len({item_1, item_2, item_3}) == 3

        # All three wire items are retrievable with their own views.
        for iid, expected_status in ((item_1, "incomplete"), (item_2, "incomplete"), (item_3, "completed")):
            ws.send_json({"type": "conversation.item.retrieve", "item_id": iid})
            got = _drain_until(ws, "conversation.item.retrieved")[-1]["item"]
            assert got["id"] == iid and got["status"] == expected_status

        # Deleting newest replay (item_3) removes "Last sentence." from model history while keeping first + middle.
        ws.send_json({"type": "conversation.item.delete", "item_id": item_3})
        assert _drain_until(ws, "conversation.item.deleted")[-1]["item_id"] == item_3
        assert [(m.role, m.status) for m in chat.buffer] == [
            ("user", None),
            ("assistant", "incomplete"),
            ("assistant", "incomplete"),
        ]
        tf_after = chat.to_transformers_chat()
        assert [m["role"] for m in tf_after] == ["user", "assistant"]
        text_after = tf_after[1]["content"]
        assert (
            "First sentence." in text_after
            and "Middle sentence." in text_after
            and "Last sentence." not in text_after
            and text_after.count("sentence.") == 2
        )
        resp_after = chat.to_responses_api_chat()
        assert [m["role"] for m in resp_after] == ["user", "assistant"] and resp_after[1]["status"] == "incomplete"
        assert resp_after[1]["content"][0]["text"] == text_after

    # 2. Deleting original item_1 after one replay keeps surviving item_2 in model history at the same position.
    with _ws_session(min_words=1) as (unit, ws, sid):
        chat = unit.service._state(sid).runtime_config.chat
        chat.add_item(make_user_message("Two sentences."))
        _queue_completed_reply(
            unit,
            sid,
            "rk_del_root",
            [("First sentence.", _tone(RATE * 2)), ("Last sentence.", _tone(RATE * 2))],
        )
        item_1 = next(
            ev["item"]["id"] for ev in _drain_until(ws, "response.done") if ev["type"] == "response.output_item.added"
        )

        unit.text_output_queue.put(SpeechStartedEvent(turn_id="turn_dr", turn_revision=0))
        _drain_until(ws, "input_audio_buffer.speech_started")
        ws.send_json(
            {"type": "conversation.item.truncate", "item_id": item_1, "content_index": 0, "audio_end_ms": 2000}
        )
        _wait_for(lambda: unit.session.pending_false_interruption.client_played_samples is not None)
        unit.text_output_queue.put(SpeechStoppedEvent(turn_id="turn_dr", turn_revision=0, duration_s=0.2))
        unit.text_output_queue.put(TranscriptionCompletedEvent(transcript="", turn_id="turn_dr", turn_revision=0))
        item_2 = next(
            ev["item"]["id"] for ev in _drain_until(ws, "response.done") if ev["type"] == "response.output_item.added"
        )

        # Add a later user turn so we also verify in-place chronological order preservation.
        chat.add_item(make_user_message("Follow up."))

        ws.send_json({"type": "conversation.item.delete", "item_id": item_1})
        assert _drain_until(ws, "conversation.item.deleted")[-1]["item_id"] == item_1

        # item_1 is gone, item_2 survives in model history (before "Follow up.") and remains retrievable & deletable.
        assert [(m.role, m.status) for m in chat.buffer] == [
            ("user", None),
            ("assistant", "completed"),
            ("user", None),
        ]
        assert chat.get_item(item_1) is None and chat.get_item(item_2) is not None
        assert "Last sentence." in chat.buffer[1].content[0].text
        assert "First" not in chat.buffer[1].content[0].text

        ws.send_json({"type": "conversation.item.retrieve", "item_id": item_1})
        assert _drain_until(ws, "error")[-1]["error"]["type"] == "conversation_item_not_found"
        ws.send_json({"type": "conversation.item.retrieve", "item_id": item_2})
        got_2 = _drain_until(ws, "conversation.item.retrieved")[-1]["item"]
        assert (
            got_2["id"] == item_2
            and got_2["status"] == "completed"
            and "Last sentence." in got_2["content"][0]["transcript"]
        )

        ws.send_json({"type": "conversation.item.delete", "item_id": item_2})
        assert _drain_until(ws, "conversation.item.deleted")[-1]["item_id"] == item_2
        assert [m for m in chat.buffer if m.role == "assistant"] == []

    # 3. After deleting item_1 in a 2-recovery chain, truncating item_3 must NOT restore deleted "First".
    with _ws_session(min_words=1) as (unit, ws, sid):
        chat = unit.service._state(sid).runtime_config.chat
        chat.add_item(make_user_message("Three parts."))
        _queue_completed_reply(
            unit,
            sid,
            "rk_del_trunc",
            [
                ("First sentence.", _tone(RATE * 2)),
                ("Second sentence.", _tone(RATE * 2)),
                ("Third sentence.", _tone(RATE * 2)),
            ],
        )
        item_1 = next(
            ev["item"]["id"] for ev in _drain_until(ws, "response.done") if ev["type"] == "response.output_item.added"
        )

        # First recovery at 2s (item_2), second recovery at 4s (item_3).
        unit.text_output_queue.put(SpeechStartedEvent(turn_id="turn_dt1", turn_revision=0))
        _drain_until(ws, "input_audio_buffer.speech_started")
        ws.send_json(
            {"type": "conversation.item.truncate", "item_id": item_1, "content_index": 0, "audio_end_ms": 2000}
        )
        _wait_for(lambda: unit.session.pending_false_interruption.client_played_samples is not None)
        unit.text_output_queue.put(SpeechStoppedEvent(turn_id="turn_dt1", turn_revision=0, duration_s=0.2))
        unit.text_output_queue.put(TranscriptionCompletedEvent(transcript="", turn_id="turn_dt1", turn_revision=0))
        item_2 = next(
            ev["item"]["id"] for ev in _drain_until(ws, "response.done") if ev["type"] == "response.output_item.added"
        )

        unit.text_output_queue.put(SpeechStartedEvent(turn_id="turn_dt2", turn_revision=0))
        _drain_until(ws, "input_audio_buffer.speech_started")
        wire_start_ms = unit.session.pending_false_interruption.audio_buffer.wire_start_sample * 1000 // RATE
        ws.send_json(
            {
                "type": "conversation.item.truncate",
                "item_id": item_2,
                "content_index": 0,
                "audio_end_ms": 4000 - wire_start_ms,
            }
        )
        _wait_for(lambda: unit.session.pending_false_interruption.client_played_samples is not None)
        unit.text_output_queue.put(SpeechStoppedEvent(turn_id="turn_dt2", turn_revision=0, duration_s=0.2))
        unit.text_output_queue.put(TranscriptionCompletedEvent(transcript="", turn_id="turn_dt2", turn_revision=0))
        item_3 = next(
            ev["item"]["id"] for ev in _drain_until(ws, "response.done") if ev["type"] == "response.output_item.added"
        )

        # Delete item_1 -> "First" disappears from Chat and serialized model history.
        ws.send_json({"type": "conversation.item.delete", "item_id": item_1})
        assert _drain_until(ws, "conversation.item.deleted")[-1]["item_id"] == item_1
        assert [(m.role, m.status) for m in chat.buffer] == [
            ("user", None),
            ("assistant", "incomplete"),
            ("assistant", "completed"),
        ]
        tf_before_trunc = chat.to_transformers_chat()
        assert [m["role"] for m in tf_before_trunc] == ["user", "assistant"]
        assert "First" not in tf_before_trunc[1]["content"] and "Second sentence." in tf_before_trunc[1]["content"]

        # Truncating surviving item_3 must NOT restore deleted "First".
        ws.send_json(
            {"type": "conversation.item.truncate", "item_id": item_3, "content_index": 0, "audio_end_ms": 1000}
        )
        ws.send_json({"type": "conversation.item.retrieve", "item_id": item_3})
        got_3 = _drain_until(ws, "conversation.item.retrieved")[-1]["item"]
        assert got_3["status"] == "incomplete"
        tf_after_trunc = chat.to_transformers_chat()
        assert [m["role"] for m in tf_after_trunc] == ["user", "assistant"]
        assert "First" not in tf_after_trunc[1]["content"] and "Second sentence." in tf_after_trunc[1]["content"]
        resp_after_trunc = chat.to_responses_api_chat()
        assert [m["role"] for m in resp_after_trunc] == ["user", "assistant"]
        assert (
            resp_after_trunc[1]["status"] == "incomplete" and "First" not in resp_after_trunc[1]["content"][0]["text"]
        )


def test_mid_reply_opt_in_takes_effect_on_next_response() -> None:
    with _ws_session(min_words=0) as (unit, ws, sid):
        chat = unit.service._state(sid).runtime_config.chat
        chat.add_item(make_user_message("Speak."))
        gen = unit.cancel_scope.generation
        unit.output_queue.put(
            AssistantOutputEvent(text="First two seconds.", response_key="rk_mid", cancel_generation=gen)
        )
        unit.output_queue.put(AudioOutput(audio=_tone(RATE * 2), response_key="rk_mid", cancel_generation=gen))
        _drain_until(ws, "response.output_audio.delta")
        assert unit.session.last_assistant_audio is None

        # Enable recovery mid-reply; current partial reply must not record a misaligned half-buffer.
        ws.send_json(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "audio": {"input": {"turn_detection": {"type": "server_vad", "min_interruption_words": 1}}},
                },
            }
        )
        _drain_until(ws, "session.updated")

        unit.output_queue.put(
            AssistantOutputEvent(text="Last two seconds.", response_key="rk_mid", cancel_generation=gen)
        )
        unit.output_queue.put(AudioOutput(audio=_tone(RATE * 2), response_key="rk_mid", cancel_generation=gen))
        unit.output_queue.put(AssistantResponseDoneEvent(response_key="rk_mid", cancel_generation=gen))
        unit.output_queue.put(AudioOutput(audio=AUDIO_RESPONSE_DONE, response_key="rk_mid", cancel_generation=gen))
        _drain_until(ws, "response.done")
        assert unit.session.last_assistant_audio is None

        # The next response starts with recovery active from sample 0 and replays cleanly.
        _queue_completed_reply(unit, sid, "rk_next", [("Next full reply.", _tone(RATE * 2))])
        _drain_until(ws, "response.done")
        assert unit.session.last_assistant_audio is not None
        unit.text_output_queue.put(SpeechStartedEvent(turn_id="turn_next", turn_revision=0))
        _drain_until(ws, "input_audio_buffer.speech_started")
        unit.text_output_queue.put(SpeechStoppedEvent(turn_id="turn_next", turn_revision=0, duration_s=0.2))
        unit.text_output_queue.put(TranscriptionCompletedEvent(transcript="", turn_id="turn_next", turn_revision=0))
        replayed = _drain_until(ws, "response.done")
        assert any(ev["type"] == "response.output_audio.delta" for ev in replayed)


def test_per_session_false_interruption_config_via_session_update() -> None:
    # 1. Server default disabled (0); Session A opts in via session.update while Session B on same service stays off.
    unit = _make_unit(min_words=0, timeout_ms=2000)
    sid_a = unit.service.register()
    sid_b = unit.service.register()
    assert unit.service.session_min_interruption_words(sid_a) == 0
    assert unit.service.session_min_interruption_words(sid_b) == 0

    ev_enable = unit.service.parse_client_event(
        {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "audio": {
                    "input": {
                        "turn_detection": {
                            "type": "server_vad",
                            "min_interruption_words": 2,
                            "false_interruption_timeout_ms": 1200,
                        }
                    }
                },
            },
        }
    )
    assert ev_enable is not None
    assert unit.service.handle_session_update(sid_a, ev_enable) is None
    assert unit.service.session_min_interruption_words(sid_a) == 2
    assert unit.service.session_false_interruption_timeout_s(sid_a) == pytest.approx(1.2)
    assert unit.service.session_min_interruption_words(sid_b) == 0
    assert unit.service.session_false_interruption_timeout_s(sid_b) == pytest.approx(2.0)

    # Validation rejects negative/bool words and non-positive timeout.
    for bad_td in (
        {"type": "server_vad", "min_interruption_words": -1},
        {"type": "server_vad", "min_interruption_words": True},
        {"type": "server_vad", "false_interruption_timeout_ms": 0},
    ):
        bad_ev = unit.service.parse_client_event(
            {"type": "session.update", "session": {"type": "realtime", "audio": {"input": {"turn_detection": bad_td}}}}
        )
        assert bad_ev is not None
        err = unit.service.handle_session_update(sid_a, bad_ev)
        assert err is not None and err.error.type == "invalid_request_error"

    # 2. End-to-end WebSocket: server default 0, client sends an idle instructions update, then opts in before reply 1.
    with _ws_session(min_words=0) as (u, ws, sid):
        ws.send_json({"type": "session.update", "session": {"type": "realtime", "instructions": "Be concise."}})
        _drain_until(ws, "session.updated")
        ws.send_json(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "audio": {
                        "input": {
                            "turn_detection": {
                                "type": "server_vad",
                                "min_interruption_words": 1,
                                "false_interruption_timeout_ms": 1500,
                            }
                        }
                    },
                },
            }
        )
        updated = _drain_until(ws, "session.updated")[-1]
        assert updated["session"]["audio"]["input"]["turn_detection"]["min_interruption_words"] == 1

        _queue_completed_reply(u, sid, "rk_optin", [("Opted in reply.", _tone(RATE * 2))])
        _drain_until(ws, "response.done")
        u.text_output_queue.put(SpeechStartedEvent(turn_id="turn_opt", turn_revision=0))
        _drain_until(ws, "input_audio_buffer.speech_started")
        u.text_output_queue.put(SpeechStoppedEvent(turn_id="turn_opt", turn_revision=0, duration_s=0.2))
        u.text_output_queue.put(TranscriptionCompletedEvent(transcript="", turn_id="turn_opt", turn_revision=0))
        replayed = _drain_until(ws, "response.done")
        assert any(ev["type"] == "response.output_audio.delta" for ev in replayed)

    # 3. Server default enabled (1), client disables via session.update(min_interruption_words=0) mid-check -> cancels.
    with _ws_session(min_words=1) as (u, ws, _sid):
        gen = u.cancel_scope.generation
        u.output_queue.put(AssistantOutputEvent(text="Streaming.", response_key="rk_off_s", cancel_generation=gen))
        u.output_queue.put(AudioOutput(audio=_tone(RATE * 2), response_key="rk_off_s", cancel_generation=gen))
        _drain_until(ws, "response.output_audio.delta")
        u.text_output_queue.put(SpeechStartedEvent(turn_id="turn_off_s", turn_revision=0))
        _drain_until(ws, "input_audio_buffer.speech_started")
        assert u.session.pending_false_interruption is not None

        ws.send_json(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "audio": {"input": {"turn_detection": {"type": "server_vad", "min_interruption_words": 0}}},
                },
            }
        )
        _drain_until(ws, "session.updated")
        assert u.session.pending_false_interruption is None
        assert u.cancel_scope.generation == gen + 1


def test_chat_serializers_merge_consecutive_assistant_items_and_deduplicate_rewind() -> None:
    from openai.types.realtime import RealtimeConversationItemFunctionCall, RealtimeConversationItemFunctionCallOutput

    from speech_to_speech.LLM.chat import Chat

    chat = Chat(10)
    chat.add_item(make_user_message("What is the capital of France?"))
    chat.add_heard_assistant_message("item_1", "The capital of France is", status="incomplete")
    chat.add_heard_assistant_message("item_2", "France is Paris.", status="completed")

    assert chat.to_transformers_chat() == [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]
    resp_chat = chat.to_responses_api_chat()
    assert [m["role"] for m in resp_chat] == ["user", "assistant"]
    assert resp_chat[1]["status"] == "completed"
    assert resp_chat[1]["content"][0]["text"] == "The capital of France is Paris."

    # Following tool call attaches cleanly to the merged assistant turn.
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
    assert tf[1]["content"] == "The capital of France is Paris."
    assert [tc["function"]["name"] for tc in tf[1]["tool_calls"]] == ["lookup"]

    # Two completed assistant messages merge for strict templates without stripping legitimate repeated words.
    chat2 = Chat(10)
    chat2.add_item(make_user_message("Repeat."))
    chat2.add_heard_assistant_message("a1", "Hello.", status="completed")
    chat2.add_heard_assistant_message("a2", "Hello.", status="completed")
    assert chat2.to_transformers_chat() == [
        {"role": "user", "content": "Repeat."},
        {"role": "assistant", "content": "Hello. Hello."},
    ]
