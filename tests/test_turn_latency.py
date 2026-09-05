from __future__ import annotations

import logging

import pytest

from speech_to_speech.pipeline.turn_latency import TurnLatencyStore, TurnLatencyTracker


def test_turn_latency_tracker_format_log_line() -> None:
    tracker = TurnLatencyTracker(
        turn_id="turn_1",
        turn_revision=0,
        stt_s=0.14,
        llm_s=1.28,
        tts_ttfa_s=0.16,
        e2e_s=1.61,
        mlx_lock_wait_s=0.0,
        status="completed",
    )
    assert (
        tracker.format_log_line()
        == "Turn turn_1 rev=0 latency: stt=0.14s llm=1.28s tts_ttfa=0.16s e2e=1.61s mlx_lock_wait=0.00s status=completed"
    )


def test_turn_latency_tracker_record_and_reset() -> None:
    tracker = TurnLatencyTracker()
    tracker.reset("turn_2", 1)
    tracker.record_stt(0.5)
    tracker.record_llm(2.0)
    tracker.record_tts_ttfa(0.2)
    tracker.record_e2e(3.0)
    tracker.record_mlx_lock_wait(0.1)
    tracker.record_mlx_lock_wait(0.05)
    tracker.status = "cancelled"

    line = tracker.format_log_line()
    assert line is not None
    assert "turn_2 rev=1" in line
    assert "stt=0.50s" in line
    assert "llm=2.00s" in line
    assert "tts_ttfa=0.20s" in line
    assert "e2e=3.00s" in line
    assert "mlx_lock_wait=0.15s" in line
    assert "status=cancelled" in line

    tracker.reset()
    assert tracker.turn_id is None
    assert tracker.format_log_line() is None


def test_turn_latency_tracker_first_write_wins_for_ttfa_and_e2e() -> None:
    tracker = TurnLatencyTracker(turn_id="turn_1", turn_revision=0)
    tracker.record_tts_ttfa(0.2)
    tracker.record_tts_ttfa(1.5)
    tracker.record_e2e(1.0)
    tracker.record_e2e(4.0)
    assert tracker.tts_ttfa_s == 0.2
    assert tracker.e2e_s == 1.0


def test_turn_latency_store_pending_turn_merges_into_response() -> None:
    store = TurnLatencyStore()
    pending = store.get_or_create_for_turn("turn_3", 0)
    pending.record_stt(0.12)
    pending.record_mlx_lock_wait(0.03)

    tracker = store.get_or_create_response("resp_a", turn_id="turn_3", turn_revision=0, session_id="sess_1")
    assert tracker.stt_s == 0.12
    assert tracker.mlx_lock_wait_s == 0.03
    assert store.get_or_create_for_turn("turn_3", 0) is not pending


def test_turn_latency_store_pop_and_clear_session() -> None:
    store = TurnLatencyStore()
    tracker = store.get_or_create_response("resp_a", turn_id="turn_1", turn_revision=0, session_id="sess_1")
    tracker.record_llm(1.0)

    popped = store.pop("resp_a", session_id="sess_1")
    assert popped is tracker
    assert store.pop("resp_a", session_id="sess_1") is None

    store.get_or_create_response("resp_b", turn_id="turn_2", turn_revision=0, session_id="sess_1")
    store.get_or_create_for_turn("turn_9", 0).record_stt(0.1)
    store.clear_session("sess_1")
    assert store.pop("resp_b", session_id="sess_1") is None
    assert store.get_or_create_for_turn("turn_9", 0).stt_s is None
    assert store.active_session_count == 0


def test_clear_session_keeps_pending_while_other_sessions_active() -> None:
    store = TurnLatencyStore()
    store.get_or_create_response("resp_a", turn_id="turn_1", turn_revision=0, session_id="sess_1")
    pending = store.get_or_create_for_turn("turn_9", 0)
    pending.record_stt(0.1)

    store.get_or_create_response("resp_b", turn_id="turn_1", turn_revision=0, session_id="sess_2")
    store.clear_session("sess_1")

    assert store.pop("resp_a", session_id="sess_1") is None
    assert store.get_or_create_for_turn("turn_9", 0).stt_s == 0.1
    assert store.active_session_count == 1

    assert store.pop("resp_b", session_id="sess_2") is not None
    assert store.active_session_count == 0

    store.clear_session("sess_2")
    assert store.get_or_create_for_turn("turn_9", 0).stt_s is None


def test_turn_latency_log_emission(caplog: pytest.LogCaptureFixture) -> None:
    """Simulates _log_turn_latency without importing the full realtime stack."""
    logger = logging.getLogger("speech_to_speech.api.openai_realtime.handlers.response")
    store = TurnLatencyStore()
    tracker = store.get_or_create_response("resp_a", turn_id="turn_1", turn_revision=0)
    tracker.record_stt(0.14)
    tracker.record_llm(1.28)
    tracker.record_tts_ttfa(0.16)
    tracker.record_e2e(1.61)

    with caplog.at_level(logging.INFO, logger="speech_to_speech.api.openai_realtime.handlers.response"):
        popped = store.pop("resp_a")
        assert popped is tracker
        line = popped.format_log_line()
        assert line is not None
        logger.info(line)

    assert len(caplog.records) == 1
    assert (
        caplog.records[0].message
        == "Turn turn_1 rev=0 latency: stt=0.14s llm=1.28s tts_ttfa=0.16s e2e=1.61s mlx_lock_wait=0.00s status=completed"
    )
