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


def test_turn_latency_store_get_or_create_and_pop() -> None:
    store = TurnLatencyStore()
    first = store.get_or_create("turn_3", 0)
    second = store.get_or_create("turn_3", 0)
    assert first is second
    first.record_stt(0.1)

    popped = store.pop("turn_3", 0)
    assert popped is first
    assert popped.stt_s == 0.1
    assert store.pop("turn_3", 0) is None


def test_turn_latency_log_emission(caplog: pytest.LogCaptureFixture) -> None:
    """Simulates _log_turn_latency without importing the full realtime stack."""
    logger = logging.getLogger("speech_to_speech.api.openai_realtime.handlers.response")
    store = TurnLatencyStore()
    tracker = store.get_or_create("turn_1", 0)
    tracker.record_stt(0.14)
    tracker.record_llm(1.28)
    tracker.record_tts_ttfa(0.16)
    tracker.record_e2e(1.61)

    with caplog.at_level(logging.INFO, logger="speech_to_speech.api.openai_realtime.handlers.response"):
        popped = store.pop("turn_1", 0)
        assert popped is tracker
        line = popped.format_log_line()
        assert line is not None
        logger.info(line)

    assert len(caplog.records) == 1
    assert (
        caplog.records[0].message
        == "Turn turn_1 rev=0 latency: stt=0.14s llm=1.28s tts_ttfa=0.16s e2e=1.61s mlx_lock_wait=0.00s status=completed"
    )


def test_end_response_logs_turn_latency(caplog: pytest.LogCaptureFixture) -> None:
    pytest.importorskip("speech_to_speech.utils.utils")

    from speech_to_speech.api.openai_realtime.handlers.response import ResponseHandler
    from speech_to_speech.api.openai_realtime.service import RealtimeService

    store = TurnLatencyStore()
    service = RealtimeService(turn_latency_store=store)
    conn_id = service.register()
    st = service._state(conn_id)
    st.speculative_user_turn_id = "turn_1"
    st.speculative_user_turn_revision = 0

    tracker = store.get_or_create("turn_1", 0)
    st.turn_latency = tracker
    tracker.record_stt(0.14)
    tracker.record_llm(1.28)
    tracker.record_tts_ttfa(0.16)
    tracker.record_e2e(1.61)

    handler = ResponseHandler(service)
    with caplog.at_level(logging.INFO):
        handler._end_response(conn_id, "completed")

    latency_logs = [record.message for record in caplog.records if "Turn turn_1 rev=0 latency:" in record.message]
    assert len(latency_logs) == 1
    assert (
        latency_logs[0]
        == "Turn turn_1 rev=0 latency: stt=0.14s llm=1.28s tts_ttfa=0.16s e2e=1.61s mlx_lock_wait=0.00s status=completed"
    )
