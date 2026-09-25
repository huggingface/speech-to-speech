from __future__ import annotations

from threading import Lock

import speech_to_speech.utils.mlx_lock as mlx_lock
from speech_to_speech.pipeline.turn_latency import (
    TurnLatencyStore,
    TurnLatencyTracker,
    bind_active_turn_latency_tracker,
)


def test_turn_latency_tracker_format_log_line() -> None:
    tracker = TurnLatencyTracker(
        turn_id="turn_1",
        turn_revision=0,
        stt_s=0.14,
        llm_ttft_s=0.11,
        llm_s=1.28,
        tts_ttfa_s=0.16,
        e2e_s=1.61,
        mlx_lock_wait_s=0.0,
        status="completed",
    )
    assert (
        tracker.format_log_line()
        == "Turn turn_1 rev=0 latency: stt=0.14s llm_ttft=0.11s llm=1.28s tts_ttfa=0.16s e2e=1.61s mlx_lock_wait=0.00s status=completed"
    )


def test_turn_latency_tracker_record() -> None:
    tracker = TurnLatencyTracker(turn_id="turn_2", turn_revision=1)
    tracker.record_stt(0.5)
    tracker.record_llm_ttft(0.3)
    tracker.record_llm_ttft(1.0)
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
    assert "llm_ttft=0.30s" in line
    assert "llm=2.00s" in line
    assert "tts_ttfa=0.20s" in line
    assert "e2e=3.00s" in line
    assert "mlx_lock_wait=0.15s" in line
    assert "status=cancelled" in line

    assert TurnLatencyTracker().format_log_line() is None


def test_timed_out_mlx_lock_acquisition_records_wait(monkeypatch) -> None:
    lock = Lock()
    monkeypatch.setattr(mlx_lock, "_mlx_lock", lock)
    tracker = TurnLatencyTracker(turn_id="turn_1", turn_revision=0)
    with lock, bind_active_turn_latency_tracker(tracker):
        with mlx_lock.MLXLockContext(handler_name="test-timeout", timeout=0.01) as acquired:
            assert not acquired

    assert tracker.mlx_lock_wait_s >= 0.01
    assert not lock.locked()


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


def test_response_lookup_does_not_create_or_revive_trackers() -> None:
    store = TurnLatencyStore()
    assert store.get_response(None) is None
    assert store.get_response("resp_a") is None
    tracker = store.get_or_create_response("resp_a", turn_id="turn_1", turn_revision=0, session_id="sess_1")
    assert store.get_response("resp_a") is tracker

    store.discard_response("resp_a", session_id="sess_1")
    assert store.get_response("resp_a") is None
    assert store._trackers == {}
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
