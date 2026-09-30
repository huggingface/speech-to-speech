from __future__ import annotations

from threading import Lock, RLock

import speech_to_speech.utils.mlx_lock as mlx_lock
from speech_to_speech.pipeline.turn_latency import (
    TurnLatencyStore,
    TurnLatencyTracker,
    bind_active_turn_latency_tracker,
)
from speech_to_speech.STT.parakeet_tdt_handler import ParakeetTDTSTTHandler


def test_turn_latency_tracker_format_log_line() -> None:
    tracker = TurnLatencyTracker(
        turn_id="turn_1",
        turn_revision=0,
        stt_s=0.14,
        llm_s=1.28,
        tts_ttfa_s=0.16,
        e2e_s=1.61,
        lock_wait_s=0.0,
        status="completed",
    )
    assert (
        tracker.format_log_line()
        == "Turn turn_1 rev=0 latency: stt=0.14s llm=1.28s tts_ttfa=0.16s e2e=1.61s lock_wait=0.00s status=completed"
    )


def test_turn_latency_tracker_record() -> None:
    tracker = TurnLatencyTracker(turn_id="turn_2", turn_revision=1)
    tracker.record_stt(0.5)
    tracker.record_llm(2.0)
    tracker.record_tts_ttfa(0.2)
    tracker.record_e2e(3.0)
    tracker.record_lock_wait(0.1)
    tracker.record_lock_wait(0.05)
    tracker.status = "cancelled"

    line = tracker.format_log_line()
    assert line is not None
    assert "turn_2 rev=1" in line
    assert "stt=0.50s" in line
    assert "llm=2.00s" in line
    assert "tts_ttfa=0.20s" in line
    assert "e2e=3.00s" in line
    # No handler name was given, so the total is printed without a breakdown.
    assert "lock_wait=0.15s " in line
    assert "status=cancelled" in line
    assert "tts=cut" in line

    assert TurnLatencyTracker().format_log_line() is None


def test_timed_out_mlx_lock_acquisition_records_wait(monkeypatch) -> None:
    lock = Lock()
    monkeypatch.setattr(mlx_lock, "_mlx_lock", lock)
    tracker = TurnLatencyTracker(turn_id="turn_1", turn_revision=0)
    with lock, bind_active_turn_latency_tracker(tracker):
        with mlx_lock.MLXLockContext(handler_name="test-timeout", timeout=0.01) as acquired:
            assert not acquired

    assert tracker.lock_wait_s >= 0.01
    assert tracker.lock_waits["test-timeout"].total_s == tracker.lock_wait_s
    assert tracker.lock_waits["test-timeout"].count == 1
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
    pending.record_lock_wait(0.03)

    tracker = store.get_or_create_response("resp_a", turn_id="turn_3", turn_revision=0, session_id="sess_1")
    assert tracker.stt_s == 0.12
    assert tracker.lock_wait_s == 0.03
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


def test_format_log_line_separates_settle_ttft_lock_and_cut_tts() -> None:
    tracker = TurnLatencyTracker(turn_id="turn_1", turn_revision=2, status="cancelled")
    tracker.record_stt(0.12)
    tracker.record_vad_settle(0.61)
    tracker.record_llm(2.08)
    tracker.record_llm_ttft(0.41)
    tracker.record_llm_ttft(9.0)
    tracker.record_tts_ttfa(0.18)
    tracker.record_e2e(3.0)
    tracker.record_lock_wait(0.02, "ParakeetSTT-Progressive")
    tracker.record_lock_hold(0.88, "Qwen3TTS")

    assert tracker.llm_ttft_s == 0.41
    assert tracker.format_log_line() == (
        "Turn turn_1 rev=2 latency: stt=0.12s vad_settle=0.61s llm=2.08s llm_ttft=0.41s "
        "tts_ttfa=0.18s e2e=3.00s "
        "lock_wait=0.02s[ParakeetSTT-Progressive:0.02s] "
        "lock_hold=0.88s[Qwen3TTS:0.88s] status=cancelled tts=cut"
    )


def test_repeated_lock_waits_are_summed_per_handler() -> None:
    """A turn takes the lock many times; the breakdown must stay one entry per
    handler and account for the printed total."""
    tracker = TurnLatencyTracker(turn_id="turn_1", turn_revision=0)
    for _ in range(12):
        tracker.record_lock_wait(0.01, "ParakeetSTT-Progressive")
    tracker.record_lock_wait(0.03, "Qwen3TTS")

    assert tracker.lock_waits["ParakeetSTT-Progressive"].count == 12
    line = tracker.format_log_line()
    assert line is not None
    assert "lock_wait=0.15s[ParakeetSTT-Progressive:0.12sx12,Qwen3TTS:0.03s]" in line


def test_pending_turn_merges_vad_settle_and_named_lock() -> None:
    store = TurnLatencyStore()
    pending = store.get_or_create_for_turn("turn_3", 1)
    pending.record_vad_settle(0.4)
    pending.record_lock_wait(0.02, "ParakeetSTT")
    pending.record_lock_hold(0.3, "ParakeetSTT")
    assert store.pending_for_turn("turn_3", 1) is pending

    tracker = store.get_or_create_response("resp_a", turn_id="turn_3", turn_revision=1)
    tracker.record_lock_hold(0.5, "ParakeetSTT")
    assert tracker.vad_settle_s == 0.4
    assert tracker.lock_waits["ParakeetSTT"].total_s == 0.02
    assert tracker.lock_hold_s == 0.8
    assert tracker.lock_holds["ParakeetSTT"].count == 2
    assert store.pending_for_turn("turn_3", 1) is None


def test_non_mlx_compute_lock_is_attributed_like_the_mlx_lock() -> None:
    """Parakeet's CUDA/CPU backend serializes on its own lock; that contention
    must reach the turn line too, not only the MLX path."""
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.backend = "nano_parakeet"
    handler.compute_lock = Lock()
    tracker = TurnLatencyTracker(turn_id="turn_1", turn_revision=0)

    with bind_active_turn_latency_tracker(tracker):
        with handler._compute_lock_context(handler_name="ParakeetSTT-Final", timeout=1.0) as acquired:
            assert acquired

    assert tracker.lock_waits["ParakeetSTT-Final"].count == 1
    assert tracker.lock_holds["ParakeetSTT-Final"].count == 1
    assert tracker.lock_hold_s > 0.0


def test_lock_hold_recorded_once_on_outer_release(monkeypatch) -> None:
    monkeypatch.setattr(mlx_lock, "_mlx_lock", RLock())
    tracker = TurnLatencyTracker(turn_id="turn_1", turn_revision=0)
    with bind_active_turn_latency_tracker(tracker):
        assert mlx_lock.acquire_mlx_lock(handler_name="Qwen3TTS")
        assert mlx_lock.acquire_mlx_lock(handler_name="Qwen3TTS")
        mlx_lock.release_mlx_lock(handler_name="Qwen3TTS")
        assert tracker.lock_hold_s == 0.0
        mlx_lock.release_mlx_lock(handler_name="Qwen3TTS")

    assert tracker.lock_hold_s > 0.0
    assert tracker.lock_holds["Qwen3TTS"].count == 1
