from __future__ import annotations

import sys
from threading import Lock, RLock
from types import SimpleNamespace

import speech_to_speech.pipeline.speculative_turns as speculative_module
import speech_to_speech.utils.mlx_lock as mlx_lock
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
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
        == "Turn turn_1 rev=0 latency: stt=0.14s llm=1.28s tts_ttfa=0.16s e2e=1.61s vad_decision=n/a hold=n/a smart_turn_status=n/a status=completed"
        + (" mlx_lock_wait=0.00s" if sys.platform == "darwin" else "")
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
    assert tracker.llm_ttft_s == 0.3
    assert "llm_ttft=" not in line
    assert "llm=2.00s" in line
    assert "tts_ttfa=0.20s" in line
    assert "e2e=3.00s" in line
    # No handler name was given, so the total carries no breakdown.
    assert ("mlx_lock_wait=0.15s" in line) == (sys.platform == "darwin")
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

    assert tracker.mlx_lock_wait_s >= 0.01
    assert tracker.mlx_lock_waits["test-timeout"].total_s == tracker.mlx_lock_wait_s
    assert tracker.mlx_lock_waits["test-timeout"].count == 1
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


def test_smart_wait_uses_union_of_actual_intervals_and_stops_with_original_response() -> None:
    store = TurnLatencyStore()
    pending = store.get_or_create_for_turn("turn_3", 0)
    pending.smart_turn_status = "incomplete"
    pending.smart_turn_wait_s = 0.0
    store.record_smart_wait("turn_3", 0, 1.0, 2.0)
    response = store.get_or_create_response("first", turn_id="turn_3", turn_revision=0, session_id="session")
    store.record_smart_wait("turn_3", 0, 1.5, 3.0)
    assert response.smart_turn_wait_s == 2.0

    store.pop("first", session_id="session")
    followup = store.get_or_create_response("followup", turn_id="turn_3", turn_revision=0, session_id="session")
    store.record_smart_wait("turn_3", 0, 3.0, 4.0)
    assert followup.smart_turn_status is None
    assert followup.smart_turn_wait_s is None


def test_configured_grace_is_not_reported_as_wait_without_a_gate(monkeypatch) -> None:
    clock = [0.0]

    class ControlledCondition:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def notify_all(self):
            pass

        def wait(self, duration):
            clock[0] += duration

    monkeypatch.setattr(
        speculative_module,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], perf_counter=lambda: clock[0]),
    )
    store = TurnLatencyStore()
    pending = store.get_or_create_for_turn("turn_1", 0)
    pending.smart_turn_status = "complete"
    pending.smart_turn_grace_s = 2.0
    pending.smart_turn_wait_s = 0.0
    turns = SpeculativeTurnTracker()
    turns._condition = ControlledCondition()
    turns.wait_observer = store.record_smart_wait
    turns.start_turn()
    turns.start_reopen_grace("turn_1", 0, 2.0)

    clock[0] = 3.0
    assert turns.wait_for_gate("turn_1", 0)
    assert pending.smart_turn_wait_s == 0.0

    clock[0] = 4.0
    turns.start_reopen_grace("turn_1", 0, 2.0)
    assert turns.wait_for_gate("turn_1", 0)
    assert pending.smart_turn_wait_s == 2.0


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
    assert len(store._session_keys) == 0


def test_response_lookup_does_not_create_or_revive_trackers() -> None:
    store = TurnLatencyStore()
    assert store.get_response(None) is None
    assert store.get_response("resp_a") is None
    tracker = store.get_or_create_response("resp_a", turn_id="turn_1", turn_revision=0, session_id="sess_1")
    assert store.get_response("resp_a") is tracker

    store.discard_response("resp_a", session_id="sess_1")
    assert store.get_response("resp_a") is None
    assert store._trackers == {}
    assert len(store._session_keys) == 0


def test_clear_session_keeps_pending_while_other_sessions_active() -> None:
    store = TurnLatencyStore()
    store.get_or_create_response("resp_a", turn_id="turn_1", turn_revision=0, session_id="sess_1")
    pending = store.get_or_create_for_turn("turn_9", 0)
    pending.record_stt(0.1)

    store.get_or_create_response("resp_b", turn_id="turn_1", turn_revision=0, session_id="sess_2")
    store.clear_session("sess_1")

    assert store.pop("resp_a", session_id="sess_1") is None
    assert store.get_or_create_for_turn("turn_9", 0).stt_s == 0.1
    assert len(store._session_keys) == 1

    assert store.pop("resp_b", session_id="sess_2") is not None
    assert len(store._session_keys) == 0

    store.clear_session("sess_2")
    assert store.get_or_create_for_turn("turn_9", 0).stt_s is None


def test_format_log_line_shows_settle_named_lock_and_cut_tts(monkeypatch) -> None:
    import speech_to_speech.pipeline.turn_latency as latency

    monkeypatch.setattr(latency.sys, "platform", "darwin")
    tracker = TurnLatencyTracker(turn_id="turn_1", turn_revision=2, status="cancelled")
    tracker.record_stt(0.12)
    tracker.record_vad_settle(0.61)
    tracker.record_llm(2.08)
    tracker.record_tts_ttfa(0.18)
    tracker.record_e2e(3.0)
    tracker.record_mlx_lock_wait(0.02, "ParakeetSTT-Progressive")
    tracker.record_mlx_lock_hold(0.88, "Qwen3TTS")

    assert tracker.format_log_line() == (
        "Turn turn_1 rev=2 latency: stt=0.12s llm=2.08s tts_ttfa=0.18s e2e=3.00s "
        "vad_settle=0.61s vad_decision=n/a hold=n/a smart_turn_status=n/a status=cancelled "
        "mlx_lock_wait=0.02s[ParakeetSTT-Progressive:0.02s] "
        "mlx_lock_hold=0.88s[Qwen3TTS:0.88s] tts=cut"
    )


def test_repeated_lock_waits_are_summed_per_handler(monkeypatch) -> None:
    """A turn takes the lock many times; the breakdown must stay one entry per
    handler and account for the printed total."""
    import speech_to_speech.pipeline.turn_latency as latency

    monkeypatch.setattr(latency.sys, "platform", "darwin")
    tracker = TurnLatencyTracker(turn_id="turn_1", turn_revision=0)
    for _ in range(12):
        tracker.record_mlx_lock_wait(0.01, "ParakeetSTT-Progressive")
    tracker.record_mlx_lock_wait(0.03, "Qwen3TTS")

    assert tracker.mlx_lock_waits["ParakeetSTT-Progressive"].count == 12
    assert "mlx_lock_wait=0.15s[ParakeetSTT-Progressive:0.12sx12,Qwen3TTS:0.03s]" in tracker.format_log_line()


def test_pending_turn_merges_vad_settle_and_named_lock() -> None:
    store = TurnLatencyStore()
    pending = store.get_or_create_for_turn("turn_3", 1)
    pending.record_vad_settle(0.4)
    pending.record_mlx_lock_wait(0.02, "ParakeetSTT")
    pending.record_mlx_lock_hold(0.3, "ParakeetSTT")
    assert store.pending_for_turn("turn_3", 1) is pending

    tracker = store.get_or_create_response("resp_a", turn_id="turn_3", turn_revision=1)
    tracker.record_mlx_lock_hold(0.5, "ParakeetSTT")
    assert tracker.vad_settle_s == 0.4
    assert tracker.mlx_lock_waits["ParakeetSTT"].total_s == 0.02
    assert tracker.mlx_lock_hold_s == 0.8
    assert tracker.mlx_lock_holds["ParakeetSTT"].count == 2
    assert store.pending_for_turn("turn_3", 1) is None


def test_lock_hold_recorded_once_on_outer_release(monkeypatch) -> None:
    monkeypatch.setattr(mlx_lock, "_mlx_lock", RLock())
    tracker = TurnLatencyTracker(turn_id="turn_1", turn_revision=0)
    with bind_active_turn_latency_tracker(tracker):
        assert mlx_lock.acquire_mlx_lock(handler_name="Qwen3TTS")
        assert mlx_lock.acquire_mlx_lock(handler_name="Qwen3TTS")
        mlx_lock.release_mlx_lock(handler_name="Qwen3TTS")
        assert tracker.mlx_lock_hold_s == 0.0
        mlx_lock.release_mlx_lock(handler_name="Qwen3TTS")

    assert tracker.mlx_lock_hold_s > 0.0
    assert tracker.mlx_lock_holds["Qwen3TTS"].count == 1


def test_log_platform_and_export_fields(monkeypatch):
    import speech_to_speech.pipeline.turn_latency as latency

    tracker = TurnLatencyTracker(turn_id="turn_1", mlx_lock_wait_s=0.25)
    for platform in ("darwin", "linux", "win32"):
        monkeypatch.setattr(latency.sys, "platform", platform)
        assert ("mlx_lock_wait=" in tracker.format_log_line()) == (platform == "darwin")
    assert set(tracker.metadata_payload(response_key="r", status="completed")) == {
        "version",
        "turn_id",
        "turn_revision",
        "response_key",
        "status",
        "e2e_s",
        "vad_decision_s",
        "smart_status",
        "stt_s",
        "llm_s",
        "tts_ttfa_s",
        "hold_s",
    }
