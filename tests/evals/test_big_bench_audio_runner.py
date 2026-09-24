"""One question per connection: the runner streams audio, reads the answer, and grades it."""

import asyncio
import json
import sys
from queue import Queue
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from speech_to_speech.api.openai_realtime.service import RealtimeService, build_error_event
from speech_to_speech.evals.big_bench_audio import runner as bba
from speech_to_speech.evals.big_bench_audio.dataset import EvalItem, Subset
from speech_to_speech.evals.big_bench_audio.report import aggregate
from speech_to_speech.evals.big_bench_audio.runner import (
    RunnerConfig,
    build_session_update,
    iter_chunks,
    run_item,
    run_subset,
)
from speech_to_speech.pipeline.events import (
    AssistantOutputEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TranscriptionCompletedEvent,
)

ITEM = EvalItem(id=7, category="web_of_lies", official_answer="Yes", file_name="data/question_7.mp3")


def transcript_events(text, *, audio_bytes=4):
    """The server events one completed spoken response produces."""
    return [
        {"type": "response.created"},
        {"type": "response.output_audio.delta", "delta": "AAAA"[:audio_bytes]},
        {"type": "response.output_audio_transcript.delta", "delta": text},
        {"type": "response.output_audio_transcript.done", "transcript": text},
        {"type": "response.done"},
    ]


class FakeSocket:
    """A Realtime server that answers once the client stops sending audio."""

    def __init__(self, events, *, respond_after, first=None):
        self.sent = []
        self.closed = False
        self.outbox = asyncio.Queue()
        self.events = events
        self.respond_after = respond_after
        self.outbox.put_nowait(first or {"type": "session.created", "session": {"id": "sess_1"}})

    async def send(self, raw):
        self.sent.append(json.loads(raw))
        if self.sent[-1].get("type") == "session.update":
            self.outbox.put_nowait({"type": "session.updated"})
        if self.appends == self.respond_after:
            for event in self.events:
                self.outbox.put_nowait(event)

    async def recv(self):
        return json.dumps(await self.outbox.get())

    @property
    def appends(self):
        return sum(1 for event in self.sent if event.get("type") == "input_audio_buffer.append")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        self.closed = True
        return False


def install(monkeypatch, socket):
    monkeypatch.setitem(sys.modules, "websockets", SimpleNamespace(connect=lambda *a, **kw: socket))
    return socket


def config(**overrides):
    """Fast settings: 20 ms chunks sent at 1000x, with a 50 ms settle window."""
    settings = {
        "speed": 1000.0,
        "chunk_ms": 20,
        "trailing_silence_ms": 40,
        "grace_s": 0.05,
        "settle_s": 0.0,
        "response_timeout_s": 2.0,
    }
    settings.update(overrides)
    return RunnerConfig(**settings)


def pcm_for(cfg, chunks=2):
    return b"\x01\x02" * (cfg.chunk_bytes * chunks // 2)


def test_session_update_carries_the_eval_turn_taking_settings():
    event = build_session_update(config(silence_duration_ms=900, vad_threshold=0.4, voice="alloy"))
    session = event["session"]

    assert event["type"] == "session.update"
    assert session["audio"]["input"]["turn_detection"] == {
        "type": "server_vad",
        "interrupt_response": True,
        "silence_duration_ms": 900,
        "threshold": 0.4,
    }
    assert session["audio"]["output"]["voice"] == "alloy"
    assert "Final answer" in session["instructions"]


def test_a_short_final_chunk_is_padded_to_full_width():
    chunks = list(iter_chunks(b"\x01" * 5, 4))

    assert chunks == [b"\x01\x01\x01\x01", b"\x01\x00\x00\x00"]


async def test_a_correct_spoken_answer_is_graded_and_timed(monkeypatch):
    cfg = config()
    events = transcript_events("Tracing it through. Final answer: Yes.")
    install(monkeypatch, FakeSocket(events, respond_after=4))

    result = await run_item(cfg, ITEM, pcm_for(cfg))

    assert result.error is None
    assert result.extracted == "yes"
    assert result.correct is True
    assert result.method == "marker"
    assert result.ttfb_s is not None and result.turn_s is not None
    assert result.audio_out_bytes > 0


async def test_the_eval_session_is_configured_before_any_audio(monkeypatch):
    cfg = config()
    socket = install(monkeypatch, FakeSocket(transcript_events("Final answer: Yes"), respond_after=4))

    await run_item(cfg, ITEM, pcm_for(cfg))

    assert socket.sent[0]["type"] == "session.update"
    assert socket.appends == 4  # two of question audio, two of trailing silence
    assert socket.closed is True


async def test_a_wrong_answer_is_recorded_rather_than_dropped(monkeypatch):
    cfg = config()
    install(monkeypatch, FakeSocket(transcript_events("Final answer: No"), respond_after=4))

    result = await run_item(cfg, ITEM, pcm_for(cfg))

    assert (result.extracted, result.correct, result.error) == ("no", False, None)


async def test_an_unreadable_reply_is_unparsed_not_wrong(monkeypatch):
    cfg = config()
    install(monkeypatch, FakeSocket(transcript_events("I could not follow that."), respond_after=4))

    result = await run_item(cfg, ITEM, pcm_for(cfg))

    assert result.extracted is None
    assert result.correct is None


async def test_a_question_split_into_several_turns_is_flagged(monkeypatch):
    """VAD cutting a long recording in two leaves the last answer with polluted context."""
    cfg = config()
    events = transcript_events("Final answer: No") + transcript_events("Final answer: Yes")
    install(monkeypatch, FakeSocket(events, respond_after=4))

    result = await run_item(cfg, ITEM, pcm_for(cfg))

    assert result.error is not None and result.error.startswith("split_turn")
    assert result.reply == "Final answer: Yes"  # retain the last reply for diagnosis


@pytest.mark.parametrize("reopened", [False, True])
async def test_speech_items_detect_a_split_before_the_next_response(monkeypatch, reopened):
    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    sid = service.register()
    try:
        events = service.dispatch_pipeline_event(sid, SpeechStartedEvent(turn_id="first", turn_revision=0))
        events += service.dispatch_pipeline_event(sid, SpeechStoppedEvent(turn_id="first", turn_revision=0))
        events += service.dispatch_pipeline_event(sid, AssistantOutputEvent(text="Final answer: Yes"))
        events += service.encode_audio_chunk(sid, b"\x01\x00" * 320)
        events += service.finish_response(sid)
        events += service.dispatch_pipeline_event(
            sid,
            SpeechStartedEvent(
                turn_id="first" if reopened else "second",
                turn_revision=1,
                reopened=reopened,
            ),
        )
        # STT/LLM has not produced another response yet. Use actual service
        # serialization, including distinct item IDs and same-item reopening.
        wire = [event.model_dump(exclude_none=True) for event in events]
        item_ids = {event["item_id"] for event in wire if event["type"] == "input_audio_buffer.speech_started"}
        assert len(item_ids) == (1 if reopened else 2)
        socket = install(monkeypatch, FakeSocket(wire, respond_after=4))

        result = await run_item(config(), ITEM, pcm_for(config()))

        assert result.reply == "Final answer: Yes"
        assert result.error == (None if reopened else "split_turn (2 input items)")
        assert aggregate([result])["totals"]["correct"] == int(reopened)
        assert socket.closed
    finally:
        service.unregister(sid)


@pytest.mark.parametrize("after_response", [False, True])
async def test_final_transcription_then_reopen_uses_new_item_identity(monkeypatch, after_response):
    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    sid = service.register()
    try:
        events = []
        for event in (
            SpeechStartedEvent(turn_id="first", turn_revision=0),
            SpeechStoppedEvent(turn_id="first", turn_revision=0),
            TranscriptionCompletedEvent(transcript="First premise.", turn_id="first", turn_revision=0),
            SpeechStartedEvent(turn_id="first", turn_revision=1, reopened=True),
        ):
            events += service.dispatch_pipeline_event(sid, event)
        events += service.dispatch_pipeline_event(sid, AssistantOutputEvent(text="Final answer: Yes"))
        events += service.encode_audio_chunk(sid, b"\x01\x00" * 320)
        events += service.finish_response(sid)
        # Repeating the current input identity must not be mistaken for a split
        # just because a speculative revision used another ID earlier.
        events += service.dispatch_pipeline_event(
            sid,
            SpeechStartedEvent(
                turn_id="second" if after_response else "first", turn_revision=2, reopened=not after_response
            ),
        )
        wire = [event.model_dump(exclude_none=True) for event in events]
        ids = [event["item_id"] for event in wire if event["type"] == "input_audio_buffer.speech_started"]
        assert ids[0] != ids[1]
        assert (ids[1] != ids[2]) == after_response
        socket = install(monkeypatch, FakeSocket(wire, respond_after=4))
        result = await run_item(config(), ITEM, pcm_for(config()))
        assert result.reply == "Final answer: Yes"
        assert bool(result.error) == after_response
        assert aggregate([result])["totals"]["correct"] == int(not after_response)
        assert socket.closed
    finally:
        service.unregister(sid)


@pytest.mark.parametrize("recovers", [False, True])
async def test_session_capacity_wait_is_bounded_and_does_not_send_audio(monkeypatch, recovers):
    monkeypatch.setattr(bba, "_SESSION_RETRY_S", 0.01)
    sockets = []
    rejection = build_error_event("All 1 session slots are in use.", "session_limit_reached").model_dump(
        exclude_none=True
    )

    def connect(*args, **kwargs):
        ready = recovers and len(sockets) >= 3
        socket = FakeSocket(transcript_events("Final answer: Yes"), respond_after=4, first=None if ready else rejection)
        sockets.append(socket)
        return socket

    monkeypatch.setitem(sys.modules, "websockets", SimpleNamespace(connect=connect))
    cfg = config(response_timeout_s=0.1)
    result = await asyncio.wait_for(run_item(cfg, ITEM, pcm_for(cfg)), timeout=1)
    assert len(sockets) >= 3
    assert all(socket.closed for socket in sockets)
    assert all(socket.appends == 0 for socket in (sockets[:-1] if recovers else sockets))
    if recovers:
        assert result.error is None and result.correct is True
        assert sockets[-1].appends == 4
    else:
        assert "session_capacity_timeout" in result.error


@pytest.mark.parametrize("completed", [False, True])
async def test_response_deadline_holds_even_when_events_keep_arriving(monkeypatch, completed):
    cfg = config(response_timeout_s=0.1, grace_s=0.5)
    events = transcript_events("Final answer: Yes")
    if not completed:
        events = events[:-1]  # A streaming response with no response.done yet.

    class BusySocket(FakeSocket):
        async def recv(self):
            if not self.outbox.empty():
                return await super().recv()
            await asyncio.sleep(0.005)
            return json.dumps(
                {"type": "rate_limits.updated"}
                if completed
                else {
                    "type": "response.output_audio.delta",
                    "delta": "AAAA",
                }
            )

    socket = install(monkeypatch, BusySocket(events, respond_after=4))
    result = await asyncio.wait_for(run_item(cfg, ITEM, pcm_for(cfg)), timeout=1.0)

    assert result.error == (None if completed else "response_timeout")
    assert socket.closed


async def test_a_server_that_never_answers_times_out(monkeypatch):
    cfg = config(response_timeout_s=0.2)
    install(monkeypatch, FakeSocket([], respond_after=4))

    result = await run_item(cfg, ITEM, pcm_for(cfg))

    assert result.error == "response_timeout"
    assert result.correct is None


async def test_a_server_error_event_ends_the_item(monkeypatch):
    cfg = config()
    events = [{"type": "error", "error": {"message": "pipeline exploded"}}]
    install(monkeypatch, FakeSocket(events, respond_after=4))

    result = await run_item(cfg, ITEM, pcm_for(cfg))

    assert result.error == "pipeline exploded"


async def test_a_rejected_connection_is_reported_as_an_item_error(monkeypatch):
    cfg = config()
    rejection = build_error_event("Invalid API key", "authentication_error").model_dump(exclude_none=True)
    socket = install(monkeypatch, FakeSocket([], respond_after=4, first=rejection))

    result = await run_item(cfg, ITEM, pcm_for(cfg))

    assert "Invalid API key" in result.error
    assert socket.closed and socket.appends == 0


async def test_input_transcription_is_kept_for_debugging(monkeypatch):
    cfg = config()
    events = [
        {"type": "conversation.item.input_audio_transcription.completed", "transcript": "Does Jamey tell the truth?"},
        *transcript_events("Final answer: Yes"),
    ]
    install(monkeypatch, FakeSocket(events, respond_after=4))

    result = await run_item(cfg, ITEM, pcm_for(cfg))

    assert result.input_transcript == "Does Jamey tell the truth?"


async def test_run_subset_retries_a_failed_item_then_reports_progress(monkeypatch, tmp_path):
    from speech_to_speech.evals.big_bench_audio.dataset import Subset

    cfg = config(retries=2)
    monkeypatch.setattr(bba, "resolve_audio", lambda item, subset: tmp_path / "q.mp3")
    monkeypatch.setattr(bba, "load_pcm16_mono", lambda path: pcm_for(cfg))

    attempts = []

    async def flaky(_config, item, _pcm):
        attempts.append(item.id)
        failed = bba.ItemResult(id=item.id, category=item.category, official_answer=item.official_answer)
        if len(attempts) == 1:
            failed.error = "response_timeout"
            return failed
        failed.correct = True
        return failed

    monkeypatch.setattr(bba, "run_item", flaky)
    seen = []
    results = await run_subset(Subset(name="t", items=(ITEM,)), cfg, progress=lambda *args: seen.append(args))

    assert attempts == [7, 7]
    assert results[0].correct is True
    assert seen[0][:2] == (1, 1)


async def test_run_subset_does_not_retry_a_split_turn(monkeypatch, tmp_path):
    """A split is a VAD-settings problem; rerunning the same audio reproduces it."""
    from speech_to_speech.evals.big_bench_audio.dataset import Subset

    cfg = config(retries=3)
    monkeypatch.setattr(bba, "resolve_audio", lambda item, subset: tmp_path / "q.mp3")
    monkeypatch.setattr(bba, "load_pcm16_mono", lambda path: pcm_for(cfg))

    attempts = []

    async def split(_config, item, _pcm):
        attempts.append(item.id)
        result = bba.ItemResult(id=item.id, category=item.category, official_answer=item.official_answer)
        result.error = "split_turn (2 responses)"
        return result

    monkeypatch.setattr(bba, "run_item", split)
    await run_subset(Subset(name="t", items=(ITEM,)), cfg)

    assert attempts == [7]


@pytest.mark.parametrize("stage", ["resolve_audio", "load_pcm16_mono"])
async def test_run_subset_retries_transient_audio_failures(monkeypatch, tmp_path, stage):
    cfg = config(retries=1)
    pcm = pcm_for(cfg)
    path = tmp_path / "question.mp3"
    resolve = Mock(return_value=path)
    decode = Mock(return_value=pcm)
    failing = resolve if stage == "resolve_audio" else decode
    failing.side_effect = [OSError("temporary audio failure"), failing.return_value]
    result = bba.ItemResult(id=ITEM.id, category=ITEM.category, official_answer=ITEM.official_answer)
    run = AsyncMock(return_value=result)
    monkeypatch.setattr(bba, "resolve_audio", resolve)
    monkeypatch.setattr(bba, "load_pcm16_mono", decode)
    monkeypatch.setattr(bba, "run_item", run)

    assert await run_subset(Subset(name="test", items=(ITEM,)), cfg) == [result]
    assert failing.call_count == 2
    run.assert_awaited_once_with(cfg, ITEM, pcm)


@pytest.mark.parametrize("chunks", [1, 3])
async def test_every_question_length_is_streamed_in_full(monkeypatch, chunks):
    cfg = config()
    socket = install(monkeypatch, FakeSocket(transcript_events("Final answer: Yes"), respond_after=chunks + 2))

    await run_item(cfg, ITEM, pcm_for(cfg, chunks))

    assert socket.appends == chunks + 2


async def test_text_without_synthesized_audio_fails(monkeypatch):
    cfg = config()
    events = [
        event for event in transcript_events("Final answer: Yes") if event["type"] != "response.output_audio.delta"
    ]
    install(monkeypatch, FakeSocket(events, respond_after=4))
    result = await run_item(cfg, ITEM, pcm_for(cfg))
    assert result.error == "no_audio_output"
    assert result.ttfb_s is None


@pytest.mark.parametrize("status", ["cancelled", "failed", "incomplete"])
async def test_unsuccessful_response_is_an_error(monkeypatch, status):
    cfg = config()
    events = transcript_events("Final answer: Yes")
    events[-1]["response"] = {"status": status}
    install(monkeypatch, FakeSocket(events, respond_after=4))
    result = await run_item(cfg, ITEM, pcm_for(cfg))
    assert result.error == f"response_{status}"


async def test_audio_latency_does_not_start_at_text(monkeypatch):
    cfg = config()
    capture = bba._TurnCapture()
    sent = asyncio.Event()
    sent.set()
    socket = FakeSocket([], respond_after=100)
    socket.outbox = asyncio.Queue()
    for event in [
        {"type": "response.created"},
        {"type": "response.output_audio_transcript.delta", "delta": "Yes"},
        {"type": "response.done"},
    ]:
        socket.outbox.put_nowait(event)
    await bba._consume(socket, cfg, capture, sent)
    assert capture.last.first_audio_at is None


async def test_rejected_session_update_never_streams_audio(monkeypatch):
    class RejectUpdate(FakeSocket):
        async def send(self, raw):
            self.sent.append(json.loads(raw))
            self.outbox.put_nowait({"type": "error", "error": {"message": "invalid session settings"}})

    socket = install(monkeypatch, RejectUpdate([], respond_after=4))
    result = await run_item(config(), ITEM, pcm_for(config()))
    assert "invalid session settings" in result.error
    assert socket.appends == 0
