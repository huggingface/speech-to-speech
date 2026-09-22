"""One question per connection: the runner streams audio, reads the answer, and grades it."""

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from speech_to_speech.evals.big_bench_audio import runner as bba
from speech_to_speech.evals.big_bench_audio.dataset import EvalItem
from speech_to_speech.evals.big_bench_audio.runner import (
    RunnerConfig,
    build_session_update,
    iter_chunks,
    run_item,
    run_subset,
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
    assert result.reply == "Final answer: Yes"  # the last turn heard the whole question


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
    rejection = {"type": "error", "error": {"message": "session_limit_reached"}}
    install(monkeypatch, FakeSocket([], respond_after=4, first=rejection))

    result = await run_item(cfg, ITEM, pcm_for(cfg))

    assert "session_limit_reached" in result.error


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
