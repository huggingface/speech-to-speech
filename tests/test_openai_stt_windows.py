from __future__ import annotations

import io
import json
import wave
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue
from threading import Event, Thread

import numpy as np
import pytest
import torch
from openai.types.realtime import RealtimeSessionCreateRequest

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.pipeline.messages import Transcription, TranscriptionFailure, VADAudio
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.STT.openai_compatible_handler import (
    OpenAICompatibleSTTHandler,
    TranscriptionRequestError,
)
from tests.test_openai_stt_lifecycle import ControlledOperation
from tests.test_speculative_turns import _StaticVADIterator, _vad_handler_for_iterator
from tests.turns import reopen

SR = 16000
pytestmark = pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(OpenAICompatibleSTTHandler, "warmup", lambda self: None)
    handlers = []
    operations = []

    def create(*, cap=1, tracker=None, texts=None):
        handler = OpenAICompatibleSTTHandler(
            Event(),
            queue_in=Queue(),
            queue_out=Queue(),
            setup_kwargs={"max_window_s": cap, "speculative_turns": tracker},
        )
        calls = []

        def make(audio, **kwargs):
            calls.append((audio.copy(), kwargs))
            text = next(texts) if texts is not None else f"part{len(calls)}"
            operation = ControlledOperation(text)
            operation.release.set()
            operations.append(operation)
            return operation

        monkeypatch.setattr(handler, "_make_operation", make)
        handlers.append(handler)
        return handler, calls

    yield create
    for operation in operations:
        operation.release.set()
    for handler in handlers:
        handler.cleanup()


def source(seconds, *, mode="final", revision=0, turn="turn_1", values=None):
    return VADAudio(
        audio=values if values is not None else np.arange(round(seconds * SR), dtype=np.float32),
        mode=mode,
        turn_id=turn,
        turn_revision=revision,
    )


def run(handler, item):
    assert list(handler.process(item)) == []
    thread = handler._progressive_thread if item.mode == "progressive" else handler._final_thread
    assert thread is not None
    thread.join(timeout=5)
    assert not thread.is_alive()
    outputs = []
    while not handler.queue_out.empty():
        outputs.append(handler.queue_out.get_nowait())
    return outputs


def test_long_turn_uses_bounded_requests_and_preserves_all_samples_and_text(windows):
    handler, calls = windows(cap=120)
    item = source(601)
    (output,) = run(handler, item)
    assert isinstance(output, Transcription)
    assert output.text == " ".join(f"part{i + 1}" for i in range(len(calls)))
    assert len(calls) > 1
    assert max(len(a) for a, _ in calls) <= 120 * SR
    np.testing.assert_array_equal(np.concatenate([a for a, _ in calls]), item.audio)


@pytest.mark.parametrize("quiet", [False, True])
def test_cut_is_bounded_without_silence_and_prefers_a_quiet_frame_when_present(quiet):
    audio = np.ones(121 * SR, dtype=np.float32)
    if quiet:
        audio[118 * SR : 118 * SR + SR // 10] = 0.1
    end = OpenAICompatibleSTTHandler._window_end(audio, 0, 120 * SR)
    assert 115 * SR <= end <= 120 * SR
    if quiet:
        assert end == round(118.05 * SR)


def test_background_windows_do_not_emit_captions_or_a_final_and_final_reuses_prefix(windows):
    handler, calls = windows()
    handler.window_partials_enabled = False
    assert run(handler, source(0.5, mode="progressive")) == []
    assert not calls
    assert run(handler, source(2, mode="progressive")) == []
    assert calls
    committed_end = handler._windowed_turns["turn_1"].end
    earlier = len(calls)
    (final,) = run(handler, source(2.3))
    assert isinstance(final, Transcription)
    assert final.text.startswith("part1 ")
    assert calls[earlier][0][0] == committed_end
    assert max(len(a) for a, _ in calls) <= SR


def test_reopened_revisions_retain_completed_windows_and_revise_only_tail(windows):
    tracker = SpeculativeTurnTracker()
    tracker.start_turn()
    handler, calls = windows(tracker=tracker)
    (first,) = run(handler, source(2.3))
    assert isinstance(first, Transcription)
    for revision in range(1, 5):
        committed = handler._windowed_turns["turn_1"].end
        count = len(calls)
        reopen(tracker, revision=revision - 1)
        (output,) = run(handler, source(2.3 + revision * 0.8, revision=revision))
        assert isinstance(output, Transcription)
        assert output.text.startswith("part1 ")
        assert output.turn_revision == revision
        assert calls[count][0][0] == committed
    assert max(len(a) for a, _ in calls) <= SR


@pytest.mark.parametrize("change", ["audio", "language", "auto"])
def test_changed_committed_audio_or_language_invalidates_cached_text(windows, change):
    handler, calls = windows()
    handler.window_partials_enabled = False
    assert run(handler, source(2, mode="progressive")) == []
    count = len(calls)
    item = source(2.3)
    if change == "audio":
        item.audio = item.audio + 100000  # Enhancement can replace a progressive prefix.
    else:
        item.runtime_config = RuntimeConfig(
            session=RealtimeSessionCreateRequest(
                type="realtime",
                audio={"input": {"transcription": {"language": "es" if change == "language" else "auto"}}},
            )
        )
    (output,) = run(handler, item)
    assert calls[count][0][0] == item.audio[0]
    assert "part1" not in output.text.split()
    if change == "language":
        assert all(kwargs["language"] == "es" for _, kwargs in calls[count:])
    elif change == "auto":
        assert all(kwargs == {"language": None, "use_setup_language": False} for _, kwargs in calls[count:])


def test_final_does_not_wait_for_cancelled_progressive_or_accept_its_late_prefix(windows, monkeypatch):
    handler, calls = windows()
    old = ControlledOperation("obsolete", ignore_cancel=True)
    normal = handler._make_operation
    pending = [old]
    monkeypatch.setattr(
        handler, "_make_operation", lambda audio, **kw: pending.pop() if pending else normal(audio, **kw)
    )
    try:
        assert list(handler.process(source(2, mode="progressive"))) == []
        assert old.started.wait(1)
        assert list(handler.process(source(2))) == []
        final = handler.queue_out.get(timeout=2)
        assert isinstance(final, Transcription)
        assert "obsolete" not in final.text
        assert old.cancelled.is_set()
        assert not old.finished.is_set()
        committed = handler._windowed_turns["turn_1"].end
        old.release.set()
        handler._progressive_thread.join(1)
        assert handler._windowed_turns["turn_1"].end == committed
        assert handler.queue_out.empty()
    finally:
        old.release.set()


def test_session_end_clears_prefix_and_fences_late_window_commit(windows, monkeypatch):
    handler, calls = windows()
    old = ControlledOperation("old session", ignore_cancel=True)
    normal = handler._make_operation
    pending = [old]
    monkeypatch.setattr(
        handler, "_make_operation", lambda audio, **kw: pending.pop() if pending else normal(audio, **kw)
    )
    try:
        assert list(handler.process(source(2, mode="progressive"))) == []
        assert old.started.wait(1)
        handler.on_session_end()
        assert not handler._windowed_turns
        (output,) = run(handler, source(2))
        assert "old session" not in output.text
        assert calls[0][0][0] == 0
        old.release.set()
        handler._progressive_thread.join(1)
        assert handler.queue_out.empty()
    finally:
        old.release.set()


def test_failed_window_does_not_publish_a_truncated_success_and_reopen_keeps_earlier_text(windows, monkeypatch):
    handler, calls = windows()
    normal = handler._make_operation
    failure = ControlledOperation(error=TranscriptionRequestError("transcription request timed out"))
    failure.release.set()
    count = 0

    def make(audio):
        nonlocal count
        count += 1
        return failure if count == 2 else normal(audio)

    monkeypatch.setattr(handler, "_make_operation", make)
    (output,) = run(handler, source(3))
    assert isinstance(output, TranscriptionFailure)
    end = handler._windowed_turns["turn_1"].end
    (output,) = run(handler, source(3.2, revision=1))
    assert isinstance(output, Transcription)
    assert output.text.startswith("part1 ")
    assert calls[1][0][0] == end


@pytest.mark.parametrize(
    "parts,expected", [(["very", "very"], "very very"), (["こんにちは。", "世界"], "こんにちは。世界")]
)
def test_join_preserves_repetition_and_cjk_text(windows, parts, expected):
    handler, _ = windows(texts=iter(parts))
    (output,) = run(handler, source(1.2))
    assert output.text == expected


def test_anonymous_turns_do_not_share_text_cache_and_old_turn_caches_retire(windows):
    handler, calls = windows()
    run(handler, source(2, turn="old"))
    assert "old" in handler._windowed_turns
    run(handler, source(2, turn="new"))
    assert "old" not in handler._windowed_turns
    for _ in range(2):
        count = len(calls)
        run(handler, source(2, turn=None))
        assert calls[count][0][0] == 0
    assert None not in handler._windowed_turns


@pytest.mark.parametrize("cap", [0, 0.5, float("nan"), float("inf"), -1])
def test_window_configuration_rejects_nonfinite_or_too_small_caps(windows, cap):
    with pytest.raises(ValueError, match="max_window_s"):
        windows(cap=cap)


def test_vad_emits_internal_snapshots_without_closing_turn_or_enabling_live_captions(windows):
    iterator = _StaticVADIterator(
        triggered=True, vad_output=None, buffer_chunks=[torch.ones(SR // 2)], active_speech_samples=2 * SR
    )
    vad = _vad_handler_for_iterator(iterator)
    vad.transcription_window_samples = SR
    vad._last_window_snapshot_samples = 0
    vad.speculative_turns.start_turn()
    assert list(vad._process_realtime(None)) == []
    iterator.buffer = iterator._speech_chunks = [torch.ones(SR)]
    (snapshot,) = list(vad._process_realtime(None))
    assert snapshot.mode == "progressive"
    assert not vad.enable_realtime_transcription
    assert vad.text_output_queue.empty()
    assert list(vad._process_realtime(None)) == []
    handler, calls = windows()
    handler.window_partials_enabled = False
    assert run(handler, snapshot) == []
    assert calls
    assert handler.queue_out.empty()
    assert not handler._completed_final_revisions()
    # A reopening keeps the cumulative cursor; a new logical turn resets it.
    vad.speculative_turns.segment_finalized(1000)
    vad._speech_started_emitted = False
    vad._ensure_turn_for_speech_start(9000)
    assert vad._last_window_snapshot_samples == 0


def test_real_http_long_upload_is_rejected_whole_and_succeeds_through_bounded_requests(monkeypatch):
    received = []

    class Server(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["content-length"]))
            message = BytesParser(policy=policy.default).parsebytes(
                f"Content-Type: {self.headers['content-type']}\r\n\r\n".encode() + body
            )
            upload = next(p.get_payload(decode=True) for p in message.iter_parts() if p.get_filename())
            with wave.open(io.BytesIO(upload)) as wav:
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").copy()
            if len(pcm) > SR:
                status = 400
                payload = {"error": {"message": "Audio exceeds maximum allowed duration of 1s."}}
            else:
                received.append(pcm)
                status = 200
                payload = {"text": f"part{len(received)}", "language": "en"}
            encoded = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args):
            pass

    monkeypatch.setattr(OpenAICompatibleSTTHandler, "warmup", lambda self: None)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Server)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    handlers = []
    item = source(3.2, values=np.linspace(-0.9, 0.9, round(3.2 * SR), dtype=np.float32))
    try:
        for cap in [None, 1]:
            handler = OpenAICompatibleSTTHandler(
                Event(),
                queue_in=Queue(),
                queue_out=Queue(),
                setup_kwargs={
                    "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                    "max_window_s": cap,
                },
            )
            handlers.append(handler)
            (output,) = run(handler, item)
            if cap is None:
                assert isinstance(output, TranscriptionFailure)
                assert "duration limit" in output.message
            else:
                assert isinstance(output, Transcription)
                assert output.text == " ".join(f"part{i + 1}" for i in range(len(received)))
        with wave.open(io.BytesIO(handler._encode_wav(item.audio))) as wav:
            expected = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        np.testing.assert_array_equal(np.concatenate(received), expected)
    finally:
        for handler in handlers:
            handler.cleanup()
        server.shutdown()
        server.server_close()
        worker.join(1)
