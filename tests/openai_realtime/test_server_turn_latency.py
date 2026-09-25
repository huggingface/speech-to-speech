"""Response latency through remote STT, LLM, and TTS handlers."""

import json
import logging
from queue import Queue
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

import speech_to_speech.LLM.base_openai_compatible_language_model as llm_module
import speech_to_speech.TTS.openai_compatible_handler as tts_module
from speech_to_speech.pipeline.events import AssistantResponseDoneEvent, SpeechStartedEvent, TranscriptionCompletedEvent
from speech_to_speech.pipeline.messages import EndOfResponse, TTSInput, VADAudio
from speech_to_speech.STT.openai_compatible_handler import (
    HttpTranscriptionResult,
    OpenAICompatibleSTTHandler,
    _TranscriptionRequest,
)
from speech_to_speech.STT.streaming_handler import VLLMRealtimeSTTHandler
from speech_to_speech.STT.transcription_notifier import TranscriptionNotifier
from tests.test_chat_completions_backend import _make_handler as make_chat_handler
from tests.test_openai_tts_handler import _openai_tts_handler
from tests.test_responses_api_language_model import _make_handler as make_responses_handler
from tests.test_streaming_stt_handler import _handler as make_streaming_handler
from tests.test_streaming_stt_handler import _SocketFactory

LATENCY_LOGGER = "speech_to_speech.api.openai_realtime.handlers.response"


def _finish(service, conn_id, response_key, caplog, *, status="completed"):
    service.dispatch_pipeline_event(conn_id, AssistantResponseDoneEvent(response_key=response_key))
    with caplog.at_level(logging.INFO, logger=LATENCY_LOGGER):
        events = service.finish_response(conn_id, status=status, response_key=response_key)
    assert any(event.type == "response.done" for event in events)
    return next(
        record.message for record in caplog.records if record.name == LATENCY_LOGGER and " latency: " in record.message
    )


def _dispatch_transcription(service, conn_id, transcription):
    notifier = object.__new__(TranscriptionNotifier)
    notifier.setup(text_output_queue=Queue(), should_listen=Event())
    list(notifier.process(transcription))
    service.dispatch_pipeline_event(conn_id, notifier.text_output_queue.get_nowait())
    return service.text_prompt_queue.get_nowait()


def test_http_stt_final_request_reaches_response_log_without_progressive_time(service, conn_id, monkeypatch, caplog):
    monkeypatch.setattr(OpenAICompatibleSTTHandler, "warmup", lambda self: None)
    handler = OpenAICompatibleSTTHandler(Event(), Queue(), Queue(), setup_kwargs={"base_url": "http://fake/v1"})
    handler.turn_latency_store = service.turn_latency_store
    handler._make_operation = lambda audio: SimpleNamespace(run=lambda cancel_check: HttpTranscriptionResult("hello"))
    service.dispatch_pipeline_event(conn_id, SpeechStartedEvent(turn_id="turn_1", turn_revision=0))
    progressive = VADAudio(audio=np.zeros(160, dtype=np.float32), mode="progressive", turn_id="turn_1", turn_revision=0)
    handler._run_request(_TranscriptionRequest(progressive, handler._session_generation))
    assert service.turn_latency_store._pending_turn == {}
    handler.queue_out.get_nowait()
    source = VADAudio(audio=np.zeros(160, dtype=np.float32), mode="final", turn_id="turn_1", turn_revision=0)
    handler._run_request(_TranscriptionRequest(source, handler._session_generation))
    request = _dispatch_transcription(service, conn_id, handler.queue_out.get_nowait())

    line = _finish(service, conn_id, request.response_key, caplog)
    assert "stt=n/a" not in line
    assert "llm=n/a" in line
    assert f"response_key={request.response_key}" in line
    assert service.turn_latency_store._pending_turn == {}


def test_vllm_realtime_final_commit_reaches_response_log(service, conn_id, caplog):
    def on_send(event, socket):
        if event == {"type": "input_audio_buffer.commit", "final": True}:
            socket.incoming.put(json.dumps({"type": "transcription.done", "text": "hello"}))

    handler = make_streaming_handler(_SocketFactory(on_send), handler_type=VLLMRealtimeSTTHandler)
    handler.turn_latency_store = service.turn_latency_store
    service.dispatch_pipeline_event(conn_id, SpeechStartedEvent(turn_id="turn_1", turn_revision=0))
    try:
        handler.start_turn("turn_1", 0)
        handler.append_audio(b"\x01\x00" * 512)
        source = VADAudio(audio=np.zeros(160, dtype=np.float32), mode="final", turn_id="turn_1", turn_revision=0)
        transcription = list(handler.process(source))[0]
        request = _dispatch_transcription(service, conn_id, transcription)
        line = _finish(service, conn_id, request.response_key, caplog)
        assert "stt=n/a" not in line
    finally:
        handler.cleanup()


@pytest.mark.parametrize("make_handler", [make_responses_handler, make_chat_handler])
@pytest.mark.parametrize("failure", [False, True])
def test_remote_llm_records_generation_through_terminal_response(
    service, conn_id, caplog, monkeypatch, make_handler, failure
):
    service.dispatch_pipeline_event(conn_id, SpeechStartedEvent(turn_id="turn_1", turn_revision=0))
    service.dispatch_pipeline_event(
        conn_id, TranscriptionCompletedEvent(transcript="hello", turn_id="turn_1", turn_revision=0)
    )
    request = service.text_prompt_queue.get_nowait()
    handler = make_handler()
    handler.turn_latency_store = service.turn_latency_store
    handler._serialize = lambda chat: [{"role": "user", "content": "hello"}]
    clock = [10.0]
    monkeypatch.setattr(llm_module, "perf_counter", lambda: clock[0])

    def provider_request(api_input, optional_kwargs):
        clock[0] = 10.25
        if failure:
            raise RuntimeError("provider failed")
        return object()

    handler._request = provider_request
    handler._iter_events = lambda response: iter([llm_module.TextDelta(text="hello")])
    outputs = list(handler.process(request))
    assert any(isinstance(output, EndOfResponse) and bool(output.error) == failure for output in outputs)
    line = _finish(service, conn_id, request.response_key, caplog, status="failed" if failure else "completed")
    assert "llm=0.25s" in line
    assert f"status={'failed' if failure else 'completed'}" in line


def test_remote_tts_tracks_provider_audio_before_block_assembly_and_preserves_first_segment(
    service, conn_id, caplog, monkeypatch
):
    service.dispatch_pipeline_event(conn_id, SpeechStartedEvent(turn_id="turn_1", turn_revision=0))
    service.dispatch_pipeline_event(
        conn_id, TranscriptionCompletedEvent(transcript="hello", turn_id="turn_1", turn_revision=0)
    )
    request = service.text_prompt_queue.get_nowait()
    handler = _openai_tts_handler(monkeypatch)
    handler.turn_latency_store = service.turn_latency_store
    clock = [10.0]
    monkeypatch.setattr(tts_module, "perf_counter", lambda: clock[0])
    original_blocks = handler._resample_to_blocks

    def delayed_blocks(*args):
        for block in original_blocks(*args):
            clock[0] += 0.3
            yield block

    handler._resample_to_blocks = delayed_blocks
    first = TTSInput(
        text="First", turn_id="turn_1", turn_revision=0, response_key=request.response_key, speech_stopped_at_s=9.0
    )
    second = TTSInput(
        text="Second", turn_id="turn_1", turn_revision=0, response_key=request.response_key, speech_stopped_at_s=9.0
    )
    assert list(handler.process(first))
    tracker = service.turn_latency_store.get_response(request.response_key)
    assert tracker.tts_ttfa_s == pytest.approx(0.0)
    assert tracker.e2e_s == pytest.approx(1.3)
    clock[0] = 20.0
    assert list(handler.process(second))
    line = _finish(service, conn_id, request.response_key, caplog)
    assert "tts_ttfa=0.00s" in line
    assert "e2e=1.30s" in line
    assert service.turn_latency_store.get_response(request.response_key) is None
    clock[0] = 30.0
    assert list(handler.process(first))
    assert service.turn_latency_store.get_response(request.response_key) is None
