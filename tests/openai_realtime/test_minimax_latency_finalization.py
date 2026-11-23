"""MiniMax's inherited first-audio timings survive real response finalization."""

import json
import logging

import numpy as np
import pytest

from speech_to_speech.pipeline.events import SpeechStartedEvent, TranscriptionCompletedEvent
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.pipeline.turn_latency import TURN_LATENCY_METADATA_KEY
from tests.test_minimax_tts_handler import _handler
from tests.test_minimax_tts_latency import _audio_frames, _set_sse_stream


def test_inherited_timings_reach_response_done_and_terminal_log(service, conn_id, monkeypatch, caplog):
    service.dispatch_pipeline_event(conn_id, SpeechStartedEvent(turn_id="turn_1", turn_revision=0))
    service.dispatch_pipeline_event(
        conn_id,
        TranscriptionCompletedEvent(transcript="Hello", turn_id="turn_1", turn_revision=0, speech_stopped_at_s=99.0),
    )
    request = service.text_prompt_queue.get_nowait()
    service.response._ensure_response(conn_id, request.response_key)
    handler = _handler()
    handler.turn_latency_store = service.turn_latency_store
    clock = [100.0]
    monkeypatch.setattr("speech_to_speech.pipeline.turn_latency.perf_counter", lambda: clock[0])
    samples = np.arange(768, dtype="<i2")
    _set_sse_stream(monkeypatch, handler, clock, _audio_frames(samples))
    item = TTSInput(
        text="Hello.",
        response_key=request.response_key,
        turn_id="turn_1",
        turn_revision=0,
        speech_stopped_at_s=99.0,
    )
    try:
        outputs = list(handler.process(item))
        np.testing.assert_array_equal(np.concatenate(outputs), samples)
        for chunk in outputs:
            service.encode_audio_chunk(conn_id, chunk.tobytes(), request.response_key)

        terminal = EndOfResponse(response_key=request.response_key, turn_id="turn_1", turn_revision=0)
        assert list(handler.process(terminal)) == [AUDIO_RESPONSE_DONE]
        logger = "speech_to_speech.api.openai_realtime.handlers.response"
        with caplog.at_level(logging.INFO, logger=logger):
            events = service.finish_response(conn_id, response_key=request.response_key)

        done = next(event for event in events if event.type == "response.done")
        payload = json.loads(done.response.metadata[TURN_LATENCY_METADATA_KEY])
        assert payload["response_key"] == request.response_key
        assert payload["status"] == "completed"
        assert payload["tts_ttfa_s"] == pytest.approx(0.10)
        assert payload["e2e_s"] == pytest.approx(1.40)
        lines = [
            record.message for record in caplog.records if record.name == logger and " latency: " in record.message
        ]
        assert len(lines) == 1
        assert f"response_key={request.response_key}" in lines[0]
        assert "tts_ttfa=0.10s e2e=1.40s" in lines[0]
        assert service.turn_latency_store.get_response(request.response_key) is None
    finally:
        handler.cleanup()
