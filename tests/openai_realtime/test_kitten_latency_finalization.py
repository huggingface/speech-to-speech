"""Kitten timing boundaries through real response finalization."""

import json
import logging

import numpy as np
import pytest

from speech_to_speech.pipeline.events import SpeechStartedEvent, TranscriptionCompletedEvent
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.pipeline.turn_latency import TURN_LATENCY_METADATA_KEY
from speech_to_speech.TTS import kitten_tts_handler as kitten_module
from tests.test_kitten_tts_latency import _handler, _waveform


@pytest.mark.parametrize("empty_first_chunk", [False, True])
def test_internal_chunks_keep_first_model_audio_in_terminal_record(
    service, conn_id, monkeypatch, caplog, empty_first_chunk
):
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
    monkeypatch.setattr(kitten_module, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(kitten_module.console, "print", lambda *_a, **_k: None)
    handler._chunk_text = lambda text: ["empty", "first", "second"] if empty_first_chunk else ["first", "second"]
    calls = []

    def generate(text, voice):
        calls.append(text)
        clock[0] = {"empty": 100.05, "first": 100.10, "second": 100.60}[text]
        return np.array([], dtype=np.float32) if text == "empty" else _waveform()

    def resample(audio, *, up, down):
        clock[0] += 0.20
        return audio

    handler.model.generate_single_chunk = generate
    monkeypatch.setattr(kitten_module, "resample_poly", resample)
    tts_input = TTSInput(
        text="Hello",
        response_key=request.response_key,
        turn_id="turn_1",
        turn_revision=0,
        speech_stopped_at_s=99.0,
    )
    outputs = list(handler.process(tts_input))
    assert calls == (["empty", "first", "second"] if empty_first_chunk else ["first", "second"])
    np.testing.assert_array_equal(
        np.concatenate(outputs),
        np.clip(np.concatenate([_waveform(), _waveform()]) * 32768, -32768, 32767).astype(np.int16),
    )
    for output in outputs:
        service.encode_audio_chunk(conn_id, output.tobytes(), request.response_key)

    # A later text segment must not replace either first-audio measurement.
    clock[0] = 110.0
    handler.model.generate_single_chunk = lambda text, voice: _waveform()
    handler._chunk_text = lambda text: [text]
    list(handler.process(tts_input))
    terminal_input = EndOfResponse(response_key=request.response_key, turn_id="turn_1", turn_revision=0)
    assert list(handler.process(terminal_input)) == [AUDIO_RESPONSE_DONE]
    logger = "speech_to_speech.api.openai_realtime.handlers.response"
    with caplog.at_level(logging.INFO, logger=logger):
        events = service.finish_response(conn_id, response_key=request.response_key)
    done = next(event for event in events if event.type == "response.done")
    payload = json.loads(done.response.metadata[TURN_LATENCY_METADATA_KEY])
    assert payload["response_key"] == request.response_key
    assert payload["status"] == "completed"
    assert payload["tts_ttfa_s"] == pytest.approx(0.10)
    assert payload["e2e_s"] == pytest.approx(1.80)
    lines = [r.message for r in caplog.records if r.name == logger and " latency: " in r.message]
    assert len(lines) == 1
    assert f"response_key={request.response_key}" in lines[0]
    assert "tts_ttfa=0.10s e2e=1.80s" in lines[0]
    assert service.turn_latency_store.get_response(request.response_key) is None
