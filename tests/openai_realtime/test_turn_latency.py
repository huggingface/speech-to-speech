"""Latency records emitted by the actual response lifecycle."""

import logging

import pytest

import speech_to_speech.LLM.language_model as language_model_module
from speech_to_speech.LLM.language_model import LanguageModelHandler
from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
from speech_to_speech.pipeline.events import PipelineEvent, SpeechStartedEvent, TranscriptionCompletedEvent
from speech_to_speech.pipeline.messages import EndOfResponse, LLMResponseChunk

LATENCY_LOGGER = "speech_to_speech.api.openai_realtime.handlers.response"


def _queue_turn(service, conn_id, *, turn_id="turn_1", revision=0, interrupt=True, reopened=False):
    service.dispatch_pipeline_event(
        conn_id,
        SpeechStartedEvent(turn_id=turn_id, turn_revision=revision, interrupt_response=interrupt, reopened=reopened),
    )
    pending = service.turn_latency_store.get_or_create_for_turn(turn_id, revision)
    pending.record_stt(0.12)
    service.dispatch_pipeline_event(
        conn_id,
        TranscriptionCompletedEvent(transcript="Hello", turn_id=turn_id, turn_revision=revision),
    )
    return service.text_prompt_queue.get_nowait()


def _latency_lines(caplog):
    return [r.message for r in caplog.records if r.name == LATENCY_LOGGER and " latency: " in r.message]


@pytest.mark.parametrize("partial_output", [False, True])
def test_failed_generation_records_duration_before_terminal_output(
    service, conn_id, monkeypatch, caplog, partial_output
):
    request = _queue_turn(service, conn_id)
    clock = [10.0]
    monkeypatch.setattr(language_model_module, "perf_counter", lambda: clock[0])

    def fail_generation(self, chat, language_code, gen, ctx, runtime_config, response):
        if partial_output:
            yield LLMResponseChunk(text="Partial answer.", runtime_config=runtime_config)
        clock[0] += 0.25
        raise RuntimeError("controlled failure")

    monkeypatch.setattr(LanguageModelHandler, "_generate", fail_generation)
    handler = object.__new__(LanguageModelHandler)
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.enable_lang_prompt = False
    handler.compactor = None
    handler.turn_latency_store = service.turn_latency_store
    processor = object.__new__(LMOutputProcessor)
    processor.setup()
    terminals = []

    with caplog.at_level(logging.INFO, logger=LATENCY_LOGGER):
        for chunk in handler.process(request):
            for output in processor.process(chunk):
                if isinstance(output, PipelineEvent):
                    service.dispatch_pipeline_event(conn_id, output)
                elif isinstance(output, EndOfResponse):
                    # Finish before the LLM generator resumes after yielding its
                    # terminal, as the downstream consumer can in production.
                    terminals.extend(service.finish_response(conn_id, response_key=output.response_key))
                    lines = _latency_lines(caplog)
                    assert len(lines) == 1
                    assert "llm=0.25s" in lines[0]
                    assert "status=failed" in lines[0]

    done = [event for event in terminals if event.type == "response.done"]
    assert len(done) == 1
    assert done[0].response.status == "failed"
    assert service.turn_latency_store._trackers == {}
