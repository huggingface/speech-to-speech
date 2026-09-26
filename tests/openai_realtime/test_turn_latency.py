"""Latency records emitted by the actual response lifecycle."""

import logging
from contextlib import nullcontext
from queue import Queue
from threading import Event, Thread
from types import SimpleNamespace

import numpy as np
import pytest
from openai.types.realtime import ConversationItemCreateEvent, ResponseCreateEvent
from openai.types.realtime.conversation_item import RealtimeConversationItemFunctionCall

import speech_to_speech.LLM.language_model as language_model_module
import speech_to_speech.TTS.qwen3_tts_handler as qwen3_tts_module
from speech_to_speech.LLM.language_model import LanguageModelHandler
from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import (
    AssistantOutputEvent,
    AssistantResponseDoneEvent,
    PipelineEvent,
    ResponseGenerationDoneEvent,
    SpeechStartedEvent,
    TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.messages import (
    PIPELINE_END,
    AssistantToolCallPart,
    EndOfResponse,
    LLMResponseChunk,
    Transcription,
    TTSInput,
    VADAudio,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.STT.parakeet_tdt_handler import ParakeetTDTSTTHandler
from speech_to_speech.STT.transcription_notifier import TranscriptionNotifier
from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler

LATENCY_LOGGER = "speech_to_speech.api.openai_realtime.handlers.response"


def _queue_turn(service, conn_id, *, turn_id="turn_1", revision=0, interrupt=True, reopened=False, stt_s=0.12):
    service.dispatch_pipeline_event(
        conn_id,
        SpeechStartedEvent(turn_id=turn_id, turn_revision=revision, interrupt_response=interrupt, reopened=reopened),
    )
    pending = service.turn_latency_store.get_or_create_for_turn(turn_id, revision)
    pending.record_stt(stt_s)
    service.dispatch_pipeline_event(
        conn_id,
        TranscriptionCompletedEvent(transcript="Hello", turn_id=turn_id, turn_revision=revision),
    )
    return service.text_prompt_queue.get_nowait()


def _latency_lines(caplog):
    return [r.message for r in caplog.records if r.name == LATENCY_LOGGER and " latency: " in r.message]


@pytest.fixture
def final_stt_event(service):
    """Run final STT and notification with model inference and locking stubbed."""
    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.enable_live_transcription = False
    handler.backend = "mlx"
    handler.last_language = "en"
    handler.start_language = None
    handler.turn_latency_store = service.turn_latency_store
    handler._compute_lock_context = lambda **kwargs: nullcontext(True)
    notifier = object.__new__(TranscriptionNotifier)
    notifier.setup(text_output_queue=Queue(), should_listen=Event())

    def transcribe(turn_id, revision, text):
        handler._process_mlx_final = lambda audio: (text, "en")
        audio = VADAudio(audio=np.zeros(1600, dtype=np.float32), mode="final", turn_id=turn_id, turn_revision=revision)
        for transcription in handler.process(audio):
            list(notifier.process(transcription))
        return notifier.text_output_queue.get_nowait()

    return transcribe


def test_empty_final_stt_discards_only_its_pending_measurement(service, conn_id, final_stt_event):
    store = service.turn_latency_store
    other = store.get_or_create_for_turn("other_turn", 0)
    other.record_stt(0.4)

    for index in range(3):
        turn_id = f"turn_empty_{index}"
        service.dispatch_pipeline_event(conn_id, SpeechStartedEvent(turn_id=turn_id, turn_revision=0))
        event = final_stt_event(turn_id, 0, "")
        assert (turn_id, 0) in store._pending_turn
        events = service.dispatch_pipeline_event(conn_id, event)

        assert [event.type for event in events] == ["conversation.item.input_audio_transcription.completed"]
        assert events[0].transcript == ""
        assert service.text_prompt_queue.empty()
        assert store._pending_turn == {("other_turn", 0): other}
        assert store._trackers == {}


@pytest.mark.parametrize("stt_finishes_after_reopen", [False, True])
def test_stale_final_stt_discards_only_superseded_revision(
    service, conn_id, final_stt_event, caplog, stt_finishes_after_reopen
):
    service.speculative_turns = SpeculativeTurnTracker()
    service.dispatch_pipeline_event(conn_id, SpeechStartedEvent(turn_id="turn_1", turn_revision=0))
    if not stt_finishes_after_reopen:
        stale = final_stt_event("turn_1", 0, "Old transcript")

    service.dispatch_pipeline_event(
        conn_id, SpeechStartedEvent(turn_id="turn_1", turn_revision=1, reopened=True, interrupt_response=False)
    )
    if stt_finishes_after_reopen:
        stale = final_stt_event("turn_1", 0, "Old transcript")
    current = final_stt_event("turn_1", 1, "Current transcript")
    store = service.turn_latency_store
    current_tracker = store._pending_turn[("turn_1", 1)]
    assert ("turn_1", 0) in store._pending_turn

    assert service.dispatch_pipeline_event(conn_id, stale) == []
    assert service.text_prompt_queue.empty()
    assert store._pending_turn == {("turn_1", 1): current_tracker}

    service.dispatch_pipeline_event(conn_id, current)
    request = service.text_prompt_queue.get_nowait()
    tracker = store.get_response(request.response_key)
    assert tracker.stt_s == current_tracker.stt_s
    assert store._pending_turn == {}
    service.dispatch_pipeline_event(conn_id, AssistantResponseDoneEvent(response_key=request.response_key))
    with caplog.at_level(logging.INFO, logger=LATENCY_LOGGER):
        service.finish_response(conn_id, response_key=request.response_key)
    lines = _latency_lines(caplog)
    assert len(lines) == 1
    assert "Turn turn_1 rev=1" in lines[0]
    assert "stt=n/a" not in lines[0]
    assert store._trackers == {}


def test_stt_worker_discards_latency_when_revision_changes_during_inference(service, conn_id, caplog):
    speculative_turns = SpeculativeTurnTracker()
    service.speculative_turns = speculative_turns
    service.dispatch_pipeline_event(conn_id, SpeechStartedEvent(turn_id="turn_1", turn_revision=0))
    store = service.turn_latency_store
    unrelated = store.get_or_create_for_turn("other_turn", 0)

    handler = object.__new__(ParakeetTDTSTTHandler)
    handler.enable_live_transcription = False
    handler.backend = "mlx"
    handler.last_language = "en"
    handler.start_language = None
    handler.turn_latency_store = store
    handler.speculative_turns = speculative_turns
    handler.stop_event = Event()
    handler.queue_in = Queue()
    handler.queue_out = Queue()
    handler.pipeline_index = None
    handler._times = []
    handler.cleanup = lambda: None
    handler._compute_lock_context = lambda **kwargs: nullcontext(True)
    inference_calls = []

    def infer(audio):
        inference_calls.append(len(inference_calls))
        if len(inference_calls) == 1:
            # Reopening while the model runs makes the final output stale at
            # the worker's output gate, before the notifier/service sees it.
            speculative_turns.observe("turn_1", 1)
            return "Superseded transcript", "en"
        return "Current transcript", "en"

    handler._process_mlx_final = infer
    for revision in (0, 1):
        handler.queue_in.put(
            VADAudio(audio=np.zeros(1600, dtype=np.float32), mode="final", turn_id="turn_1", turn_revision=revision)
        )
    handler.queue_in.put(PIPELINE_END)
    handler.run()

    assert inference_calls == [0, 1]
    output = handler.queue_out.get_nowait()
    assert isinstance(output, Transcription)
    assert output.turn_revision == 1
    assert handler.queue_out.get_nowait() == PIPELINE_END
    assert handler.queue_out.empty()
    assert set(store._pending_turn) == {("other_turn", 0), ("turn_1", 1)}
    current_stt_s = store._pending_turn[("turn_1", 1)].stt_s

    notifier = object.__new__(TranscriptionNotifier)
    notifier.setup(text_output_queue=Queue(), should_listen=Event())
    list(notifier.process(output))
    service.dispatch_pipeline_event(conn_id, notifier.text_output_queue.get_nowait())
    request = service.text_prompt_queue.get_nowait()
    assert store.get_response(request.response_key).stt_s == current_stt_s
    service.dispatch_pipeline_event(conn_id, AssistantResponseDoneEvent(response_key=request.response_key))
    with caplog.at_level(logging.INFO, logger=LATENCY_LOGGER):
        service.finish_response(conn_id, response_key=request.response_key)
    assert len(_latency_lines(caplog)) == 1
    assert "Turn turn_1 rev=1" in _latency_lines(caplog)[0]
    assert store._pending_turn == {("other_turn", 0): unrelated}
    assert store._trackers == {}


def test_cancelled_inflight_llm_cannot_recreate_latency_tracker(service, conn_id, monkeypatch):
    request = _queue_turn(service, conn_id)
    scope = CancelScope()
    entered, resume = Event(), Event()
    handler = object.__new__(LanguageModelHandler)
    handler.cancel_scope = scope
    handler.speculative_turns = None
    handler.enable_lang_prompt = False
    handler.compactor = None
    handler.turn_latency_store = service.turn_latency_store
    apply_instructions = LanguageModelHandler._apply_instructions

    def pause_after_instructions(self, *args, **kwargs):
        apply_instructions(self, *args, **kwargs)
        entered.set()
        assert resume.wait(5)

    def cancelled_generation(self, chat, language_code, gen, ctx, runtime_config, response):
        assert self._check_stop(gen, ctx)
        return
        yield

    monkeypatch.setattr(LanguageModelHandler, "_apply_instructions", pause_after_instructions)
    monkeypatch.setattr(LanguageModelHandler, "_generate", cancelled_generation)
    outputs, errors = [], []

    def work():
        try:
            outputs.extend(handler.process(request))
        except Exception as exc:
            errors.append(exc)

    worker = Thread(target=work)
    worker.start()
    try:
        assert entered.wait(5)
        service.handle_response_cancel(conn_id)
        scope.cancel()
        assert service.turn_latency_store._trackers == {}
    finally:
        resume.set()
        worker.join(5)

    assert not worker.is_alive()
    assert not errors
    assert outputs and isinstance(outputs[-1], EndOfResponse)
    assert outputs[-1].response_key == request.response_key
    # Check before the late terminal is delivered: it must not be responsible
    # for cleaning up a tracker resurrected by the cancelled worker.
    assert service.turn_latency_store._trackers == {}
    service.close_response_key(conn_id, request.response_key)
    service.unregister(conn_id)
    assert service.turn_latency_store.active_session_count == 0
    assert service.turn_latency_store._trackers == {}


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
                    assert "llm_ttft=n/a llm=0.25s" in lines[0]
                    assert "status=failed" in lines[0]

    done = [event for event in terminals if event.type == "response.done"]
    assert len(done) == 1
    assert done[0].response.status == "failed"
    assert service.turn_latency_store._trackers == {}


@pytest.mark.parametrize("status", ["completed", "cancelled", "failed", "incomplete"])
def test_terminal_response_emits_one_latency_record(service, conn_id, caplog, status):
    request = _queue_turn(service, conn_id)
    tracker = service.turn_latency_store.get_or_create_response(request.response_key)
    tracker.record_llm_ttft(0.19)
    tracker.record_llm(1.28)
    tracker.record_tts_ttfa(0.16)
    tracker.record_e2e(1.61)
    tracker.record_mlx_lock_wait(0.03)
    service.dispatch_pipeline_event(conn_id, AssistantResponseDoneEvent(response_key=request.response_key))

    with caplog.at_level(logging.INFO, logger=LATENCY_LOGGER):
        events = service.finish_response(conn_id, status=status, response_key=request.response_key)
        service.finish_response(conn_id, status=status, response_key=request.response_key)

    assert _latency_lines(caplog) == [
        "Turn turn_1 rev=0 latency: stt=0.12s llm_ttft=0.19s llm=1.28s tts_ttfa=0.16s e2e=1.61s "
        f"mlx_lock_wait=0.03s status={status} response_key={request.response_key}"
    ]
    done = [event for event in events if event.type == "response.done"]
    assert len(done) == 1
    assert done[0].response.status == status
    assert service.turn_latency_store._trackers == {}
    assert service.turn_latency_store.active_session_count == 0


@pytest.mark.parametrize("new_turn,revision,reopened", [("turn_2", 0, False), ("turn_1", 1, True)])
def test_non_interrupting_speech_keeps_original_response_attribution(
    service, conn_id, caplog, new_turn, revision, reopened
):
    original = _queue_turn(service, conn_id)
    service.dispatch_pipeline_event(conn_id, AssistantResponseDoneEvent(response_key=original.response_key))
    newer = _queue_turn(
        service, conn_id, turn_id=new_turn, revision=revision, interrupt=False, reopened=reopened, stt_s=0.34
    )
    assert service._state(conn_id).current_response_key == original.response_key

    with caplog.at_level(logging.INFO, logger=LATENCY_LOGGER):
        service.finish_response(conn_id, response_key=original.response_key)
        assert newer.response_key in service.turn_latency_store._trackers
        service.dispatch_pipeline_event(conn_id, AssistantResponseDoneEvent(response_key=newer.response_key))
        service.finish_response(conn_id, response_key=newer.response_key)

    lines = _latency_lines(caplog)
    assert len(lines) == 2
    assert "Turn turn_1 rev=0 latency: stt=0.12s" in lines[0]
    assert f"response_key={original.response_key}" in lines[0]
    assert f"Turn {new_turn} rev={revision} latency: stt=0.34s" in lines[1]
    assert f"response_key={newer.response_key}" in lines[1]


def test_unregister_clears_unfinished_measurements_before_session_reuse(service, conn_id, caplog):
    request = _queue_turn(service, conn_id)
    old = service.turn_latency_store.get_or_create_response(request.response_key)
    old.record_llm(9.0)
    old.record_tts_ttfa(8.0)
    old.record_e2e(7.0)
    service.turn_latency_store.get_or_create_for_turn("turn_2", 0).record_stt(6.0)

    service.unregister(conn_id)
    assert service.turn_latency_store._trackers == {}
    assert service.turn_latency_store._pending_turn == {}
    assert service.turn_latency_store.active_session_count == 0

    new_conn_id = service.register()
    try:
        fresh = _queue_turn(service, new_conn_id)
        service.dispatch_pipeline_event(new_conn_id, AssistantResponseDoneEvent(response_key=fresh.response_key))
        with caplog.at_level(logging.INFO, logger=LATENCY_LOGGER):
            service.finish_response(new_conn_id, response_key=fresh.response_key)
        lines = _latency_lines(caplog)
        assert len(lines) == 1
        assert "stt=0.12s llm_ttft=n/a llm=n/a tts_ttfa=n/a e2e=n/a" in lines[0]
        assert f"response_key={fresh.response_key}" in lines[0]
    finally:
        service.unregister(new_conn_id)


def test_multiple_qwen_segments_keep_first_audio_timings_in_terminal_log(service, conn_id, monkeypatch, caplog):
    request = _queue_turn(service, conn_id)
    clock = [10.0]
    durations = iter([0.25, 0.75])
    generated_texts = []

    def generate_custom_voice(**kwargs):
        generated_texts.append(kwargs["text"])
        clock[0] += next(durations)
        yield SimpleNamespace(audio=np.full(512, 0.1, dtype=np.float32), sample_rate=16000)

    def load_model(self, model_name):
        self.model = SimpleNamespace(
            config=SimpleNamespace(tts_model_type="custom_voice"), generate_custom_voice=generate_custom_voice
        )

    monkeypatch.setattr(qwen3_tts_module, "platform", "darwin")
    monkeypatch.setattr(qwen3_tts_module, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(Qwen3TTSHandler, "_setup_mlx", load_model)
    monkeypatch.setattr(Qwen3TTSHandler, "warmup", lambda self: None)
    handler = object.__new__(Qwen3TTSHandler)
    handler.setup(Event())
    handler.queue_in = Queue()
    handler.turn_latency_store = service.turn_latency_store

    for start, text in [(10.0, "First sentence."), (20.0, "Second sentence.")]:
        clock[0] = start
        output = list(
            handler.process(
                TTSInput(
                    text=text,
                    response_key=request.response_key,
                    turn_id=request.turn_id,
                    turn_revision=request.turn_revision,
                    speech_stopped_at_s=8.0,
                )
            )
        )
        assert len(output) == 1
        assert np.any(output[0])

    assert generated_texts == ["First sentence.", "Second sentence."]
    service.dispatch_pipeline_event(conn_id, AssistantResponseDoneEvent(response_key=request.response_key))
    with caplog.at_level(logging.INFO, logger=LATENCY_LOGGER):
        service.finish_response(conn_id, response_key=request.response_key)
    lines = _latency_lines(caplog)
    assert len(lines) == 1
    assert "tts_ttfa=0.25s e2e=2.25s" in lines[0]


@pytest.mark.parametrize("followup_status", ["completed", "cancelled"])
def test_tool_followup_logs_distinct_responses_in_same_turn(service, conn_id, caplog, followup_status):
    request = _queue_turn(service, conn_id)
    call = RealtimeConversationItemFunctionCall(
        type="function_call", id="fc_lookup", call_id="call_lookup", name="lookup", arguments="{}"
    )
    request.runtime_config.chat.add_provisional_generation_items(request.response_key, [call])
    service.dispatch_pipeline_event(
        conn_id,
        AssistantOutputEvent(
            response_key=request.response_key,
            parts=[
                AssistantToolCallPart(
                    tool={"type": "function_call", **call.model_dump(include={"id", "call_id", "name", "arguments"})}
                )
            ],
        ),
    )
    service.dispatch_pipeline_event(
        conn_id, ResponseGenerationDoneEvent(response_key=request.response_key, call_ids=[call.call_id])
    )
    service.handle_conversation_item_create(
        conn_id,
        ConversationItemCreateEvent(
            type="conversation.item.create",
            item={"type": "function_call_output", "call_id": call.call_id, "output": "result"},
        ),
    )
    followup = service.text_prompt_queue.get_nowait()
    assert followup.response_key != request.response_key

    with caplog.at_level(logging.INFO, logger=LATENCY_LOGGER):
        service.finish_response(conn_id, response_key=request.response_key)
        created = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
        assert created.type == "response.created"
        assert service._state(conn_id).current_response_key == followup.response_key
        service.finish_response(conn_id, status=followup_status, response_key=followup.response_key)

    lines = _latency_lines(caplog)
    assert len(lines) == 2
    assert all("Turn turn_1 rev=0" in line for line in lines)
    assert f"response_key={request.response_key}" in lines[0]
    assert f"response_key={followup.response_key}" in lines[1]
    assert "stt=0.12s" in lines[0]
    assert "stt=n/a" in lines[1]
    assert "status=completed" in lines[0]
    assert f"status={followup_status}" in lines[1]
    assert service.turn_latency_store._trackers == {}
