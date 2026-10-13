"""Speech E2E belongs to one first-audio answer, not every later response."""

import json
from types import SimpleNamespace

import pytest
from openai.types.realtime import ConversationItemCreateEvent, ResponseCreateEvent
from openai.types.realtime.conversation_item import RealtimeConversationItemFunctionCall

import speech_to_speech.LLM.language_model as llm_module
from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
from speech_to_speech.pipeline.events import (
    AssistantOutputEvent,
    AssistantResponseDoneEvent,
    ResponseGenerationDoneEvent,
    TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.messages import AssistantToolCallPart, LLMResponseChunk, TTSInput
from speech_to_speech.pipeline.turn_latency import TURN_LATENCY_METADATA_KEY
from speech_to_speech.TTS import openai_compatible_handler as tts_module
from tests.test_openai_tts_handler import _FakeSpeechOperation, _openai_tts_handler
from tests.test_response_overrides import _RecordingLocalHandler


def speech(service, conn_id, *, automatic=True, revision=0):
    if not automatic:
        service._state(conn_id).runtime_config.session.audio.input.turn_detection = {
            "type": "server_vad",
            "create_response": False,
        }
    service.dispatch_pipeline_event(
        conn_id,
        TranscriptionCompletedEvent(
            transcript="Hello", turn_id="turn_1", turn_revision=revision, speech_stopped_at_s=10.0
        ),
    )
    if automatic:
        return service.text_prompt_queue.get_nowait()
    return create(service, conn_id)


def create(service, conn_id, response=None):
    result = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create", response=response))
    assert result.type == "response.created"
    service.response.mark_response_created_sent(conn_id, service._state(conn_id).current_response_key)
    return service.text_prompt_queue.get_nowait()


def audio(service, conn_id, request, monkeypatch, clock):
    handler = _openai_tts_handler(monkeypatch)
    handler.turn_latency_store = service.turn_latency_store
    monkeypatch.setattr(tts_module, "perf_counter", lambda: clock[0])
    _FakeSpeechOperation.startup_action = lambda: clock.__setitem__(0, clock[0] + 0.25)
    processor = LMOutputProcessor.__new__(LMOutputProcessor)
    processor.setup(text_output_queue=service.text_prompt_queue)
    chunk = LLMResponseChunk(
        text="Hello",
        runtime_config=request.runtime_config,
        response=request.response,
        response_key=request.response_key,
        turn_id=request.turn_id,
        turn_revision=request.turn_revision,
        speech_stopped_at_s=request.speech_stopped_at_s,
    )
    for item in processor.process(chunk):
        if isinstance(item, TTSInput):
            outputs = list(handler.process(item))
            assert outputs
            for output in outputs:
                service.encode_audio_chunk(conn_id, output.tobytes(), request.response_key)


def finish(service, conn_id, request, status="completed"):
    events = service.finish_response(conn_id, response_key=request.response_key, status=status)
    done = next(event for event in events if event.type == "response.done")
    return json.loads(done.response.metadata[TURN_LATENCY_METADATA_KEY])


@pytest.mark.parametrize("new_input", [False, True, "inline"])
def test_independent_create_after_idle_has_stage_timings_without_speech_e2e(service, conn_id, monkeypatch, new_input):
    clock = [11.0]
    original = speech(service, conn_id)
    audio(service, conn_id, original, monkeypatch, clock)
    assert finish(service, conn_id, original)["e2e_s"] == pytest.approx(1.25)
    clock[0] = 1011.0
    item = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Again"}]}
    if new_input is True:
        service.handle_conversation_item_create(
            conn_id, ConversationItemCreateEvent(type="conversation.item.create", item=item)
        )
    independent = create(service, conn_id, {"input": [item]} if new_input == "inline" else None)
    assert independent.speech_stopped_at_s is None
    # Run the shared generation path with only model output/clock stubbed.
    handler = object.__new__(_RecordingLocalHandler)
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.enable_lang_prompt = False
    handler.compactor = None
    handler.tokenizer = SimpleNamespace(encode=lambda _text: [])
    handler.turn_latency_store = service.turn_latency_store
    monkeypatch.setattr(llm_module, "perf_counter", lambda: clock[0])

    def generate(*_args, **_kwargs):
        clock[0] += 0.5
        yield LLMResponseChunk(text="Hello", runtime_config=independent.runtime_config)

    monkeypatch.setattr(handler, "_generate", generate)
    assert list(handler.process(independent))
    audio(service, conn_id, independent, monkeypatch, clock)
    record = finish(service, conn_id, independent)
    assert record["e2e_s"] is None
    assert record["llm_s"] == 0.5
    assert record["tts_ttfa_s"] == pytest.approx(0.25)
    assert record["source"] == "client"


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("already_spoke", [False, True])
def test_tool_chain_keeps_delay_only_until_first_audio(service, conn_id, monkeypatch, prefetch, already_spoke):
    clock = [11.0]
    original = speech(service, conn_id)
    call = RealtimeConversationItemFunctionCall(
        type="function_call", id="fc_lookup", call_id="call_lookup", name="lookup", arguments="{}"
    )
    original.runtime_config.chat.add_provisional_generation_items(original.response_key, [call])
    service.dispatch_pipeline_event(
        conn_id,
        AssistantOutputEvent(
            response_key=original.response_key,
            turn_id=original.turn_id,
            turn_revision=original.turn_revision,
            parts=[AssistantToolCallPart(tool=call.model_dump())],
        ),
    )
    if prefetch:
        service.dispatch_pipeline_event(
            conn_id, ResponseGenerationDoneEvent(response_key=original.response_key, call_ids=[call.call_id])
        )
    service.handle_conversation_item_create(
        conn_id,
        ConversationItemCreateEvent(
            type="conversation.item.create",
            item={"type": "function_call_output", "call_id": call.call_id, "output": "result"},
        ),
    )
    if already_spoke:
        # A prefetch can already exist before its originating TTS yields audio.
        audio(service, conn_id, original, monkeypatch, clock)
    record = finish(service, conn_id, original)
    followup = create(service, conn_id) if not prefetch else None
    if prefetch:
        result = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
        assert result.type == "response.created"
        service.response.mark_response_created_sent(conn_id, service._state(conn_id).current_response_key)
        followup = service.text_prompt_queue.get_nowait()
    clock[0] = 31.0
    audio(service, conn_id, followup, monkeypatch, clock)
    service.dispatch_pipeline_event(
        conn_id,
        AssistantResponseDoneEvent(
            response_key=followup.response_key,
            input_tool_call_ids=[call.call_id],
            turn_id=followup.turn_id,
            turn_revision=followup.turn_revision,
        ),
    )
    followup_record = finish(service, conn_id, followup)
    assert record["e2e_s"] == (pytest.approx(1.25) if already_spoke else None)
    assert followup_record["e2e_s"] == (None if already_spoke else pytest.approx(21.25))
    assert followup_record["source"] == "tool"
    assert followup_record["tts_ttfa_s"] == pytest.approx(0.25)
    again = create(service, conn_id)
    assert again.speech_stopped_at_s is None
    assert service.turn_latency_store.get_response(again.response_key).source == "client"


@pytest.mark.parametrize("pending", [False, True])
def test_cancelled_speech_does_not_anchor_independent_retry(service, conn_id, monkeypatch, pending):
    original = speech(service, conn_id)
    if pending:
        service.close_pending_responses(conn_id)
    else:
        service.response._ensure_response(conn_id, original.response_key)
        service.handle_response_cancel(conn_id)
    retry = create(service, conn_id)
    assert retry.speech_stopped_at_s is None
    audio(service, conn_id, retry, monkeypatch, [1011.0])
    assert finish(service, conn_id, retry)["e2e_s"] is None


def test_session_reuse_and_new_revision_have_fresh_speech_anchor(service, conn_id, runtime_config, monkeypatch):
    original = speech(service, conn_id)
    audio(service, conn_id, original, monkeypatch, [11.0])
    finish(service, conn_id, original)
    revised = speech(service, conn_id, revision=1)
    audio(service, conn_id, revised, monkeypatch, [12.0])
    assert finish(service, conn_id, revised)["e2e_s"] == pytest.approx(2.25)
    service.unregister(conn_id)
    reused = service.register()
    try:
        service._state(reused).runtime_config = runtime_config.model_copy(
            update={"chat": runtime_config.chat.copy(deep=True)}
        )
        fresh = speech(service, reused)
        audio(service, reused, fresh, monkeypatch, [11.0])
        assert finish(service, reused, fresh)["e2e_s"] == pytest.approx(1.25)
    finally:
        service.unregister(reused)


@pytest.mark.parametrize("automatic", [False, True])
def test_fresh_speech_response_keeps_anchor_and_source(service, conn_id, monkeypatch, automatic):
    request = speech(service, conn_id, automatic=automatic)
    assert request.speech_stopped_at_s == 10.0
    audio(service, conn_id, request, monkeypatch, [11.0])
    record = finish(service, conn_id, request)
    assert record["e2e_s"] == pytest.approx(1.25)
    assert record["source"] == "speech"


@pytest.mark.parametrize("status", ["completed", "cancelled", "failed", "incomplete"])
def test_terminal_answer_without_audio_closes_anchor(service, conn_id, status):
    request = speech(service, conn_id)
    service.response._ensure_response(conn_id, request.response_key)
    assert finish(service, conn_id, request, status)["e2e_s"] is None
    assert create(service, conn_id).speech_stopped_at_s is None


def test_cancel_after_first_audio_cannot_record_again(service, conn_id, monkeypatch):
    request = speech(service, conn_id)
    audio(service, conn_id, request, monkeypatch, [11.0])
    record = finish(service, conn_id, request, "cancelled")
    assert record["e2e_s"] == pytest.approx(1.25)
    assert create(service, conn_id).speech_stopped_at_s is None


def test_discarding_prefetch_does_not_abandon_tool_speech_anchor(service, conn_id, monkeypatch):
    request = speech(service, conn_id)
    call = RealtimeConversationItemFunctionCall(
        type="function_call", id="fc_lookup", call_id="call_lookup", name="lookup", arguments="{}"
    )
    request.runtime_config.chat.add_provisional_generation_items(request.response_key, [call])
    service.dispatch_pipeline_event(
        conn_id,
        AssistantOutputEvent(
            response_key=request.response_key,
            turn_id=request.turn_id,
            turn_revision=request.turn_revision,
            parts=[AssistantToolCallPart(tool=call.model_dump())],
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
    discarded = service.text_prompt_queue.get_nowait()
    finish(service, conn_id, request)
    replacement = create(service, conn_id, {"instructions": "Be brief"})
    assert discarded.prefetch_transaction.discarded
    assert replacement.speech_stopped_at_s == 10.0
    audio(service, conn_id, replacement, monkeypatch, [31.0])
    assert finish(service, conn_id, replacement)["e2e_s"] == pytest.approx(21.25)


@pytest.mark.parametrize("supplied_text", ["conversation_item", "response_input"])
def test_rebuilt_tool_prefetch_after_new_text_has_no_speech_e2e(service, conn_id, monkeypatch, supplied_text):
    request = speech(service, conn_id)
    call = RealtimeConversationItemFunctionCall(
        type="function_call", id="fc_lookup", call_id="call_lookup", name="lookup", arguments="{}"
    )
    request.runtime_config.chat.add_provisional_generation_items(request.response_key, [call])
    service.dispatch_pipeline_event(
        conn_id,
        AssistantOutputEvent(
            response_key=request.response_key,
            turn_id=request.turn_id,
            turn_revision=request.turn_revision,
            parts=[AssistantToolCallPart(tool=call.model_dump())],
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
    old_prefetch = service.text_prompt_queue.get_nowait()
    finish(service, conn_id, request)
    clock = [1011.0]
    item = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "Forget weather. Tell a joke."}],
    }
    if supplied_text == "conversation_item":
        service.handle_conversation_item_create(
            conn_id, ConversationItemCreateEvent(type="conversation.item.create", item=item)
        )
        rebuilt = service.text_prompt_queue.get_nowait()
        assert old_prefetch.prefetch_transaction.discarded
        # Generation ownership stays on its original call; timing does not.
        assert (rebuilt.turn_id, rebuilt.turn_revision) == (request.turn_id, request.turn_revision)
        created = service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
        assert created.type == "response.created"
        service.response.mark_response_created_sent(conn_id, rebuilt.response_key)
    else:
        rebuilt = create(service, conn_id, {"input": [item]})
    assert rebuilt.speech_stopped_at_s is None
    audio(service, conn_id, rebuilt, monkeypatch, clock)
    record = finish(service, conn_id, rebuilt)
    assert record["source"] == "client"
    assert record["e2e_s"] is None
    assert record["tts_ttfa_s"] == pytest.approx(0.25)


def test_typed_tool_chain_exports_stage_measurements_without_speech(service, conn_id, monkeypatch):
    service.handle_conversation_item_create(
        conn_id,
        ConversationItemCreateEvent(
            type="conversation.item.create",
            item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Weather?"}]},
        ),
    )
    request = create(service, conn_id)
    call = RealtimeConversationItemFunctionCall(
        type="function_call", id="fc_lookup", call_id="call_lookup", name="lookup", arguments="{}"
    )
    request.runtime_config.chat.add_provisional_generation_items(request.response_key, [call])
    service.dispatch_pipeline_event(
        conn_id,
        AssistantOutputEvent(response_key=request.response_key, parts=[AssistantToolCallPart(tool=call.model_dump())]),
    )
    finish(service, conn_id, request)
    service.handle_conversation_item_create(
        conn_id,
        ConversationItemCreateEvent(
            type="conversation.item.create",
            item={"type": "function_call_output", "call_id": call.call_id, "output": "result"},
        ),
    )
    followup = create(service, conn_id)
    assert followup.turn_id is None and followup.speech_stopped_at_s is None
    tracker = service.turn_latency_store.get_response(followup.response_key)
    tracker.record_llm(0.5)
    audio(service, conn_id, followup, monkeypatch, [1011.0])
    record = finish(service, conn_id, followup)
    assert record["source"] == "tool"
    assert record["e2e_s"] is None
    assert record["llm_s"] == 0.5
    assert record["tts_ttfa_s"] == pytest.approx(0.25)
