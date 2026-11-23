import pytest
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from openai.types.responses import ResponseFunctionToolCall

from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
from speech_to_speech.pipeline.messages import (
    AssistantToolCallPart,
    EndOfResponse,
    LLMResponseChunk,
    ResponsePrefetchTransaction,
    TTSInput,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.pipeline.turn_latency import TurnLatencyStore, TurnLatencyTracker
from tests.turns import reopen


def _processor(store):
    processor = object.__new__(LMOutputProcessor)
    processor.setup()
    processor.turn_latency_store = store
    return processor


def test_first_tts_uses_generation_origin_and_actual_forwarding_boundary(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("speech_to_speech.pipeline.turn_latency.perf_counter", lambda: clock[0])
    store = TurnLatencyStore()
    tracker = store.get_or_create_response("first", turn_id="turn_1", turn_revision=0)
    tracker.record_llm_start(10.0)
    tracker.record_llm_ttft(0.05)
    processor = _processor(store)
    outputs = processor.process(LLMResponseChunk(text="Hello.", response_key="first"))

    next(outputs)  # The text event precedes its TTS input.
    assert tracker.llm_first_tts_s is None
    clock[0] = 10.75
    assert isinstance(next(outputs), TTSInput)
    assert tracker.llm_first_tts_s == pytest.approx(0.75)
    assert tracker.llm_ttft_s == 0.05

    clock[0] = 12.0
    tracker.record_llm_start(11.0)  # A later origin cannot replace this generation.
    list(processor.process(LLMResponseChunk(text="Later.", response_key="first")))
    tracker.record_llm(2.0)
    assert tracker.llm_first_tts_s == pytest.approx(0.75)
    assert tracker.llm_s == 2.0
    assert tracker.metadata_payload(response_key="first", status="completed")["llm_first_tts_s"] == 0.75
    assert "llm_first_tts=0.75s" in tracker.format_log_line()


@pytest.mark.parametrize("kind", ["empty", "whitespace", "tool", "text-only", "terminal"])
def test_no_measurement_without_an_audio_text_batch(kind):
    store = TurnLatencyStore()
    tracker = store.get_or_create_response("first", turn_id="turn_1", turn_revision=0)
    tracker.record_llm_start(10.0)
    chunk = LLMResponseChunk(text="Hello.", response_key="first")
    if kind in ("empty", "whitespace"):
        chunk = LLMResponseChunk(text="" if kind == "empty" else " \n", response_key="first")
    elif kind == "tool":
        chunk = LLMResponseChunk(
            parts=[
                AssistantToolCallPart(
                    tool=ResponseFunctionToolCall(type="function_call", call_id="call_1", name="test", arguments="{}")
                )
            ],
            response_key="first",
        )
    elif kind == "text-only":
        chunk.response = RealtimeResponseCreateParams(output_modalities=["text"])
    elif kind == "terminal":
        chunk = EndOfResponse(response_key="first")

    outputs = list(_processor(store).process(chunk))

    assert not any(isinstance(output, TTSInput) for output in outputs)
    assert tracker.llm_first_tts_s is None


def test_missing_origin_and_discarded_tracker_stay_unavailable():
    store = TurnLatencyStore()
    tracker = store.get_or_create_response("first", turn_id="turn_1", turn_revision=0)
    processor = _processor(store)
    chunk = LLMResponseChunk(text="Hello.", response_key="first")
    list(processor.process(chunk))
    assert tracker.llm_first_tts_s is None

    tracker.record_llm_start(10.0)
    store.discard_response("first")
    list(processor.process(chunk))
    assert tracker.llm_first_tts_s is None
    assert store.get_response("first") is None
    assert store._trackers == {}


def test_stale_and_discarded_prefetch_outputs_do_not_record():
    store = TurnLatencyStore()
    tracker = store.get_or_create_response("first", turn_id="turn_1", turn_revision=0)
    tracker.record_llm_start(10.0)
    processor = _processor(store)
    turns = processor.speculative_turns = SpeculativeTurnTracker()
    turns.start_turn()
    reopen(turns, revision=0)
    chunk = LLMResponseChunk(text="Old.", response_key="first", turn_id="turn_1", turn_revision=0)
    assert list(processor.process(chunk)) == []
    assert tracker.llm_first_tts_s is None

    transaction = ResponsePrefetchTransaction()
    transaction.discard()
    processor.speculative_turns = None
    list(processor.process(chunk.model_copy(update={"prefetch_transaction": transaction})))
    assert tracker.llm_first_tts_s is None


def test_turn_reopened_between_text_event_and_tts_handoff_does_not_record():
    store = TurnLatencyStore()
    tracker = store.get_or_create_response("first", turn_id="turn_1", turn_revision=0)
    tracker.record_llm_start(10.0)
    processor = _processor(store)
    turns = processor.speculative_turns = SpeculativeTurnTracker()
    turns.start_turn()
    outputs = processor.process(LLMResponseChunk(text="Old.", response_key="first", turn_id="turn_1", turn_revision=0))
    next(outputs)
    reopen(turns, revision=0)

    # Keep existing forwarding policy; the downstream handler filters stale input.
    assert isinstance(next(outputs), TTSInput)
    assert tracker.llm_first_tts_s is None


def test_each_response_keeps_its_own_origin_and_tool_followup_does_not_inherit(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("speech_to_speech.pipeline.turn_latency.perf_counter", lambda: clock[0])
    store = TurnLatencyStore()
    first = store.get_or_create_response("first", turn_id="turn_1", turn_revision=0, session_id="session")
    first.record_llm_start(9.0)
    first.start_llm()
    first.record_llm(1.0)
    first.record_llm_first_tts()
    store.pop("first", session_id="session", tool_call_ids=["call_1"])
    second = store.get_or_create_response(
        "second", turn_id="turn_1", turn_revision=0, session_id="session", parent_call_id="call_1"
    )
    assert second.llm_first_tts_s is None
    assert second.llm_total_s == 1.0
    processor = _processor(store)
    list(processor.process(LLMResponseChunk(text="No origin yet.", response_key="second")))
    assert second.llm_first_tts_s is None

    second.record_llm_start(10.0)
    clock[0] = 10.2
    list(processor.process(LLMResponseChunk(text="Follow-up.", response_key="second")))
    assert second.llm_first_tts_s == pytest.approx(0.2)
    assert first.llm_first_tts_s == 1.0


def test_first_tts_elapsed_time_is_clamped_and_optional(monkeypatch):
    tracker = TurnLatencyTracker()
    tracker.record_llm_first_tts()
    assert tracker.llm_first_tts_s is None
    assert tracker.metadata_payload(response_key="empty", status="failed")["llm_first_tts_s"] is None
    monkeypatch.setattr("speech_to_speech.pipeline.turn_latency.perf_counter", lambda: 10.0)
    tracker.record_llm_start(11.0)
    tracker.record_llm_first_tts()
    assert tracker.llm_first_tts_s == 0.0
