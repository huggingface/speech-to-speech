import pytest
from openai.types.realtime import ResponseCreateEvent

from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
from speech_to_speech.pipeline.events import (
    AssistantOutputEvent,
    AssistantResponseDoneEvent,
    AssistantToolCallReadyEvent,
)
from speech_to_speech.pipeline.messages import AssistantReasoningPart, AssistantToolCallPart, LLMResponseChunk
from tests.openai_realtime.test_speculative_turn_protocol import _Session


@pytest.fixture
def speculative_session(runtime_config, should_listen):
    session = _Session(runtime_config, should_listen)
    runtime_config.session.extensions = ["speech_to_speech.reasoning.delta"]
    session.start_turn("turn_1")
    session.tracker.segment_finalized(1000)
    session.stop_speech("turn_1", 0, duration_s=1, audio_end_ms=1000)
    session.final("turn_1", 0, "A paused utterance")
    yield session
    session.close()


def _reasoning(request, text="Private reasoning.", sequence=0):
    return AssistantOutputEvent(
        parts=[AssistantReasoningPart(text=text)],
        response_key=request.response_key,
        turn_id=request.turn_id,
        turn_revision=request.turn_revision,
        output_sequence=sequence,
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_reasoning_alone_preserves_reopen_until_answer_commits(speculative_session, enabled):
    s = speculative_session
    s.runtime_config.session.extensions = ["speech_to_speech.reasoning.delta"] if enabled else []
    request = s.text_prompt_queue.get_nowait()
    state = s.service._state(s.conn_id)
    assert s.tracker.can_reopen(1200)
    assert s.dispatch(_reasoning(request)) == []
    assert not s.tracker.is_committed("turn_1", 0)
    assert s.tracker.can_reopen(1200)
    assert state.current_response_id is None
    events = s.dispatch(
        AssistantOutputEvent(
            text="The answer.",
            response_key=request.response_key,
            turn_id="turn_1",
            turn_revision=0,
            output_sequence=1,
        )
    )
    assert s.tracker.is_committed("turn_1", 0)
    assert not s.tracker.can_reopen(1200)
    types = [e.type for e in events]
    if enabled:
        assert (
            types.index("response.created")
            < types.index("speech_to_speech.reasoning.delta")
            < types.index("response.output_audio_transcript.delta")
        )
        assert next(e.delta for e in events if e.type == "speech_to_speech.reasoning.delta") == "Private reasoning."
        live = s.dispatch(_reasoning(request, "After the commit.", sequence=2))
        assert [e.delta for e in live if e.type == "speech_to_speech.reasoning.delta"] == ["After the commit."]
    else:
        assert "speech_to_speech.reasoning.delta" not in types
    assert state.pending_reasoning == {}


def test_reopen_discards_buffered_reasoning_from_the_previous_revision(speculative_session):
    s = speculative_session
    request = s.text_prompt_queue.get_nowait()
    assert s.dispatch(_reasoning(request, "Superseded reasoning.")) == []
    s.resume_turn("turn_1", 0, audio_start_ms=1200)
    assert s.service._state(s.conn_id).pending_reasoning == {}

    s.service.close_pending_responses(s.conn_id)
    s.tracker.segment_finalized(2000)
    s.stop_speech("turn_1", 1, duration_s=2, audio_end_ms=2000)
    s.final("turn_1", 1, "The complete utterance")
    revised = s.text_prompt_queue.get_nowait()
    assert s.dispatch(_reasoning(revised, "Current reasoning.")) == []
    events = s.dispatch(
        AssistantOutputEvent(
            text="The answer.",
            response_key=revised.response_key,
            turn_id="turn_1",
            turn_revision=1,
            output_sequence=1,
        )
    )
    assert [e.delta for e in events if e.type == "speech_to_speech.reasoning.delta"] == ["Current reasoning."]
    assert s.chat_user_turns() == ["The complete utterance"]


@pytest.mark.parametrize("close", ["pending", "key"])
def test_cancelled_pending_reasoning_is_discarded(speculative_session, close):
    s = speculative_session
    request = s.text_prompt_queue.get_nowait()
    s.dispatch(_reasoning(request))
    if close == "pending":
        s.service.close_pending_responses(s.conn_id)
    else:
        s.service.close_response_key(s.conn_id, request.response_key)
    assert s.service._state(s.conn_id).pending_reasoning == {}

    # A cancelled implicit response never opened a public response, but its
    # consumed reasoning positions must not make the next sequence 0 stale.
    next_request = request.model_copy(update={"response_key": "next-response"})
    s.service._state(s.conn_id).mark_response_pending(next_request.response_key)
    assert s.dispatch(_reasoning(next_request, "Next response.")) == []
    assert s.service._state(s.conn_id).pending_reasoning


def test_reasoning_only_completion_flushes_at_the_existing_terminal_commit(speculative_session):
    s = speculative_session
    request = s.text_prompt_queue.get_nowait()
    assert s.dispatch(_reasoning(request)) == []
    events = s.dispatch(
        AssistantResponseDoneEvent(
            response_key=request.response_key,
            turn_id="turn_1",
            turn_revision=0,
        )
    )
    types = [e.type for e in events]
    assert types.index("response.created") < types.index("speech_to_speech.reasoning.delta")
    assert s.tracker.is_committed("turn_1", 0)
    assert s.service._state(s.conn_id).pending_reasoning == {}


def test_buffered_reasoning_does_not_block_an_early_tool_call(speculative_session):
    from openai.types.responses import ResponseFunctionToolCall

    s = speculative_session
    request = s.text_prompt_queue.get_nowait()
    tool = AssistantToolCallReadyEvent(
        part=AssistantToolCallPart(
            tool=ResponseFunctionToolCall(
                type="function_call",
                id="fc_test",
                call_id="call_test",
                name="lookup",
                arguments="{}",
            )
        ),
        response_key=request.response_key,
        turn_id="turn_1",
        turn_revision=0,
        output_sequence=1,
    )
    assert s.dispatch(tool) == []
    events = s.dispatch(_reasoning(request))
    types = [e.type for e in events]
    assert types.index("speech_to_speech.reasoning.delta") < types.index("response.function_call_arguments.done")
    assert s.tracker.is_committed("turn_1", 0)
    assert s.service._state(s.conn_id).pending_reasoning == {}


@pytest.mark.parametrize("enabled", [False, True])
def test_reasoning_is_an_opt_in_response_scoped_extension(service, conn_id, runtime_config, enabled):
    runtime_config.session.extensions = ["speech_to_speech.reasoning.delta"] if enabled else []
    service.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
    state = service._state(conn_id)
    response_id = state.current_response_id
    processor = LMOutputProcessor.__new__(LMOutputProcessor)
    processor.setup()
    outputs = list(
        processor.process(
            LLMResponseChunk(
                parts=[AssistantReasoningPart(text="Think first.", kind="summary")],
                response_key=state.current_response_key,
            )
        )
    )
    events = [
        out
        for event in outputs
        if isinstance(event, AssistantOutputEvent)
        for out in service.dispatch_pipeline_event(conn_id, event)
    ]
    wire = [event.model_dump() for event in events if event.type == "speech_to_speech.reasoning.delta"]
    assert len(wire) == int(enabled)
    if enabled:
        assert wire[0]["response_id"] == response_id
        assert wire[0]["delta"] == "Think first."
        assert wire[0]["kind"] == "summary"
    assert not any(
        event.type in {"response.output_audio_transcript.delta", "response.output_text.delta"} for event in events
    )
    assert state.pending_text_outputs == []
