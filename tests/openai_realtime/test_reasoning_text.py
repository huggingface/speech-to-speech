"""Reasoning accompanies committing output; it has no separate server lifecycle."""

from queue import Queue
from types import SimpleNamespace

import pytest
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
    ResponseReasoningTextDeltaEvent,
)

from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import AssistantOutputEvent, PipelineEvent
from speech_to_speech.pipeline.messages import AssistantReasoningPart, EndOfResponse, LLMResponseChunk, TTSInput
from tests.openai_realtime.test_speculative_turn_protocol import _Session
from tests.test_responses_api_language_model import _make_handler, _make_stream, _make_text_delta_event


@pytest.fixture
def session(runtime_config, should_listen):
    active = _Session(runtime_config, should_listen)
    runtime_config.session.extensions = ["speech_to_speech.reasoning.delta"]
    active.start_turn("turn_1")
    active.tracker.segment_finalized(1000)
    active.stop_speech("turn_1", 0, duration_s=1, audio_end_ms=1000)
    active.final("turn_1", 0, "A paused utterance")
    yield active
    active.close()


def _reasoning(text, sequence=0):
    return ResponseReasoningTextDeltaEvent(
        type="response.reasoning_text.delta",
        item_id="rs1",
        output_index=0,
        content_index=0,
        sequence_number=sequence,
        delta=text,
    )


def _handler(session, events, *, scope=None):
    handler = _make_handler(cancel_scope=scope)
    handler.speculative_turns = session.service.speculative_turns
    handler.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: _make_stream(events)))
    return handler, session.text_prompt_queue.get_nowait()


def _drive(session, handler, request):
    processor = LMOutputProcessor.__new__(LMOutputProcessor)
    processor.setup(speculative_turns=session.service.speculative_turns)
    chunks, outputs, events = [], [], []
    for chunk in handler.process(request):
        chunks.append(chunk)
        # The real pipeline/router discards stale generation output before it
        # reaches the service. Keep that cancellation boundary in this driver.
        if handler.cancel_scope is not None and handler.cancel_scope.is_stale(chunk.cancel_generation):
            continue
        for output in processor.process(chunk):
            outputs.append(output)
            if isinstance(output, PipelineEvent):
                events.extend(session.dispatch(output))
            elif isinstance(output, EndOfResponse) and not output.cleanup_only:
                events.extend(session.service.finish_response(session.conn_id, response_key=output.response_key))
    return chunks, outputs, events


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("mode", ["speculative", "untracked", "committed"])
def test_reasoning_waits_for_first_answer_on_every_setup(session, enabled, mode):
    session.runtime_config.session.extensions = ["speech_to_speech.reasoning.delta"] if enabled else []
    if mode == "untracked":
        session.service.speculative_turns = None
    elif mode == "committed":
        session.tracker.gate("turn_1", 0, commit=True)

    def stream():
        yield _reasoning("Think ")
        yield _reasoning("first.", 1)
        assert session.service._state(session.conn_id).current_response_id is None
        if mode == "speculative":
            assert not session.tracker.is_committed("turn_1", 0)
            assert session.tracker.can_reopen(1200)
        yield _make_text_delta_event("The answer.")
        yield _make_text_delta_event(" Another sentence.")

    handler, request = _handler(session, stream())
    handler.stream_batch_sentences = 1
    chunks, outputs, events = _drive(session, handler, request)
    answer_chunks = [chunk for chunk in chunks if isinstance(chunk, LLMResponseChunk)]
    assert answer_chunks
    assert all(chunk.text or chunk.tools for chunk in answer_chunks)
    assert [part.text for part in answer_chunks[0].parts if isinstance(part, AssistantReasoningPart)] == (
        ["Think first."] if enabled else []
    )
    assert not any(isinstance(part, AssistantReasoningPart) for chunk in answer_chunks[1:] for part in chunk.parts)
    wire = [event for event in events if event.type == "speech_to_speech.reasoning.delta"]
    assert [event.delta for event in wire] == (["Think first."] if enabled else [])
    if enabled:
        types = [event.type for event in events]
        assert (
            types.index("response.created")
            < types.index("speech_to_speech.reasoning.delta")
            < types.index("response.output_audio_transcript.delta")
        )
    assert all("Think first." not in output.text for output in outputs if isinstance(output, TTSInput))
    assert "Think first." not in str(session.runtime_config.chat.to_responses_api_chat())
    assert not any(chunk.reasoning for chunk in chunks if isinstance(chunk, EndOfResponse))


@pytest.mark.parametrize("interrupt", ["cancel", "reopen"])
def test_cancelled_or_superseded_thinking_never_reaches_server(session, interrupt):
    scope = CancelScope()

    def stream():
        yield _reasoning("Do not publish this.")
        assert session.tracker.can_reopen(1200)
        if interrupt == "cancel":
            scope.cancel()
        else:
            session.resume_turn("turn_1", 0, audio_start_ms=1200)
        yield _make_text_delta_event("Stale answer.")

    handler, request = _handler(session, stream(), scope=scope)
    chunks, outputs, events = _drive(session, handler, request)
    assert not any(isinstance(chunk, LLMResponseChunk) for chunk in chunks)
    assert all(not chunk.reasoning for chunk in chunks if isinstance(chunk, EndOfResponse))
    assert not any(event.type == "speech_to_speech.reasoning.delta" for event in events)
    assert not any(isinstance(output, AssistantOutputEvent) for output in outputs)
    assert not session.tracker.is_committed("turn_1", 0)


@pytest.mark.parametrize("enabled", [False, True])
def test_reasoning_only_completion_uses_the_normal_terminal(session, enabled):
    session.runtime_config.session.extensions = ["speech_to_speech.reasoning.delta"] if enabled else []
    handler, request = _handler(session, [_reasoning("Think "), _reasoning("first.", 1)])
    chunks, outputs, events = _drive(session, handler, request)
    assert not any(isinstance(chunk, LLMResponseChunk) for chunk in chunks)
    assert not any(isinstance(output, (AssistantOutputEvent, TTSInput)) for output in outputs)
    assert [part.text for chunk in chunks if isinstance(chunk, EndOfResponse) for part in chunk.reasoning] == (
        ["Think first."] if enabled else []
    )
    types = [event.type for event in events]
    if enabled:
        assert (
            types.index("response.created")
            < types.index("speech_to_speech.reasoning.delta")
            < types.index("response.done")
        )
    else:
        assert "speech_to_speech.reasoning.delta" not in types
    assert session.tracker.is_committed("turn_1", 0)


def test_reasoning_accompanies_an_early_tool_call_and_is_not_repeated(session):
    tool = ResponseFunctionToolCall(
        type="function_call", id="fc_test", call_id="call_test", name="lookup", arguments="{}"
    )
    handler, request = _handler(
        session,
        [
            _reasoning("Check first."),
            ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                output_index=1,
                sequence_number=1,
                item=tool,
            ),
        ],
    )
    chunks = list(handler.process(request))
    first = next(chunk for chunk in chunks if isinstance(chunk, LLMResponseChunk))
    assert first.tools and isinstance(first.parts[0], AssistantReasoningPart)
    side_channel = Queue()
    processor = LMOutputProcessor.__new__(LMOutputProcessor)
    processor.setup(speculative_turns=session.tracker, text_output_queue=side_channel)
    ordered = list(processor.process(first))
    ready = side_channel.get_nowait()
    assert ready.output_sequence == 0
    assert [part.text for part in ready.reasoning] == ["Check first."]
    events = session.dispatch(ready)
    types = [event.type for event in events]
    assert (
        types.index("response.created")
        < types.index("speech_to_speech.reasoning.delta")
        < types.index("response.function_call_arguments.done")
    )
    for output in ordered:
        if isinstance(output, PipelineEvent):
            assert not any(event.type == "speech_to_speech.reasoning.delta" for event in session.dispatch(output))
    assert session.tracker.is_committed("turn_1", 0)
