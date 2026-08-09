from queue import Queue
from threading import Event, Thread

import pytest
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from openai.types.responses import ResponseFunctionToolCall

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
from speech_to_speech.pipeline.events import (
    AssistantResponseDoneEvent,
    AssistantTextEvent,
    ResponseFailedEvent,
    TokenUsageEvent,
)
from speech_to_speech.pipeline.messages import (
    AUDIO_RESPONSE_DONE,
    PIPELINE_END,
    AssistantTextPart,
    AssistantToolCallPart,
    AudioOutput,
    EndOfResponse,
    LLMResponseChunk,
    TokenUsage,
    TTSInput,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker


def _processor(
    tracker: SpeculativeTurnTracker,
    text_output_queue: Queue | None = None,
) -> LMOutputProcessor:
    processor = LMOutputProcessor.__new__(LMOutputProcessor)
    processor.setup(text_output_queue=text_output_queue, speculative_turns=tracker)
    return processor


def _tracked_processor(revision: int = 0) -> tuple[SpeculativeTurnTracker, LMOutputProcessor]:
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", revision)
    return tracker, _processor(tracker)


def test_stale_end_of_response_is_dropped_or_keyed_for_cleanup():
    _, processor = _tracked_processor(revision=1)

    assert list(processor.process(EndOfResponse(turn_id="turn_1", turn_revision=0))) == []
    outputs = list(
        processor.process(
            EndOfResponse(
                response_key="response_1",
                turn_id="turn_1",
                turn_revision=0,
                cancel_generation=7,
            )
        )
    )

    assert len(outputs) == 1
    terminal = outputs[0]
    assert isinstance(terminal, EndOfResponse)
    assert (terminal.response_key, terminal.cancel_generation, terminal.turn_id, terminal.cleanup_only) == (
        "response_1",
        7,
        None,
        True,
    )


def test_latest_end_of_response_follows_ordered_done_event():
    _, processor = _tracked_processor(revision=1)

    outputs = list(processor.process(EndOfResponse(turn_id="turn_1", turn_revision=1)))

    assert [type(output) for output in outputs] == [AssistantResponseDoneEvent, EndOfResponse]
    assert all(output.turn_id == "turn_1" and output.turn_revision == 1 for output in outputs)


def test_failed_response_event_precedes_terminal_and_keeps_identity():
    _, processor = _tracked_processor()

    outputs = list(
        processor.process(
            EndOfResponse(
                error="provider rejected input",
                response_key="response_1",
                turn_id="turn_1",
                turn_revision=0,
                cancel_generation=7,
            )
        )
    )

    assert [type(output) for output in outputs] == [ResponseFailedEvent, EndOfResponse]
    assert all(output.response_key == "response_1" and output.cancel_generation == 7 for output in outputs)


def test_token_usage_keeps_generation_and_response_identity():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    text_output_queue = Queue()
    processor = _processor(tracker, text_output_queue)

    outputs = list(
        processor.process(
            TokenUsage(
                input_tokens=11,
                output_tokens=7,
                response_key="response_1",
                turn_id="turn_1",
                turn_revision=0,
                cancel_generation=7,
            )
        )
    )

    assert outputs == []
    event = text_output_queue.get_nowait()
    assert isinstance(event, TokenUsageEvent)
    assert (event.response_key, event.cancel_generation, event.input_tokens, event.output_tokens) == (
        "response_1",
        7,
        11,
        7,
    )


@pytest.mark.parametrize(
    ("modalities", "text", "expect_tts"),
    [
        (["text"], "hello", False),
        (["audio"], "hello", True),
        (["audio"], "   \n", False),
        ([], "hello", True),
    ],
)
def test_text_event_precedes_optional_tts_input(modalities, text, expect_tts):
    _, processor = _tracked_processor()

    outputs = list(
        processor.process(
            LLMResponseChunk(
                text=text,
                turn_id="turn_1",
                turn_revision=0,
                cancel_generation=7,
                response=RealtimeResponseCreateParams(output_modalities=modalities),
            )
        )
    )

    assert isinstance(outputs[0], AssistantTextEvent)
    assert outputs[0].text == text
    assert outputs[0].cancel_generation == 7
    assert [isinstance(output, TTSInput) for output in outputs] == [False, *([True] if expect_tts else [])]


def test_ordered_parts_and_tts_inputs_share_one_queue():
    _, processor = _tracked_processor()
    first_tool = ResponseFunctionToolCall(type="function_call", call_id="call_1", name="first", arguments="{}")
    second_tool = ResponseFunctionToolCall(type="function_call", call_id="call_2", name="second", arguments="{}")

    outputs = list(
        processor.process(
            LLMResponseChunk(
                parts=[
                    AssistantToolCallPart(tool=first_tool),
                    AssistantTextPart(text="between"),
                    AssistantToolCallPart(tool=second_tool),
                    AssistantTextPart(text="after"),
                ],
                turn_id="turn_1",
                turn_revision=0,
            )
        )
    )

    assert [type(output) for output in outputs] == [
        AssistantTextEvent,
        AssistantTextEvent,
        TTSInput,
        AssistantTextEvent,
        AssistantTextEvent,
        TTSInput,
    ]
    events = [output for output in outputs if isinstance(output, AssistantTextEvent)]
    assert [event.parts[0].type for event in events] == ["tool_call", "text", "tool_call", "text"]
    assert [output.text for output in outputs if isinstance(output, TTSInput)] == ["between", "after"]


def test_audio_output_keeps_response_identity():
    handler = object.__new__(BaseHandler)

    queued = handler.output_for_queue(b"audio", TTSInput(text="hello", response_key="response_1"))

    assert isinstance(queued, AudioOutput)
    assert (queued.audio, queued.response_key) == (b"audio", "response_1")


def test_tts_handler_forwards_response_events_without_processing_them():
    queue_in, queue_out = Queue(), Queue()
    handler = BaseHandler(Event(), queue_in, queue_out)
    event = AssistantTextEvent(text="before")
    queue_in.put(event)
    queue_in.put(PIPELINE_END)

    handler.run()

    assert queue_out.get_nowait() is event
    assert queue_out.get_nowait() == PIPELINE_END


def test_stale_response_cleanup_reaches_the_audio_queue():
    handler = object.__new__(BaseHandler)

    queued = handler.output_for_queue(
        AUDIO_RESPONSE_DONE,
        EndOfResponse(response_key="response_1", cancel_generation=7, cleanup_only=True),
    )

    assert isinstance(queued, AudioOutput)
    assert (queued.audio, queued.response_key, queued.cancel_generation, queued.cleanup_only) == (
        AUDIO_RESPONSE_DONE,
        "response_1",
        7,
        True,
    )


def _process_in_thread(processor: LMOutputProcessor) -> tuple[Event, list, Thread]:
    done = Event()
    outputs = []

    def run_processor():
        outputs.extend(processor.process(LLMResponseChunk(text="hello", turn_id="turn_1", turn_revision=0)))
        done.set()

    thread = Thread(target=run_processor)
    thread.start()
    return done, outputs, thread


def test_pending_reopen_holds_output_until_cancelled():
    tracker, processor = _tracked_processor()
    candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)
    done, outputs, thread = _process_in_thread(processor)

    assert not done.wait(0.05)
    tracker.cancel_reopen_candidate("turn_1", candidate_revision)
    assert done.wait(1.0)
    thread.join(timeout=1.0)

    assert [type(output) for output in outputs] == [AssistantTextEvent, TTSInput]


def test_reopen_grace_holds_output_until_elapsed():
    tracker, processor = _tracked_processor()
    tracker.start_reopen_grace("turn_1", 0, grace_s=0.08)
    done, outputs, thread = _process_in_thread(processor)

    assert not done.wait(0.02)
    assert done.wait(1.0)
    thread.join(timeout=1.0)

    assert [type(output) for output in outputs] == [AssistantTextEvent, TTSInput]


def test_confirmed_reopen_drops_held_output():
    tracker, processor = _tracked_processor()
    candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)
    done, outputs, thread = _process_in_thread(processor)

    assert not done.wait(0.05)
    assert tracker.confirm_reopen_candidate("turn_1", 0, candidate_revision)
    assert done.wait(1.0)
    thread.join(timeout=1.0)

    assert outputs == []
