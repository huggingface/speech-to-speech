from queue import Queue
from threading import Event, Thread
from types import SimpleNamespace

import pytest
from openai.types.realtime import RealtimeSessionCreateRequest
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from openai.types.responses import ResponseFunctionToolCall
from pydantic import ValidationError

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.LLM.chat import Chat
from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
from speech_to_speech.pipeline import language_detection
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import (
    AssistantOutputEvent,
    AssistantResponseDoneEvent,
    AssistantToolCallReadyEvent,
    ResponseFailedEvent,
    ResponseGenerationDoneEvent,
    TokenUsageEvent,
)
from speech_to_speech.pipeline.messages import (
    AUDIO_RESPONSE_DONE,
    PIPELINE_END,
    AssistantTextPart,
    AssistantToolCallPart,
    AudioOutput,
    EndOfResponse,
    GenerateResponseRequest,
    LLMResponseChunk,
    ResponsePrefetchTransaction,
    TokenUsage,
    TTSInput,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from tests.turns import reopen


def _processor(tracker: SpeculativeTurnTracker) -> LMOutputProcessor:
    processor = LMOutputProcessor.__new__(LMOutputProcessor)
    processor.setup(speculative_turns=tracker)
    return processor


def _tracked_processor(revision: int = 0) -> tuple[SpeculativeTurnTracker, LMOutputProcessor]:
    tracker = SpeculativeTurnTracker()
    tracker.start_turn()
    for base_revision in range(revision):
        reopen(tracker, revision=base_revision)
    return tracker, _processor(tracker)


def _session_with_language(language: str) -> RuntimeConfig:
    return RuntimeConfig(
        session=RealtimeSessionCreateRequest(
            type="realtime",
            audio={"input": {"transcription": {"language": language}}},
        )
    )


def _spoken_tts_input(processor: LMOutputProcessor, config: RuntimeConfig, text: str) -> TTSInput:
    outputs = list(
        processor.process(
            LLMResponseChunk(
                text=text,
                language_code="es",
                runtime_config=config,
                turn_id="turn_1",
                turn_revision=0,
            )
        )
    )
    return next(item for item in outputs if isinstance(item, TTSInput))


def test_auto_tts_language_uses_assistant_text_instead_of_stt_language():
    _, processor = _tracked_processor()
    config = _session_with_language("auto")

    tts_input = _spoken_tts_input(processor, config, "I can help you find the train station in London.")

    assert tts_input.tts_language_code == "en"


def test_short_auto_reply_uses_prior_assistant_language_only_in_same_session():
    _, processor = _tracked_processor()
    config = _session_with_language("auto")

    first = _spoken_tts_input(processor, config, "Sure.")
    assert first.tts_language_code is None
    list(processor.process(EndOfResponse(turn_id="turn_1", turn_revision=0)))
    longer = _spoken_tts_input(processor, config, "I can help you find the train station in London.")
    assert longer.tts_language_code == "en"
    list(processor.process(EndOfResponse(turn_id="turn_1", turn_revision=0)))
    later = _spoken_tts_input(processor, config, "Sure.")

    assert later.tts_language_code == "en"

    list(processor.process(EndOfResponse(turn_id="turn_1", turn_revision=0)))
    new_session = _session_with_language("auto")
    assert _spoken_tts_input(processor, new_session, "Sure.").tts_language_code is None


def test_failed_auto_response_does_not_become_short_reply_fallback():
    _, processor = _tracked_processor()
    config = _session_with_language("auto")

    assert (
        _spoken_tts_input(processor, config, "I can help you find the train station in London.").tts_language_code
        == "en"
    )
    list(processor.process(EndOfResponse(turn_id="turn_1", turn_revision=0, error="generation failed")))

    assert config.last_assistant_language is None
    assert _spoken_tts_input(processor, config, "Sure.").tts_language_code is None


def test_later_assistant_part_seeds_future_auto_fallback_without_changing_current_voice():
    _, processor = _tracked_processor()
    config = _session_with_language("auto")

    first = _spoken_tts_input(processor, config, "Sure.")
    assert first.tts_language_code is None
    later = _spoken_tts_input(processor, config, "I can help you find the train station in London.")
    assert later.tts_language_code is None

    list(processor.process(EndOfResponse(turn_id="turn_1", turn_revision=0)))

    assert config.last_assistant_language == "en"
    assert _spoken_tts_input(processor, config, "Okay.").tts_language_code == "en"


def test_short_streamed_parts_combine_for_future_auto_fallback():
    _, processor = _tracked_processor()
    config = _session_with_language("auto")

    first = _spoken_tts_input(processor, config, "I can help you")
    assert first.tts_language_code is None
    second = _spoken_tts_input(processor, config, "find the station.")
    assert second.tts_language_code is None

    list(processor.process(EndOfResponse(turn_id="turn_1", turn_revision=0)))

    assert config.last_assistant_language == "en"
    assert _spoken_tts_input(processor, config, "Sure.").tts_language_code == "en"


def test_explicit_tts_language_overrides_assistant_detection():
    _, processor = _tracked_processor()
    config = _session_with_language("es")

    tts_input = _spoken_tts_input(processor, config, "I can help you find the train station in London.")

    assert tts_input.tts_language_code == "es"


def _detecting_processor() -> LMOutputProcessor:
    _, processor = _tracked_processor()
    processor.detect_llm_output_language = True
    return processor


@pytest.mark.parametrize(
    ("selected", "text", "expected"),
    [
        ("en", "Puedo ayudarte a encontrar la estación de tren más cercana.", "es"),
        ("es", "I can help you find the train station in London.", "en"),
    ],
)
def test_detected_assistant_language_overrides_named_selection_for_tts_only(selected, text, expected):
    processor = _detecting_processor()
    config = _session_with_language(selected)

    tts_input = _spoken_tts_input(processor, config, text)

    assert tts_input.tts_language_code == expected
    assert tts_input.selected_language == selected
    assert config.selected_language == selected  # STT reads this per-session selection.


def test_short_first_batch_keeps_selection_for_the_whole_response():
    processor = _detecting_processor()
    config = _session_with_language("en")

    first = _spoken_tts_input(processor, config, "Sí.")
    later = _spoken_tts_input(processor, config, "Puedo ayudarte a encontrar la estación de tren más cercana.")
    list(processor.process(EndOfResponse(turn_id="turn_1", turn_revision=0)))
    next_response = _spoken_tts_input(processor, config, "Claro, la estación está a dos calles de aquí.")

    assert [first.tts_language_code, later.tts_language_code] == ["en", "en"]
    assert next_response.tts_language_code == "es"


def test_confident_first_batch_keeps_detected_language_for_the_whole_response():
    processor = _detecting_processor()
    config = _session_with_language("en")

    first = _spoken_tts_input(processor, config, "Puedo ayudarte a encontrar la estación de tren más cercana.")
    later = _spoken_tts_input(processor, config, "I can also help you find the train station in London.")

    assert [first.tts_language_code, later.tts_language_code] == ["es", "es"]


def test_detected_language_choices_stay_isolated_between_concurrent_sessions():
    first_processor, second_processor = _detecting_processor(), _detecting_processor()
    first_config, second_config = _session_with_language("en"), _session_with_language("fr")

    first = _spoken_tts_input(first_processor, first_config, "Puedo ayudarte a encontrar la estación de tren.")
    second = _spoken_tts_input(second_processor, second_config, "Sure.")
    first_later = _spoken_tts_input(first_processor, first_config, "Sure.")
    second_later = _spoken_tts_input(second_processor, second_config, "I can help you find the train station.")

    assert [first.tts_language_code, first_later.tts_language_code] == ["es", "es"]
    assert [second.tts_language_code, second_later.tts_language_code] == ["fr", "fr"]
    assert [first_config.selected_language, second_config.selected_language] == ["en", "fr"]


@pytest.mark.parametrize("model_cls", [LLMResponseChunk, AssistantOutputEvent])
def test_ordered_parts_reject_inconsistent_legacy_text(model_cls):
    with pytest.raises(ValidationError, match="text must match the ordered parts"):
        model_cls(parts=[AssistantTextPart(text="before")], text="after")


@pytest.mark.parametrize("model_cls", [LLMResponseChunk, AssistantOutputEvent])
def test_ordered_parts_reject_inconsistent_legacy_tools(model_cls):
    tool = ResponseFunctionToolCall(type="function_call", call_id="call_1", name="tool", arguments="{}")

    with pytest.raises(ValidationError, match="tools must match the ordered parts"):
        model_cls(parts=[AssistantTextPart(text="before")], tools=[tool])


@pytest.mark.parametrize("model_cls", [LLMResponseChunk, AssistantOutputEvent])
def test_explicit_empty_ordered_parts_reject_legacy_text(model_cls):
    with pytest.raises(ValidationError, match="text must match the ordered parts"):
        model_cls(parts=[], text="legacy")


@pytest.mark.parametrize("model_cls", [LLMResponseChunk, AssistantOutputEvent])
def test_explicit_empty_ordered_parts_reject_legacy_tools(model_cls):
    tool = ResponseFunctionToolCall(type="function_call", call_id="call_1", name="tool", arguments="{}")

    with pytest.raises(ValidationError, match="tools must match the ordered parts"):
        model_cls(parts=[], tools=[tool])


@pytest.mark.parametrize("model_cls", [LLMResponseChunk, AssistantOutputEvent])
def test_omitted_ordered_parts_keep_legacy_compatibility(model_cls):
    tool = ResponseFunctionToolCall(type="function_call", call_id="call_1", name="tool", arguments="{}")

    model = model_cls(text="legacy", tools=[tool])

    assert model.parts == [AssistantTextPart(text="legacy"), AssistantToolCallPart(tool=tool)]


@pytest.mark.parametrize("model_cls", [LLMResponseChunk, AssistantOutputEvent])
def test_legacy_instance_revalidation_does_not_duplicate_synthesized_parts(model_cls):
    original = model_cls(text="legacy")

    restored = model_cls.model_validate(original)

    assert restored.parts == [AssistantTextPart(text="legacy")]


@pytest.mark.parametrize("model_cls", [LLMResponseChunk, AssistantOutputEvent])
def test_legacy_exclude_unset_dump_round_trip(model_cls):
    original = model_cls(text="legacy")

    dumped = original.model_dump(exclude_unset=True)
    restored = model_cls.model_validate(dumped)

    assert dumped == {"text": "legacy"}
    assert restored.parts == [AssistantTextPart(text="legacy")]


@pytest.mark.parametrize("model_cls", [LLMResponseChunk, AssistantOutputEvent])
def test_ordered_parts_allow_consistent_model_dump_round_trip(model_cls):
    tool = ResponseFunctionToolCall(type="function_call", call_id="call_1", name="tool", arguments="{}")
    original = model_cls(
        parts=[
            AssistantTextPart(text="before"),
            AssistantToolCallPart(tool=tool),
        ]
    )

    restored = model_cls.model_validate(original.model_dump())

    assert restored.parts == original.parts
    assert restored.text == "before"
    assert restored.tools == [tool]


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


def test_generation_done_side_channel_does_not_wait_for_tts_delivery():
    tracker = SpeculativeTurnTracker()
    tracker.start_turn()
    side_events = Queue()
    processor = LMOutputProcessor.__new__(LMOutputProcessor)
    processor.setup(speculative_turns=tracker, text_output_queue=side_events)
    tool = ResponseFunctionToolCall(type="function_call", call_id="call_1", name="lookup", arguments="{}")

    ordered = [
        *processor.process(
            LLMResponseChunk(
                parts=[AssistantTextPart(text="One moment."), AssistantToolCallPart(tool=tool)],
                response_key="response_1",
                turn_id="turn_1",
                turn_revision=0,
            )
        ),
        *processor.process(
            EndOfResponse(
                response_key="response_1", turn_id="turn_1", turn_revision=0, input_tool_call_ids=["call_previous"]
            )
        ),
    ]

    tool_ready = side_events.get_nowait()
    logical_done = side_events.get_nowait()
    assert isinstance(tool_ready, AssistantToolCallReadyEvent)
    assert tool_ready.response_key == "response_1"
    assert tool_ready.output_sequence == 1
    assert tool_ready.part.tool == tool
    assert isinstance(logical_done, ResponseGenerationDoneEvent)
    assert logical_done.response_key == "response_1"
    assert logical_done.call_ids == ["call_1"]
    assert logical_done.succeeded is True
    assert [type(item) for item in ordered] == [
        AssistantOutputEvent,
        TTSInput,
        AssistantOutputEvent,
        AssistantResponseDoneEvent,
        EndOfResponse,
    ]
    assert [item.output_sequence for item in ordered if isinstance(item, AssistantOutputEvent)] == [0, 1]
    ordered_done = next(item for item in ordered if isinstance(item, AssistantResponseDoneEvent))
    assert ordered_done.response_key == "response_1"
    assert ordered_done.input_tool_call_ids == ["call_previous"]


def test_unclaimed_prefetch_emits_tool_ready_side_channel_before_tts_delivery():
    side_events = Queue()
    processor = LMOutputProcessor.__new__(LMOutputProcessor)
    processor.setup(speculative_turns=SpeculativeTurnTracker(), text_output_queue=side_events)
    transaction = ResponsePrefetchTransaction()
    tool = ResponseFunctionToolCall(type="function_call", call_id="call_2", name="lookup", arguments="{}")

    ordered = list(
        processor.process(
            LLMResponseChunk(
                parts=[AssistantTextPart(text="One more check."), AssistantToolCallPart(tool=tool)],
                response_key="response_prefetch",
                prefetch_transaction=transaction,
            )
        )
    )

    tool_ready = side_events.get_nowait()
    assert isinstance(tool_ready, AssistantToolCallReadyEvent)
    assert tool_ready.response_key == "response_prefetch"
    assert tool_ready.output_sequence == 1
    assert tool_ready.part.tool == tool
    assert not transaction.claimed
    assert [type(item) for item in ordered] == [AssistantOutputEvent, TTSInput, AssistantOutputEvent]


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


def test_token_usage_stays_on_ordered_response_path():
    tracker = SpeculativeTurnTracker()
    tracker.start_turn()
    processor = _processor(tracker)

    outputs = [
        *processor.process(
            TokenUsage(
                input_tokens=11,
                output_tokens=7,
                response_key="response_1",
                turn_id="turn_1",
                turn_revision=0,
                cancel_generation=7,
            )
        ),
        *processor.process(
            EndOfResponse(
                response_key="response_1",
                turn_id="turn_1",
                turn_revision=0,
                cancel_generation=7,
            )
        ),
    ]

    assert [type(output) for output in outputs] == [TokenUsageEvent, AssistantResponseDoneEvent, EndOfResponse]
    event = outputs[0]
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

    assert isinstance(outputs[0], AssistantOutputEvent)
    assert outputs[0].text == text
    assert outputs[0].cancel_generation == 7
    assert [isinstance(output, TTSInput) for output in outputs] == [False, *([True] if expect_tts else [])]


def test_llm_output_language_flag_warms_detector_and_routes_detected_code(monkeypatch):
    seen = []

    class Detector:
        def compute_language_confidence_values(self, text):
            seen.append(text)
            return [
                SimpleNamespace(language=SimpleNamespace(iso_code_639_1=SimpleNamespace(name="FR")), value=0.9),
                SimpleNamespace(language=SimpleNamespace(iso_code_639_1=SimpleNamespace(name="EN")), value=0.1),
            ]

    detector = Detector()
    monkeypatch.setattr(language_detection, "warm_language_detector", lambda: detector)
    processor = LMOutputProcessor.__new__(LMOutputProcessor)
    processor.setup(detect_llm_output_language=True)

    assert processor._language_detector is detector
    outputs = list(
        processor.process(LLMResponseChunk(text="Bonjour, je peux vous aider aujourd'hui.", language_code="en"))
    )

    assert seen == ["Bonjour, je peux vous aider aujourd'hui."]
    assert next(item for item in outputs if isinstance(item, TTSInput)).language_code == "fr"


def test_llm_output_language_falls_back_on_short_text_and_resets_per_response(monkeypatch):
    class Detector:
        def compute_language_confidence_values(self, text):
            return [
                SimpleNamespace(language=SimpleNamespace(iso_code_639_1=SimpleNamespace(name="FR")), value=0.9),
                SimpleNamespace(language=SimpleNamespace(iso_code_639_1=SimpleNamespace(name="EN")), value=0.1),
            ]

    monkeypatch.setattr(language_detection, "warm_language_detector", Detector)
    processor = LMOutputProcessor.__new__(LMOutputProcessor)
    processor.setup(detect_llm_output_language=True)

    def code(text):
        outputs = list(processor.process(LLMResponseChunk(text=text, language_code="en")))
        return next(item for item in outputs if isinstance(item, TTSInput)).language_code

    assert code("Oui.") is None
    assert code("Bonjour, je peux vous aider aujourd'hui.") == "fr"
    assert code("Merci.") == "fr"
    list(processor.process(EndOfResponse()))
    assert code("Oui.") is None


def test_llm_output_language_disabled_keeps_input_language_without_warming(monkeypatch):
    monkeypatch.setattr(
        language_detection,
        "warm_language_detector",
        lambda: (_ for _ in ()).throw(AssertionError("should not warm")),
    )
    processor = LMOutputProcessor.__new__(LMOutputProcessor)
    processor.setup()

    outputs = list(
        processor.process(LLMResponseChunk(text="Bonjour, je peux vous aider aujourd'hui.", language_code="en"))
    )

    assert next(item for item in outputs if isinstance(item, TTSInput)).language_code == "en"


def test_ambiguous_assistant_chunk_does_not_force_a_tts_language():
    if not language_detection.LINGUA_AVAILABLE:
        pytest.skip("lingua-language-detector is not installed")

    processor = LMOutputProcessor.__new__(LMOutputProcessor)
    processor.setup(detect_llm_output_language=True)
    outputs = list(processor.process(LLMResponseChunk(text="100 times 100000 is 10,000,000.", language_code="en")))

    assert next(item for item in outputs if isinstance(item, TTSInput)).language_code is None


def test_llm_output_language_detection_failure_preserves_tts_input(monkeypatch):
    class FailingDetector:
        def compute_language_confidence_values(self, text):
            raise RuntimeError("detector failed")

    monkeypatch.setattr(language_detection, "warm_language_detector", FailingDetector)
    processor = LMOutputProcessor.__new__(LMOutputProcessor)
    processor.setup(detect_llm_output_language=True)

    outputs = list(
        processor.process(LLMResponseChunk(text="Bonjour, je peux vous aider aujourd'hui.", language_code="en"))
    )

    assert next(item for item in outputs if isinstance(item, TTSInput)).language_code is None


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
        AssistantOutputEvent,
        AssistantOutputEvent,
        TTSInput,
        AssistantOutputEvent,
        AssistantOutputEvent,
        TTSInput,
    ]
    events = [output for output in outputs if isinstance(output, AssistantOutputEvent)]
    assert [event.parts[0].type for event in events] == ["tool_call", "text", "tool_call", "text"]
    assert [output.text for output in outputs if isinstance(output, TTSInput)] == ["between", "after"]


def test_audio_output_keeps_response_identity():
    handler = object.__new__(BaseHandler)

    queued = handler.output_for_queue(b"audio", TTSInput(text="hello", response_key="response_1"))

    assert isinstance(queued, AudioOutput)
    assert (queued.audio, queued.response_key) == (b"audio", "response_1")


def test_prefetch_transaction_propagates_to_tts_input():
    processor = _processor(SpeculativeTurnTracker())
    transaction = ResponsePrefetchTransaction()

    outputs = list(
        processor.process(
            LLMResponseChunk(
                text="hidden audio",
                prefetch_transaction=transaction,
            )
        )
    )

    tts_input = next(output for output in outputs if isinstance(output, TTSInput))
    assert tts_input.prefetch_transaction is transaction


def test_prefetch_transaction_does_not_gate_lm_worker():
    started = Event()

    class RecordingLMHandler(BaseHandler):
        def process(self, item):
            started.set()
            yield EndOfResponse(response_key=item.response_key)

    queue_in, queue_out = Queue(), Queue()
    handler = RecordingLMHandler(Event(), queue_in, queue_out)
    transaction = ResponsePrefetchTransaction()
    request = GenerateResponseRequest(
        runtime_config=RuntimeConfig(chat=Chat(2)),
        prefetch_transaction=transaction,
    )
    queue_in.put(request)
    queue_in.put(PIPELINE_END)
    worker = Thread(target=handler.run)
    worker.start()

    assert started.wait(timeout=1.0)
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert not transaction.claimed
    assert isinstance(queue_out.get_nowait(), EndOfResponse)
    assert queue_out.get_nowait() == PIPELINE_END


def test_discarded_prefetch_releases_tts_worker_without_synthesis():
    class RecordingTTSHandler(BaseHandler):
        def process(self, item):
            yield item.text.encode()

    queue_in, queue_out = Queue(), Queue()
    handler = RecordingTTSHandler(Event(), queue_in, queue_out)
    transaction = ResponsePrefetchTransaction()
    queue_in.put(TTSInput(text="hidden", prefetch_transaction=transaction))
    worker = Thread(target=handler.run)
    worker.start()

    Event().wait(0.1)
    assert queue_out.empty()
    transaction.discard()
    queue_in.put(TTSInput(text="fresh"))
    queue_in.put(PIPELINE_END)
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert queue_out.get_nowait() == b"fresh"
    assert queue_out.get_nowait() == PIPELINE_END


def test_claimed_prefetch_releases_tts_worker_for_synthesis():
    class RecordingTTSHandler(BaseHandler):
        def process(self, item):
            yield item.text.encode()

    queue_in, queue_out = Queue(), Queue()
    handler = RecordingTTSHandler(Event(), queue_in, queue_out)
    transaction = ResponsePrefetchTransaction()
    queue_in.put(TTSInput(text="hidden", prefetch_transaction=transaction))
    worker = Thread(target=handler.run)
    worker.start()

    Event().wait(0.1)
    assert queue_out.empty()
    assert transaction.claim()
    queue_in.put(PIPELINE_END)
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert queue_out.get_nowait() == b"hidden"
    assert queue_out.get_nowait() == PIPELINE_END


def test_tts_handler_forwards_response_events_without_processing_them():
    queue_in, queue_out = Queue(), Queue()
    handler = BaseHandler(Event(), queue_in, queue_out)
    event = AssistantOutputEvent(text="before")
    queue_in.put(event)
    queue_in.put(PIPELINE_END)

    handler.run()

    assert queue_out.get_nowait() is event
    assert queue_out.get_nowait() == PIPELINE_END


def test_tts_handler_forwards_provider_usage_after_cancellation():
    queue_in, queue_out = Queue(), Queue()
    handler = BaseHandler(Event(), queue_in, queue_out)
    handler.cancel_scope = CancelScope()
    generation = handler.cancel_scope.generation
    handler.cancel_scope.cancel()
    event = TokenUsageEvent(input_tokens=11, output_tokens=7, cancel_generation=generation)
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

    assert [type(output) for output in outputs] == [AssistantOutputEvent, TTSInput]


def test_reopen_grace_holds_output_until_elapsed():
    tracker, processor = _tracked_processor()
    tracker.start_reopen_grace("turn_1", 0, grace_s=0.08)
    done, outputs, thread = _process_in_thread(processor)

    assert not done.wait(0.02)
    assert done.wait(1.0)
    thread.join(timeout=1.0)

    assert [type(output) for output in outputs] == [AssistantOutputEvent, TTSInput]


def test_confirmed_reopen_drops_held_output():
    tracker, processor = _tracked_processor()
    candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)
    done, outputs, thread = _process_in_thread(processor)

    assert not done.wait(0.05)
    assert tracker.confirm_reopen_candidate("turn_1", 0, candidate_revision)
    assert done.wait(1.0)
    thread.join(timeout=1.0)

    assert outputs == []
