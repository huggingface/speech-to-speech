"""Readable reasoning stays separate from speech, history, and encrypted items."""

from types import SimpleNamespace

import pytest
from openai.types.responses import ResponseOutputItemDoneEvent, ResponseReasoningItem

from speech_to_speech.LLM.base_openai_compatible_language_model import ReasoningDelta
from speech_to_speech.LLM.chat_completions_language_model import _iter_chat_response_events, _iter_chat_stream_events
from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import AssistantOutputEvent, AssistantResponseDoneEvent
from speech_to_speech.pipeline.messages import AssistantReasoningPart, EndOfResponse, LLMResponseChunk, TTSInput
from tests.test_chat_completions_backend import _FakeStream
from tests.test_chat_completions_backend import _make_handler as chat_handler
from tests.test_responses_api_language_model import _make_handler as responses_handler
from tests.test_responses_api_language_model import _make_request, _make_response, _make_stream


def chat_chunk(text=None, **extra):
    return SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason=None,
                delta=SimpleNamespace(content=text, tool_calls=None, **extra),
            )
        ],
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_chat_extracts_readable_reasoning_separately(stream, field):
    message = SimpleNamespace(content="42.", tool_calls=None, **{field: "Think first."})
    if stream:
        events = list(_iter_chat_stream_events([chat_chunk("42.", **{field: "Think first."})]))
    else:
        events = list(
            _iter_chat_response_events(
                SimpleNamespace(
                    usage=None,
                    choices=[SimpleNamespace(message=message)],
                )
            )
        )
    assert [event.text for event in events if isinstance(event, ReasoningDelta)] == ["Think first."]


def test_chat_never_stringifies_opaque_reasoning():
    events = list(_iter_chat_stream_events([chat_chunk(reasoning={"encrypted_content": "secret"})]))
    assert not any(isinstance(event, ReasoningDelta) for event in events)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("backend", ["chat", "responses"])
def test_reasoning_reaches_ordered_pipeline_only_for_opted_in_clients(stream, enabled, backend):
    request = _make_request()
    request.runtime_config.session.extensions = ["speech_to_speech.reasoning.delta"] if enabled else []
    if backend == "chat":
        handler = chat_handler(stream=stream)
        result = (
            _FakeStream([chat_chunk("42.", reasoning_content="Think first.")])
            if stream
            else SimpleNamespace(
                usage=None,
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="42.",
                            tool_calls=None,
                            reasoning_content="Think first.",
                        )
                    )
                ],
            )
        )
        handler.client.chat.completions.create = lambda **kwargs: result
    else:
        handler = responses_handler(stream=stream)
        item = ResponseReasoningItem(
            id="rs1",
            type="reasoning",
            content=[{"type": "reasoning_text", "text": "Think first."}],
            summary=[{"type": "summary_text", "text": "A brief summary."}],
            encrypted_content="secret",
        )
        result = (
            _make_stream(
                [
                    ResponseOutputItemDoneEvent(
                        type="response.output_item.done",
                        sequence_number=0,
                        output_index=0,
                        item=item,
                    )
                ]
            )
            if stream
            else _make_response([item])
        )
        handler.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: result))
    processor = LMOutputProcessor.__new__(LMOutputProcessor)
    processor.setup()
    outputs = [item for chunk in handler.process(request) for item in processor.process(chunk)]
    parts = [
        part
        for event in outputs
        if isinstance(event, (AssistantOutputEvent, AssistantResponseDoneEvent))
        for part in (event.parts if isinstance(event, AssistantOutputEvent) else event.reasoning)
        if isinstance(part, AssistantReasoningPart)
    ]
    assert [part.text for part in parts] == (
        ["Think first."] + (["A brief summary."] if backend == "responses" else []) if enabled else []
    )
    spoken = [item.text for item in outputs if isinstance(item, TTSInput)]
    assert spoken == (["42."] if backend == "chat" else [])
    assert all("secret" not in part.text for part in parts)


@pytest.mark.parametrize("kind", ["reasoning", "summary"])
def test_responses_stream_deduplicates_completed_reasoning_and_keeps_missing_suffix(kind):
    from openai.types.responses import ResponseReasoningSummaryTextDeltaEvent, ResponseReasoningTextDeltaEvent

    cls = ResponseReasoningSummaryTextDeltaEvent if kind == "summary" else ResponseReasoningTextDeltaEvent
    index = {"summary_index" if kind == "summary" else "content_index": 0}
    events = [
        cls(
            type="response.reasoning_summary_text.delta" if kind == "summary" else "response.reasoning_text.delta",
            item_id="rs1",
            output_index=0,
            sequence_number=i,
            delta=text,
            **index,
        )
        for i, text in enumerate(["Think ", "first."])
    ]
    item = ResponseReasoningItem(
        id="rs1",
        type="reasoning",
        encrypted_content="never readable",
        summary=[{"type": "summary_text", "text": "Think first. Then answer."}] if kind == "summary" else [],
        content=[{"type": "reasoning_text", "text": "Think first. Then answer."}] if kind == "reasoning" else None,
    )
    events.append(
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            sequence_number=2,
            output_index=0,
            item=item,
        )
    )
    parsed = list(responses_handler()._iter_stream_events(_make_stream(events)))
    reasoning = [event for event in parsed if isinstance(event, ReasoningDelta)]
    assert "".join(event.text for event in reasoning) == "Think first. Then answer."
    assert all(event.kind == kind for event in reasoning)
    assert [event for event in parsed if isinstance(event, ResponseReasoningItem)] == [item]


def test_reasoning_parts_never_become_legacy_text_or_tts():
    chunk = LLMResponseChunk(parts=[AssistantReasoningPart(text="Not spoken.")])
    assert chunk.text == "" and chunk.tools == []
    processor = LMOutputProcessor.__new__(LMOutputProcessor)
    processor.setup()
    outputs = list(processor.process(chunk))
    assert outputs == []


def test_cancellation_during_thinking_discards_collected_reasoning():
    from openai.types.responses import ResponseReasoningTextDeltaEvent

    scope = CancelScope()
    handler = responses_handler(cancel_scope=scope)
    request = _make_request()
    request.runtime_config.session.extensions = ["speech_to_speech.reasoning.delta"]

    def stream():
        for index, text in enumerate(["Before cancellation.", "Must not escape."]):
            if index:
                scope.cancel()
            yield ResponseReasoningTextDeltaEvent(
                type="response.reasoning_text.delta",
                item_id="rs1",
                output_index=0,
                content_index=0,
                sequence_number=index,
                delta=text,
            )

    handler.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: _make_stream(stream())))
    outputs = list(handler.process(request))
    assert not any(isinstance(item, LLMResponseChunk) for item in outputs)
    assert all(not item.reasoning for item in outputs if isinstance(item, EndOfResponse))
