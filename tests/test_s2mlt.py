import json
from queue import Queue

import pytest

from speech_to_speech.LLM.translation import (
    TranslationChatCompletionsHandler,
    TranslationOutputProcessor,
    build_translation_response_format,
    parse_partial_translation,
    repair_partial_json,
)
from speech_to_speech.pipeline.events import (
    InputTranscriptionDeltaEvent,
    InputTranscriptionDoneEvent,
    TranslationDeltaEvent,
    TranslationDoneEvent,
)
from speech_to_speech.pipeline.messages import EndOfResponse, LLMResponseChunk, PartialTranscription, Transcription
from speech_to_speech.s2mlt import ContinuousListeningEvent, validate_target_languages
from speech_to_speech.STT.translation_notifier import TranslationNotifier


def _translation_processor(delta_interval_s: float = 0.0) -> TranslationOutputProcessor:
    processor = TranslationOutputProcessor.__new__(TranslationOutputProcessor)
    processor.setup(["de", "fr"], delta_interval_s=delta_interval_s)
    return processor


def _translation_notifier() -> tuple[TranslationNotifier, Queue]:
    text_output_queue = Queue()
    notifier = TranslationNotifier.__new__(TranslationNotifier)
    notifier.setup(["de", "fr"], text_output_queue=text_output_queue)
    return notifier, text_output_queue


def test_partial_json_repair_parses_every_prefix_of_valid_output():
    complete = '{"de":"Ein \\\"Zitat\\\"","fr":"ligne\\nsuivante","corrected":"cafe ☕"}'

    for end in range(1, len(complete) + 1):
        fragment = complete[:end]
        if "{" not in fragment:
            continue
        repaired = repair_partial_json(fragment)
        parsed = json.loads(repaired)
        assert isinstance(parsed, dict), (fragment, repaired)

    assert parse_partial_translation(complete) == {
        "de": 'Ein "Zitat"',
        "fr": "ligne\nsuivante",
        "corrected": "cafe ☕",
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('```json\n{"de":"Hal', {"de": "Hal"}),
        ('noise {"de":"Hallo", "fr"', {"de": "Hallo"}),
        ('{"de": tru', {}),
        ("no object yet", None),
    ],
)
def test_partial_json_parser_handles_streaming_boundaries(raw, expected):
    assert parse_partial_translation(raw) == expected


def test_translation_output_processor_streams_snapshots_and_closes_segment():
    processor = _translation_processor()

    first = list(
        processor.process(
            LLMResponseChunk(text='{"de":"Hal', turn_id="turn-1", turn_revision=2)
        )
    )
    second = list(
        processor.process(
            LLMResponseChunk(
                text='lo","fr":"Bonjour","corrected":"Hello"}',
                turn_id="turn-1",
                turn_revision=2,
            )
        )
    )
    done = list(processor.process(EndOfResponse(turn_id="turn-1", turn_revision=2)))

    assert first == [
        TranslationDeltaEvent(
            translations={"de": "Hal"},
            turn_id="turn-1",
            turn_revision=2,
        )
    ]
    assert second == [
        TranslationDeltaEvent(
            translations={"de": "Hallo", "fr": "Bonjour"},
            corrected="Hello",
            turn_id="turn-1",
            turn_revision=2,
        )
    ]
    assert done == [
        TranslationDoneEvent(
            translations={"de": "Hallo", "fr": "Bonjour"},
            corrected="Hello",
            turn_id="turn-1",
            turn_revision=2,
        )
    ]


def test_translation_output_processor_closes_empty_and_invalid_responses_with_errors():
    processor = _translation_processor()

    empty = list(processor.process(EndOfResponse(turn_id="empty", turn_revision=0)))
    list(processor.process(LLMResponseChunk(text="not JSON", turn_id="bad", turn_revision=1)))
    invalid = list(processor.process(EndOfResponse(turn_id="bad", turn_revision=1)))

    assert empty == [
        TranslationDoneEvent(
            turn_id="empty",
            turn_revision=0,
            error="Model output was empty",
        )
    ]
    assert invalid == [
        TranslationDoneEvent(
            turn_id="bad",
            turn_revision=1,
            error="Model output could not be parsed as JSON",
        )
    ]


def test_translation_output_processor_throttles_delta_but_done_has_latest_snapshot():
    processor = _translation_processor(delta_interval_s=60.0)

    first = list(processor.process(LLMResponseChunk(text='{"de":"H', turn_id="turn-1", turn_revision=0)))
    throttled = list(
        processor.process(
            LLMResponseChunk(
                text='allo","fr":"Bonjour","corrected":"Hello"}',
                turn_id="turn-1",
                turn_revision=0,
            )
        )
    )
    done = list(processor.process(EndOfResponse(turn_id="turn-1", turn_revision=0)))

    assert len(first) == 1
    assert throttled == []
    assert done[0].translations == {"de": "Hallo", "fr": "Bonjour"}
    assert done[0].corrected == "Hello"


def test_translation_notifier_emits_transcription_events_and_stateless_request():
    notifier, events = _translation_notifier()

    assert list(
        notifier.process(PartialTranscription(text="Hel", turn_id="turn-1", turn_revision=0))
    ) == []
    requests = list(
        notifier.process(
            Transcription(
                text="  Hello  ",
                language_code="en-auto",
                turn_id="turn-1",
                turn_revision=0,
                speech_stopped_at_s=12.5,
            )
        )
    )

    assert events.get_nowait() == InputTranscriptionDeltaEvent(
        text="Hel", turn_id="turn-1", turn_revision=0
    )
    assert events.get_nowait() == InputTranscriptionDoneEvent(
        text="Hello", language_code="en", turn_id="turn-1", turn_revision=0
    )
    assert len(requests) == 1
    request = requests[0]
    assert request.response is not None
    assert request.response.output_modalities == ["text"]
    assert request.turn_id == "turn-1"
    assert request.turn_revision == 0
    messages = request.runtime_config.chat.to_transformers_chat()
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[1]["content"] == "Hello"


def test_empty_transcription_is_closed_without_an_llm_request():
    notifier, events = _translation_notifier()

    assert list(notifier.process(Transcription(text="  ", turn_id="turn-1", turn_revision=0))) == []
    assert events.get_nowait() == InputTranscriptionDoneEvent(
        text="", turn_id="turn-1", turn_revision=0
    )


def test_translation_schema_preserves_low_latency_key_order():
    response_format = build_translation_response_format(["de", "fr"])
    schema = response_format["json_schema"]["schema"]

    assert list(schema["properties"]) == ["de", "fr", "corrected"]
    assert schema["required"] == ["de", "fr", "corrected"]
    assert schema["additionalProperties"] is False


def test_chat_completions_handler_attaches_response_format(monkeypatch):
    response_format = build_translation_response_format(["de", "fr"])
    captured = {}

    def fake_request(_self, api_input, optional_kwargs):
        captured["api_input"] = api_input
        captured["optional_kwargs"] = optional_kwargs
        return "response"

    monkeypatch.setattr(
        "speech_to_speech.LLM.translation.ChatCompletionsApiModelHandler._request",
        fake_request,
    )
    handler = TranslationChatCompletionsHandler.__new__(TranslationChatCompletionsHandler)
    handler._response_format = response_format
    original_kwargs = {"temperature": 0}

    result = handler._request([{"role": "user", "content": "Hello"}], original_kwargs)

    assert result == "response"
    assert captured["optional_kwargs"] == {
        "temperature": 0,
        "response_format": response_format,
    }
    assert original_kwargs == {"temperature": 0}


@pytest.mark.parametrize("languages", [[], ["de"], ["de", "fr", "it"], ["de", "de"], ["de", "corrected"]])
def test_target_languages_require_two_distinct_non_reserved_keys(languages):
    with pytest.raises(ValueError):
        validate_target_languages(languages)

    validate_target_languages(["de", "fr"])


def test_continuous_listening_event_cannot_be_cleared():
    should_listen = ContinuousListeningEvent()
    should_listen.set()

    should_listen.clear()

    assert should_listen.is_set()
