from queue import Queue
from threading import Event

from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
from speech_to_speech.pipeline.control import SESSION_END
from speech_to_speech.pipeline.events import AssistantOutputEvent
from speech_to_speech.pipeline.messages import EndOfResponse, LLMResponseChunk, TTSInput
from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler


def _make_handler():
    handler = object.__new__(Qwen3TTSHandler)
    handler.queue_in = Queue()
    handler.queue_out = Queue()
    handler.stop_event = Event()
    return handler


def test_coalesce_pending_tts_input_merges_ready_sentences_and_stops_before_response_end():
    handler = _make_handler()

    handler.queue_in.put(TTSInput(text="Second sentence.", language_code="en"))
    handler.queue_in.put(TTSInput(text="Third sentence.", language_code="en"))
    handler.queue_in.put(EndOfResponse())

    text, lang = handler._coalesce_pending_tts_input(TTSInput(text="First sentence.", language_code="en"))

    assert text == "First sentence. Second sentence. Third sentence."
    assert lang == "en"
    remaining = handler.queue_in.get_nowait()
    assert isinstance(remaining, EndOfResponse)


def test_coalesce_pending_processor_output_for_one_response():
    processor = LMOutputProcessor.__new__(LMOutputProcessor)
    processor.setup()
    outputs = [
        *processor.process(LLMResponseChunk(text="First sentence.", response_key="response_1")),
        *processor.process(LLMResponseChunk(text="Second sentence.", response_key="response_1")),
    ]
    assert [type(output) for output in outputs] == [
        AssistantOutputEvent,
        TTSInput,
        AssistantOutputEvent,
        TTSInput,
    ]

    handler = _make_handler()
    handler.queue_in.put(outputs[2])
    handler.queue_in.put(outputs[3])

    text, _ = handler._coalesce_pending_tts_input(outputs[1])

    assert text == "First sentence. Second sentence."
    assert handler.queue_out.get_nowait() == outputs[2]
    assert handler.queue_in.empty()


def test_coalesce_pending_tts_input_stops_before_control_messages():
    handler = _make_handler()

    handler.queue_in.put(SESSION_END)
    text, lang = handler._coalesce_pending_tts_input(TTSInput(text="Hello.", language_code="en"))

    assert text == "Hello."
    assert lang == "en"
    assert handler.queue_in.get_nowait() == SESSION_END


def test_coalesce_pending_tts_input_stops_at_ordered_output_boundary():
    handler = _make_handler()
    handler.queue_in.put(
        AssistantOutputEvent(tools=[{"type": "function_call", "call_id": "call_1", "name": "tool", "arguments": "{}"}])
    )
    handler.queue_in.put(TTSInput(text="After tool."))

    text, _ = handler._coalesce_pending_tts_input(TTSInput(text="Before tool."))

    assert text == "Before tool."
    assert isinstance(handler.queue_in.get_nowait(), AssistantOutputEvent)
