from queue import Queue

from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
from speech_to_speech.pipeline.messages import EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker


def _processor(tracker: SpeculativeTurnTracker) -> LMOutputProcessor:
    processor = LMOutputProcessor.__new__(LMOutputProcessor)
    processor.setup(text_output_queue=Queue(), speculative_turns=tracker)
    return processor


def test_stale_end_of_response_is_not_forwarded_to_tts():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 1)
    processor = _processor(tracker)

    outputs = list(processor.process(EndOfResponse(turn_id="turn_1", turn_revision=0)))

    assert outputs == []


def test_latest_end_of_response_is_forwarded_to_tts():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 1)
    processor = _processor(tracker)

    outputs = list(processor.process(EndOfResponse(turn_id="turn_1", turn_revision=1)))

    assert len(outputs) == 1
    assert outputs[0].turn_id == "turn_1"
    assert outputs[0].turn_revision == 1
