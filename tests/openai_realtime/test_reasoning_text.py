import pytest
from openai.types.realtime import ResponseCreateEvent

from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
from speech_to_speech.pipeline.events import AssistantOutputEvent
from speech_to_speech.pipeline.messages import AssistantReasoningPart, LLMResponseChunk


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
