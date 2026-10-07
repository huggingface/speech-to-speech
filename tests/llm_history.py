"""Drive backend tests through the real service-owned history path."""

from queue import Queue

from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
from speech_to_speech.pipeline.events import PipelineEvent, ResponseGenerationDoneEvent
from speech_to_speech.pipeline.messages import EndOfResponse


def drive_llm(handler, request):
    service = RealtimeService()
    conn_id = service.register()
    state = service._state(conn_id)
    state.runtime_config = request.runtime_config
    state.current_response_params = request.response
    state.current_response_turn_id = request.turn_id
    state.current_response_turn_revision = request.turn_revision
    state.mark_response_pending(request.response_key)
    service.speculative_turns = getattr(handler, "speculative_turns", None)
    if request.prefetch_transaction is not None:
        state.tool_followup_prefetch_request = request
    side_events = Queue()
    processor = object.__new__(LMOutputProcessor)
    processor.setup(text_output_queue=side_events)
    # LM and service relevance checks stay in production code; this helper
    # only supplies the queue consumer absent from a backend unit test.
    complete = False
    try:
        for output in handler.process(request):
            for processed in processor.process(output):
                while not side_events.empty():
                    side = side_events.get_nowait()
                    if service.response.is_response_output_blocked(conn_id, side.response_key) and not isinstance(
                        side, ResponseGenerationDoneEvent
                    ):
                        service.history.stage(conn_id, side)
                    else:
                        service.dispatch_pipeline_event(conn_id, side)
                if isinstance(processed, PipelineEvent):
                    if service.response.is_response_output_blocked(conn_id, processed.response_key):
                        service.history.stage(conn_id, processed)
                    else:
                        service.dispatch_pipeline_event(conn_id, processed)
                elif isinstance(processed, EndOfResponse) and processed.cleanup_only:
                    if state.in_response:
                        service.finish_response(conn_id, status="cancelled", response_key=processed.response_key)
                    else:
                        service.close_response_key(conn_id, processed.response_key)
            if isinstance(output, EndOfResponse):
                complete = True
            yield output
    finally:
        if not complete:
            # Closing the driver stands for the transport cancelling this reply;
            # the model itself has no authority to roll back shared history.
            if state.in_response:
                service.finish_response(conn_id, status="cancelled", response_key=request.response_key)
            else:
                service.close_response_key(conn_id, request.response_key)
