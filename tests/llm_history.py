"""Run a language model backend with the service writing its history."""

from speech_to_speech.api.openai_realtime.service import RealtimeService


def drive_llm(handler, request, *, service=None, conn_id=None):
    """Yield the backend's outputs while the service applies their history.

    Output counts as accepted as soon as it is produced. A prefetch instead
    waits for ``request.prefetch_transaction.claim()``. History stays
    provisional, as it does until ``response.done``. A response that proposed
    history but never completed it is rolled back, as the service does when
    it cancels or fails one.
    """
    if service is None:
        service = RealtimeService()
    if conn_id is None:
        conn_id = service.register()
    state = service._state(conn_id)
    state.runtime_config = request.runtime_config
    key = request.response_key
    if request.prefetch_transaction is None:
        service.history.accept(conn_id, key)
    else:
        state.tool_followup_prefetch_request = request
    proposal = None
    try:
        for output in handler.process(request):
            service.history.stage(conn_id, output)
            if getattr(output, "history", None) is not None:
                proposal = output.history
            yield output
    finally:
        if request.prefetch_transaction is None and proposal is not None and not proposal.complete:
            service.history.close(conn_id, key)
