"""Run a language model backend with the service writing its history."""

import pytest

from speech_to_speech.api.openai_realtime.service import RealtimeService

_async_runners = []


@pytest.fixture(autouse=True)
def close_llm_runners():
    yield
    for runner in _async_runners:
        runner.close()
    _async_runners.clear()


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


def mock_async_client(handler):
    """Forward async SDK calls to the backend tests' existing synchronous fakes."""
    from functools import reduce
    from unittest.mock import AsyncMock, MagicMock

    _async_runners.append(handler._async_runner)
    client = MagicMock()
    client.close = AsyncMock()
    for path in (("responses",), ("chat", "completions")):

        async def create(*args, _path=path, **kwargs):
            resource = reduce(getattr, _path, handler.client)
            response = resource.create(*args, **kwargs)
            stream = MagicMock()
            stream.__aenter__.return_value = stream
            stream.__aiter__.return_value = response

            def close(*exc):
                response.close()
                return False

            stream.__aexit__.side_effect = close
            return stream

        reduce(getattr, path, client).create = AsyncMock(side_effect=create)
    return client
