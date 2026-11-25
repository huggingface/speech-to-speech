"""Run a language model backend with the service writing its history."""

import asyncio
import threading

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


class _ThreadedStream:
    """A synchronous fake stream whose blocking calls can be cancelled like awaits."""

    _END = object()

    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self._response.close()
        return False

    async def __aiter__(self):
        iterator = await asyncio.to_thread(iter, self._response)
        while (item := await asyncio.to_thread(next, iterator, self._END)) is not self._END:
            yield item


async def _create_in_thread(create, *args, **kwargs):
    abandoned = threading.Event()

    def call():
        response = create(*args, **kwargs)
        if abandoned.is_set():
            response.close()  # The request was cancelled; nobody will read this.
        return response

    try:
        return _ThreadedStream(await asyncio.to_thread(call))
    except asyncio.CancelledError:
        abandoned.set()
        raise


def mock_async_client(handler, *, in_threads=False):
    """Forward async SDK calls to the backend tests' existing synchronous fakes.

    A fake that blocks, to stand for a silent provider, blocks the event loop
    with it. ``in_threads`` runs the fake's calls in threads instead, so that a
    blocked call can be cancelled, as a real provider read can.
    """
    from functools import reduce
    from unittest.mock import AsyncMock, MagicMock

    _async_runners.append(handler._async_runner)
    client = MagicMock()
    client.close = AsyncMock()
    # Request-scoped options apply to the synchronous fake the calls end up on.
    client.with_options.side_effect = lambda **options: (handler.client.with_options(**options), client)[1]
    for path in (("responses",), ("chat", "completions")):

        async def create(*args, _path=path, **kwargs):
            resource = reduce(getattr, _path, handler.client)
            if in_threads:
                return await _create_in_thread(resource.create, *args, **kwargs)
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
