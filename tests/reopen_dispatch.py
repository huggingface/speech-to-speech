"""Exercise the send loop's turn hold while VAD resolves a reopen."""

import time
from contextlib import contextmanager


class _HeldDispatch:
    def __init__(self, service, conn_id, event):
        self._service = service
        self._conn_id = conn_id
        self._event = event

    def result(self, timeout: float):
        """Retry the held event as the send loop does, then dispatch it."""
        deadline = time.monotonic() + timeout
        while self._service.is_turn_output_held(self._event):
            assert time.monotonic() < deadline, "turn hold did not resolve"
            time.sleep(0.005)
        return self._service.dispatch_pipeline_event(self._conn_id, self._event)


@contextmanager
def pending_dispatch(service, conn_id, event):
    assert service.is_turn_output_held(event)
    yield _HeldDispatch(service, conn_id, event)
