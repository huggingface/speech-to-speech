"""Exercise the transport's blocking dispatch path while VAD resolves a reopen."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager

import pytest


@contextmanager
def pending_dispatch(service, conn_id, event):
    with ThreadPoolExecutor(max_workers=1) as executor:
        dispatch = executor.submit(service.dispatch_pipeline_event, conn_id, event)
        with pytest.raises(TimeoutError):
            dispatch.result(timeout=0.02)
        yield dispatch
