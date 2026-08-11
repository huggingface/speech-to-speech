"""Queued provider token usage must survive a client disconnect.

`LMOutputProcessor` routes `TokenUsageEvent` through the router's output queues. Releasing a
session calls `_clean_unit()`, which drains those queues, and by then the send loop is gone —
so a usage event that reached a queue but was never dispatched was discarded before
`RealtimeService.unregister()` rolled the connection's usage into the global totals. The
provider had already billed those tokens.

`_release_session()` now accounts queued usage directly, before the flush and while the
session is still registered.
"""

from __future__ import annotations

from queue import Queue
from threading import Event as ThreadingEvent

import pytest

from speech_to_speech.api.openai_realtime import websocket_router as router
from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit, SessionState
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import TokenUsageEvent


@pytest.fixture
def unit():
    text_prompt_queue: Queue = Queue()
    should_listen = ThreadingEvent()
    should_listen.set()
    service = RealtimeService(text_prompt_queue=text_prompt_queue, should_listen=should_listen)
    return PipelineUnit(
        index=0,
        service=service,
        cancel_scope=CancelScope(),
        should_listen=should_listen,
        response_playing=ThreadingEvent(),
        input_queue=Queue(),
        output_queue=Queue(),
        text_output_queue=Queue(),
        text_prompt_queue=text_prompt_queue,
        handlers=[],
    )


def usage(input_tokens: int, output_tokens: int) -> TokenUsageEvent:
    return TokenUsageEvent(input_tokens=input_tokens, output_tokens=output_tokens)


def totals(service: RealtimeService) -> tuple[int, int]:
    return service.total_usage.input_tokens, service.total_usage.output_tokens


# --- the dropped usage --------------------------------------------------------------------


@pytest.mark.parametrize("queue_name", ["text_output_queue", "output_queue"])
def test_queued_usage_survives_disconnect(unit, queue_name):
    """The core bug: usage on a router queue was flushed away by release."""
    session_id = unit.service.register()
    getattr(unit, queue_name).put(usage(120, 45))

    router._account_queued_token_usage(unit, session_id)
    router._clean_unit(unit)
    unit.service.unregister(session_id)

    assert totals(unit.service) == (120, 45)


def test_usage_on_both_queues_is_accounted_once_each(unit):
    session_id = unit.service.register()
    unit.output_queue.put(usage(10, 1))
    unit.text_output_queue.put(usage(20, 2))

    router._account_queued_token_usage(unit, session_id)
    router._clean_unit(unit)
    unit.service.unregister(session_id)

    assert totals(unit.service) == (30, 3)


def test_multiple_queued_usage_events_all_count(unit):
    session_id = unit.service.register()
    for _ in range(3):
        unit.text_output_queue.put(usage(5, 2))

    router._account_queued_token_usage(unit, session_id)
    router._clean_unit(unit)
    unit.service.unregister(session_id)

    assert totals(unit.service) == (15, 6)


def test_no_queued_usage_leaves_totals_untouched(unit):
    session_id = unit.service.register()

    router._account_queued_token_usage(unit, session_id)
    router._clean_unit(unit)
    unit.service.unregister(session_id)

    assert totals(unit.service) == (0, 0)


def test_accounting_failure_does_not_break_release(unit, monkeypatch):
    """Release must not be blocked by an accounting error."""
    session_id = unit.service.register()
    unit.text_output_queue.put(usage(7, 7))

    def boom(*args, **kwargs):
        raise RuntimeError("accounting exploded")

    monkeypatch.setattr(unit.service, "dispatch_pipeline_event", boom)

    router._account_queued_token_usage(unit, session_id)  # must not raise
    router._clean_unit(unit)
    unit.service.unregister(session_id)


def test_unknown_session_does_not_raise(unit):
    """A duplicate/late release path must not explode on an already-gone session."""
    unit.text_output_queue.put(usage(1, 1))

    router._account_queued_token_usage(unit, "no-such-session")  # must not raise


# --- end to end through the real release path ---------------------------------------------


async def test_release_session_preserves_queued_usage(unit, monkeypatch):
    """Drives `_release_session` itself, so this fails against the pre-fix router."""
    session_id = unit.service.register()
    unit.session = SessionState(session_id=session_id)
    unit.text_output_queue.put(usage(64, 32))

    # The drain-and-release task needs a live loop but nothing here depends on it
    # completing, so stub it out to keep the test deterministic.
    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(router, "_release_unit_after_drain", noop)

    router._release_session(unit, session_id)
    unit.service.unregister(session_id)

    assert totals(unit.service) == (64, 32)


# --- _drain_token_usage must not disturb anything else ------------------------------------


def test_drain_removes_only_usage_events():
    q: Queue = Queue()
    a, b = object(), object()
    u = usage(3, 4)
    for item in (a, u, b):
        q.put(item)

    drained = router._drain_token_usage(q)

    assert drained == [u]
    assert [q.get_nowait(), q.get_nowait()] == [a, b]
    assert q.empty()


def test_drain_preserves_order_of_remaining_items():
    q: Queue = Queue()
    items = [object() for _ in range(5)]
    q.put(items[0])
    q.put(usage(1, 1))
    for item in items[1:]:
        q.put(item)

    router._drain_token_usage(q)

    assert [q.get_nowait() for _ in range(5)] == items


def test_drain_on_empty_queue_returns_nothing():
    q: Queue = Queue()

    assert router._drain_token_usage(q) == []
    assert q.empty()


# --- stale output must still be discarded (acceptance criterion) --------------------------


def test_release_still_discards_stale_assistant_output(unit):
    """Accounting usage must not start preserving assistant text/audio across sessions."""
    session_id = unit.service.register()
    unit.output_queue.put(b"stale-audio")
    unit.text_output_queue.put(object())
    unit.text_output_queue.put(usage(9, 9))

    router._account_queued_token_usage(unit, session_id)
    router._clean_unit(unit)

    assert unit.output_queue.empty()
    assert unit.text_output_queue.empty()

    unit.service.unregister(session_id)
    assert totals(unit.service) == (9, 9)
