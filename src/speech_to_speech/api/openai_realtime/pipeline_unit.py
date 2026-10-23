import asyncio
from queue import Queue
from threading import Event
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.transports import SessionTransport
from speech_to_speech.pipeline.cancel_scope import CancelScope


class OutputSignal:
    """Level-triggered wakeup from producer threads to the async send loop.

    Producers (pipeline handler threads) call :meth:`notify_threadsafe` after
    enqueueing work; the send loop awaits :meth:`wait` instead of polling the
    queues on a fixed timer (#692). Units are built before the event loop
    starts, so the loop binds late via :meth:`bind_loop`; notifications
    arriving earlier set a missed flag that ``bind_loop`` replays.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._missed = False
        self._event = asyncio.Event()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the sending loop and replay any notification that arrived early."""
        self._loop = loop
        if self._missed:
            self._missed = False
            self._event.set()

    def notify_threadsafe(self) -> None:
        """Wake the send loop from any thread. Safe before the loop binds."""
        loop = self._loop
        if loop is None:
            self._missed = True
            return
        loop.call_soon_threadsafe(self._event.set)

    async def wait(self) -> None:
        await self._event.wait()

    def clear(self) -> None:
        self._event.clear()


class NotifyingQueue(Queue):
    """``queue.Queue`` that wakes the send loop's waiters after every put.

    Producers keep calling the ordinary ``put``/``put_nowait`` API — only the
    queues the realtime send loop consumes are constructed as notifying.
    """

    def __init__(self, *args: Any, signal: OutputSignal, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._signal = signal

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        super().put(item, block=block, timeout=timeout)
        self._signal.notify_threadsafe()


class SessionState(BaseModel):
    """Per-client ephemeral state.

    Created when a route handler claims a PipelineUnit (WebSocket accept or
    WebRTC SDP offer); dropped when the client disconnects. Holding the
    transport reference, the service session id, and any send-loop scratch
    (pending_output_item) here ensures these fields share one lifecycle — a
    stale value can't outlive its session.

    `transport` may briefly be None while a WebRTC claim finishes
    constructing the session, so the send loop must tolerate a
    transport-less snapshot.

    `drained` is set by the send loop when SESSION_END travels through the handler
    chain back to the output queue; the release path awaits it before clearing
    `PipelineUnit.session`, so a new client cannot claim the unit until in-flight
    work from this session has fully reset.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    transport: Optional[SessionTransport] = None
    session_id: str = ""
    pending_output_item: Any = None
    # Response-keyed side-channel events can be generated before a hidden
    # prefetch is claimed. Keep them ordered and private without letting one
    # blocked event stall unrelated origin-response output.
    pending_text_output_items: list[Any] = Field(default_factory=list)
    drained: asyncio.Event = Field(default_factory=asyncio.Event)
    # Wall-clock time when the client disconnected (route handler released its
    # claim). `None` while the client is still active. Used by /v1/pool to
    # surface stuck units (handlers haven't finished propagating SESSION_END).
    released_at: Optional[float] = None
    # Wall-clock time when the drain wait gave up and quarantined the unit
    # (SESSION_END_QUARANTINE_TIMEOUT_S elapsed). The unit stays unclaimable —
    # its handlers may still emit this session's output — until SESSION_END
    # actually drains. Reported as "stuck" by /v1/pool.
    quarantined_at: Optional[float] = None


class PipelineUnit(BaseModel):
    """One isolated realtime pipeline.

    Each unit owns its queues, events, RealtimeService, and the chain of handler
    instances (VAD, STT, transcription notifier, LM, LM output processor, TTS).
    Lives inside the pool managed by RealtimeServer; the websocket route handler
    claims a free unit (`session is None`) on `accept` and releases it on disconnect
    by setting `session` back to None.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: int
    service: RealtimeService
    cancel_scope: CancelScope
    should_listen: Event
    response_playing: Event
    input_queue: Queue
    output_queue: Queue
    text_output_queue: Queue
    text_prompt_queue: Queue
    # Level-triggered wakeups for the async send loop (#692): queue puts fire
    # the output/text signals via NotifyingQueue; state transitions that may
    # release held output fire the state signal.
    output_signal: OutputSignal
    text_signal: OutputSignal
    state_signal: OutputSignal
    handlers: list[Any]

    session: Optional[SessionState] = None
