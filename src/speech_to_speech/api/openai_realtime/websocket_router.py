import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from queue import Empty, Queue
from threading import Event as ThreadingEvent
from typing import Any, Callable, TypeVar

import numpy as np
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from openai.types.realtime import (
    ConversationItemCreateEvent,
    ConversationItemTruncateEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferCommitEvent,
    OutputAudioBufferClearEvent,
    ResponseCancelEvent,
    ResponseCreateEvent,
    SessionUpdateEvent,
)

from speech_to_speech.api.openai_realtime.llm_proxy import LLMProxyConfig, mount_llm_proxy
from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit, SessionState
from speech_to_speech.api.openai_realtime.service import (
    PIPELINE_SAMPLE_RATE,
    build_error_event,
)
from speech_to_speech.api.openai_realtime.transports import (
    SessionTransport,
    WebSocketTransport,
    send_ws_event,
)
from speech_to_speech.pipeline.control import SESSION_END, PipelineControlMessage, is_control_message
from speech_to_speech.pipeline.events import (
    AssistantOutputEvent,
    AssistantResponseDoneEvent,
    AssistantToolCallReadyEvent,
    AudioInputCompletedEvent,
    PartialTranscriptionEvent,
    PipelineEvent,
    ResponseFailedEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TokenUsageEvent,
    TranscriptionCompletedEvent,
)
from speech_to_speech.pipeline.log_context import pipeline_log_ctx
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, PIPELINE_END, AudioOutput

# aiortc (the 'webrtc' extra) is optional. Import it here, at module load,
# rather than lazily in the calls endpoint: the av/cryptography C extensions
# take up to a second to load cold, which would block the shared event loop —
# and every live conversation's audio — on the first WebRTC handshake.
try:
    from aiortc import RTCPeerConnection

    from speech_to_speech.api.openai_realtime.webrtc_session import (
        WebRTCSession,
        rtc_configuration_from_env,
    )

    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

logger = logging.getLogger(__name__)
MAX_AUDIO_BATCH_BYTES = 6400
# How long the release path waits for SESSION_END to propagate through the
# handler chain back to output_queue before warning that the unit is stuck.
# Tests monkeypatch this to a small value since their fixtures usually skip
# the real handler chain.
SESSION_END_DRAIN_TIMEOUT_S = 10.0
# Past this, the unit is quarantined: its service session is unregistered
# (closing the chat so late handler output can't mutate or bill it), but the
# unit stays unclaimable. Releasing it instead would let a new client claim a
# unit whose handlers may still emit the previous session's output (e.g. a
# transcript, which carries no session identity and would be appended to the
# new session's conversation) — a cross-session leak. If SESSION_END does
# eventually drain, the chain has proven itself clean and the unit returns to
# the pool; a dead handler keeps it quarantined forever, visible in /v1/pool
# as "stuck".
SESSION_END_QUARANTINE_TIMEOUT_S = 180.0
QItem = TypeVar("QItem")


def _keep_cancel_bookkeeping(item: Any) -> bool:
    # Usage and lifecycle sentinels must survive cancellation queue flushes.
    # Dropping SESSION_END would leave the release path waiting forever.
    return isinstance(item, TokenUsageEvent) or _is_audio_done(item) or is_control_message(item, SESSION_END.kind)


def _keep_user_text_event(item: Any) -> bool:
    return isinstance(
        item,
        (
            SpeechStoppedEvent,
            PartialTranscriptionEvent,
            TranscriptionCompletedEvent,
            AudioInputCompletedEvent,
        ),
    )


def _keep_pipeline_control(item: Any) -> bool:
    return isinstance(item, (PipelineControlMessage, bytes))


def _audio_payload(item: Any) -> Any:
    return item.audio if isinstance(item, AudioOutput) else item


def _audio_generation(item: Any) -> int | None:
    return item.cancel_generation if isinstance(item, AudioOutput) else None


def _audio_response_key(item: Any) -> str | None:
    return item.response_key if isinstance(item, AudioOutput) else None


def _audio_cleanup_only(item: Any) -> bool:
    return item.cleanup_only if isinstance(item, AudioOutput) else False


_RESPONSE_PIPELINE_EVENTS = (
    AssistantOutputEvent,
    AssistantResponseDoneEvent,
    ResponseFailedEvent,
)


def _keep_non_audio_output(item: Any) -> bool:
    """Preserve response bookkeeping when WebRTC clears buffered audio."""
    return _keep_cancel_bookkeeping(item) or isinstance(item, _RESPONSE_PIPELINE_EVENTS)


def _response_event_key(item: Any) -> str | None:
    if isinstance(item, _RESPONSE_PIPELINE_EVENTS):
        return item.response_key
    return None


def _response_key_is_obsolete(unit: PipelineUnit, session_id: str, response_key: str | None) -> bool:
    """Whether *response_key* belongs to a closed response rather than a queued one."""
    if response_key is None:
        return False
    st = unit.service._state(session_id)
    if response_key in st.closed_response_keys:
        return True
    return (
        st.in_response
        and st.current_response_key not in (None, response_key)
        and response_key not in st.pending_response_keys
    )


def _output_response_key(item: Any) -> str | None:
    if isinstance(item, AudioOutput):
        return item.response_key
    if isinstance(item, PipelineEvent):
        return getattr(item, "response_key", None)
    return None


def _response_key_output_is_blocked(
    unit: PipelineUnit,
    session_id: str,
    response_key: str | None,
) -> bool:
    if response_key is None:
        return False
    return unit.service.response.is_response_output_blocked(session_id, response_key)


def _discard_obsolete_response_key(unit: PipelineUnit, session_id: str, response_key: str | None) -> None:
    if response_key is None:
        return
    unit.service.close_response_key(session_id, response_key)
    logger.debug("Pipeline %d: discarded obsolete response %s output", unit.index, response_key)


def _flush_queue(
    q: Queue[QItem],
    *,
    preserve: Callable[[QItem], bool] | None = None,
    on_discard: Callable[[QItem], None] | None = None,
) -> None:
    """Drain a queue, optionally preserving items matching *preserve*.

    Preserved items are re-inserted at the **front** of the queue
    (atomically under the queue's mutex) so they are processed before
    anything a pipeline thread may have enqueued during the drain.
    """
    preserved: list[QItem] = []
    while True:
        try:
            item = q.get_nowait()
            if preserve and preserve(item):
                preserved.append(item)
            elif on_discard is not None:
                on_discard(item)
        except Empty:
            break
    if preserved:
        with q.mutex:
            for item in reversed(preserved):
                q.queue.appendleft(item)
            q.not_empty.notify(len(preserved))


def _clean_unit(
    unit: PipelineUnit,
    preserve: Callable[[Any], bool] | None = None,
    on_discard: Callable[[Any], None] | None = None,
) -> None:
    """Cancel in-flight work and flush queues for a single pipeline unit.

    All four pipeline queues are drained — input audio, transcript-to-LM,
    LM-to-TTS output, and the text-event side channel — so pending work from
    a released session cannot be picked up by handlers and leak into the next
    session that claims this unit. SESSION_END is enqueued by the route
    handler *after* this returns to serve as the soft reset signal for
    stateful handlers.
    """
    unit.cancel_scope.cancel()
    _flush_queue(unit.input_queue)
    _flush_queue(unit.text_prompt_queue)
    _flush_queue(unit.output_queue, preserve=preserve, on_discard=on_discard)
    _flush_queue(unit.text_output_queue, preserve=preserve, on_discard=on_discard)
    unit.response_playing.clear()
    unit.cancel_scope.reset()
    unit.should_listen.set()


def _to_audio_bytes(chunk: Any) -> bytes:
    chunk = _audio_payload(chunk)
    if isinstance(chunk, PipelineControlMessage):
        raise TypeError(f"unexpected control message on audio output queue: {chunk!r}")
    if isinstance(chunk, np.ndarray) or hasattr(chunk, "tobytes"):
        return chunk.tobytes()
    return chunk


def _is_audio_done(item: Any) -> bool:
    payload = _audio_payload(item)
    return isinstance(payload, bytes) and payload == AUDIO_RESPONSE_DONE


def _is_pipeline_end(item: Any) -> bool:
    payload = _audio_payload(item)
    return isinstance(payload, bytes) and payload == PIPELINE_END


def _generation_is_discardable(unit: PipelineUnit, generation: int | None) -> bool:
    """Whether output tagged with *generation* should be dropped.

    A generation is discardable if it has been superseded (``is_stale``) or if the
    cancel scope is in its post-cancel discard window and this is not the current
    live generation. Shared by audio and assistant-text so the two paths stay in
    lockstep: dropping text whenever ``discarding`` is set (without this generation
    check) silently swallows the transcript of a fresh response when ``discarding``
    lingers — e.g. a superseded speculative turn whose TTS never emitted an
    AUDIO_RESPONSE_DONE sentinel, so response_done() never cleared the flag.
    """
    if generation is not None and unit.cancel_scope.is_stale(generation):
        return True
    if unit.cancel_scope.discarding and generation != unit.cancel_scope.generation:
        return True
    return False


def _should_discard_audio(unit: PipelineUnit, item: Any) -> bool:
    return _generation_is_discardable(unit, _audio_generation(item))


def _safe_unregister(unit: PipelineUnit, session_id: str) -> None:
    try:
        unit.service.unregister(session_id)
    except Exception:
        logger.exception(f"Pipeline {unit.index}: unregister failed for session {session_id}")


async def _release_unit_after_drain(unit: PipelineUnit, session: Any, session_id: str) -> None:
    """Wait for SESSION_END to propagate, then release the unit.

    Runs in its own asyncio task so the route handler's finally block can return
    immediately. The unit stays unavailable for new claims (unit.session != None)
    until SESSION_END travels all the way through the handler chain back to
    output_queue — observed by the send loop, which sets session.drained.

    Past SESSION_END_QUARANTINE_TIMEOUT_S (a wedged or dead handler thread) the
    unit is quarantined, NOT released: still-running handlers could emit the old
    session's output (transcripts carry no session identity) into whichever
    session claimed the unit next, and a dead handler would make the unit accept
    clients it can never serve. The session is unregistered right away so late
    output can't mutate or bill the closed conversation; the unit itself only
    returns to the pool if SESSION_END eventually drains, proving the chain is
    clean. Operators can spot quarantined units in `/v1/pool` (state "stuck").
    """
    elapsed = 0.0
    warned = False
    try:
        while not session.drained.is_set():
            await asyncio.sleep(0.05)
            elapsed += 0.05
            if not warned and elapsed >= SESSION_END_DRAIN_TIMEOUT_S:
                logger.warning(
                    f"Pipeline {unit.index}: SESSION_END not drained after {elapsed:.1f}s — "
                    f"unit will remain unavailable until handlers finish (session {session_id})"
                )
                warned = True
            if session.quarantined_at is None and elapsed >= SESSION_END_QUARANTINE_TIMEOUT_S:
                session.quarantined_at = time.monotonic()
                _safe_unregister(unit, session_id)
                logger.error(
                    f"Pipeline {unit.index}: SESSION_END still not drained after {elapsed:.0f}s — "
                    f"quarantining unit until the handler chain drains (session {session_id})"
                )
    finally:
        # Runs when the drain completed (chain proven clean) or the task is
        # cancelled at shutdown. Release unconditionally: even if unregister
        # raises, the unit must not stay claimed forever.
        try:
            _safe_unregister(unit, session_id)
        finally:
            unit.session = None
        recovered = " after quarantine" if session.quarantined_at is not None else ""
        logger.info(f"Pipeline {unit.index} released{recovered} (session {session_id} ended)")


# Strong references to in-flight drain-and-release tasks (asyncio only
# holds tasks weakly); each task removes itself on completion.
_release_tasks: set[asyncio.Task[None]] = set()


def _release_session(unit: PipelineUnit, session_id: str) -> None:
    """Start the release of a unit after its client disconnected.

    Shared by the WebSocket route's finally block and the WebRTC session's
    close callback. Marks the session as released, resets the unit, enqueues
    SESSION_END, and spawns the drain-and-release task — the unit stays
    claimed until SESSION_END propagates back to output_queue.
    """
    old_session = unit.session
    if old_session is None:
        # Already released (e.g. duplicate close callbacks racing).
        return
    old_session.released_at = time.monotonic()
    # The send loop can be parked on output from an unclaimed internal
    # prefetch. Invalidate that response while its connection state is still
    # registered, and drop the per-session held item so SESSION_END can drain.
    try:
        unit.service.close_pending_responses(session_id)
    except KeyError:
        pass

    def account_usage(item: Any) -> None:
        if not isinstance(item, TokenUsageEvent):
            return
        try:
            unit.service.dispatch_pipeline_event(session_id, item)
        except KeyError:
            # A duplicate close callback may race the drain task's unregister.
            logger.debug("Skipped late usage for unregistered session %s", session_id)

    if old_session.pending_output_item is not None:
        account_usage(old_session.pending_output_item)
        old_session.pending_output_item = None
    for item in old_session.pending_text_output_items:
        account_usage(item)
    old_session.pending_text_output_items.clear()
    _clean_unit(unit, on_discard=account_usage)
    # Tag SESSION_END with this session's id so that, after a force
    # release, a late arrival can't satisfy the next session's drain.
    unit.input_queue.put(PipelineControlMessage(SESSION_END.kind, session_id=session_id))
    task = asyncio.create_task(_release_unit_after_drain(unit, old_session, session_id))
    _release_tasks.add(task)
    task.add_done_callback(_release_tasks.discard)


async def _dispatch_client_event(
    unit: PipelineUnit,
    session_id: str,
    raw: dict[str, Any],
    transport: SessionTransport,
    *,
    transport_kind: str = "websocket",
) -> None:
    """Parse and apply one client event, replying over *transport*.

    Shared by both transports; ``transport_kind`` gates the events whose
    validity depends on how audio travels: ``input_audio_buffer.append`` is
    WebSocket-only (WebRTC audio arrives on the media track), and
    ``output_audio_buffer.clear`` is WebRTC-only (over WebSocket the unplayed
    audio sits client-side).
    """
    service = unit.service
    client_event_id = raw.get("event_id")

    async def send_correlated(events: list[Any]) -> None:
        if isinstance(client_event_id, str):
            for outgoing in events:
                if getattr(outgoing, "type", None) == "error":
                    outgoing.error.event_id = client_event_id
        await transport.send_events(events)

    event = service.parse_client_event(raw)
    if event is None:
        await send_correlated(
            [service.make_error(f"Unknown or invalid event: {raw.get('type')}", "unknown_or_invalid_event")]
        )
        return

    if isinstance(event, InputAudioBufferAppendEvent):
        if transport_kind == "webrtc":
            await send_correlated(
                [
                    service.make_error(
                        "In WebRTC mode audio arrives via the media track; input_audio_buffer.append is not supported.",
                        "invalid_event_for_transport",
                    )
                ]
            )
            return
        chunks = service.handle_audio_append(session_id, event)
        rt_cfg = service._state(session_id).runtime_config
        for chunk in chunks:
            unit.input_queue.put((chunk, rt_cfg))

    elif isinstance(event, InputAudioBufferCommitEvent):
        err = service.handle_audio_commit(session_id)
        if err:
            await send_correlated([err])

    elif isinstance(event, OutputAudioBufferClearEvent):
        if transport_kind != "webrtc":
            await send_correlated(
                [
                    service.make_error(
                        "output_audio_buffer.clear is only supported on the WebRTC transport.",
                        "invalid_event_for_transport",
                    )
                ]
            )
            return
        _flush_queue(unit.output_queue, preserve=_keep_non_audio_output)
        transport.discard_pending_audio()

    elif isinstance(event, SessionUpdateEvent):
        err = service.handle_session_update(session_id, event)
        if err:
            await send_correlated([err])
        else:
            await send_correlated([service.build_session_updated(session_id)])

    elif isinstance(event, ConversationItemCreateEvent):
        events = service.handle_conversation_item_create(session_id, event)
        if events:
            await send_correlated(events)

    elif isinstance(event, ConversationItemTruncateEvent):
        # The stock Agents SDK sends this after an audible WebSocket
        # interruption. An explicit response.cancel or automatic server-VAD
        # cancellation has already discarded provisional generation, and
        # client-side playback owns the unheard tail, so there is no additional
        # server state to mutate.
        logger.debug("Accepted conversation.item.truncate for %s", event.item_id)

    elif isinstance(event, ResponseCreateEvent):
        result = service.handle_response_create(session_id, event)
        if result:
            response_key = None
            if result.type != "error":
                unit.cancel_scope.new_response()
                response_key = service._state(session_id).current_response_key
            await send_correlated([result])
            if result.type == "response.created":
                service.response.mark_response_created_sent(session_id, response_key)

    elif isinstance(event, ResponseCancelEvent):
        st = service._state(session_id)
        had_response = st.in_response or st.response_pending
        if had_response:
            unit.cancel_scope.cancel()
            _flush_queue(unit.text_prompt_queue, preserve=_keep_pipeline_control)
        _flush_queue(unit.output_queue, preserve=_keep_cancel_bookkeeping)
        _flush_queue(unit.text_output_queue, preserve=_keep_user_text_event)
        transport.discard_pending_audio()
        events = service.handle_response_cancel(session_id)
        if events:
            await send_correlated(events)
        unit.response_playing.clear()


def create_app(
    pool: list[PipelineUnit],
    stop_event: ThreadingEvent,
    llm_proxy_config: LLMProxyConfig | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # One send loop per pipeline unit; each polls its own queues and forwards
        # to the websocket currently attached via unit.session.
        send_tasks = [asyncio.create_task(_send_loop_for(unit)) for unit in pool]
        yield
        for task in send_tasks:
            task.cancel()
        for task in send_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        for unit in pool:
            sess = unit.session
            if sess is not None and sess.transport is not None:
                try:
                    await sess.transport.close()
                except Exception:
                    pass

    app = FastAPI(lifespan=lifespan)

    llm_proxy_usage = mount_llm_proxy(app, llm_proxy_config)

    def _claim_unit(transport: SessionTransport | None) -> PipelineUnit | None:
        """Atomically (between asyncio yield points) reserve the first idle unit.

        Creates a placeholder SessionState that the caller fills in with the
        session_id after RealtimeService.register(). The WebRTC route claims
        with transport=None and attaches the session object once constructed.
        """
        for unit in pool:
            if unit.session is None:
                unit.session = SessionState(transport=transport)
                return unit
        return None

    @app.websocket("/v1/realtime")
    async def realtime_endpoint(ws: WebSocket) -> None:
        offered_subprotocols = {
            protocol.strip() for protocol in ws.headers.get("sec-websocket-protocol", "").split(",")
        }
        await ws.accept(subprotocol="realtime" if "realtime" in offered_subprotocols else None)

        transport = WebSocketTransport(ws)
        unit = _claim_unit(transport)
        if unit is None:
            logger.warning(f"Rejected connection: all {len(pool)} pipeline slots in use")
            # Stateless error event — rejection is not chargeable to any unit's usage metrics.
            await send_ws_event(
                ws,
                build_error_event(
                    f"All {len(pool)} session slots are in use. Disconnect an existing client first.",
                    error_type="session_limit_reached",
                ),
            )
            await ws.close(code=1008, reason="All session slots are in use")
            return

        pipeline_log_ctx.set(unit.index)
        # _claim_unit guarantees unit.session is not None for the returned unit.
        assert unit.session is not None
        # Everything after the claim runs inside try so the finally below always
        # releases the unit, even if session setup fails.
        session_id = ""
        try:
            session_id = unit.service.register()
            unit.session.session_id = session_id
            logger.info(f"Client connected to pipeline {unit.index} (session {session_id})")

            # Defensive: drain edge queues and reset events so stale data from a
            # previous session that survived SESSION_END propagation doesn't leak.
            _clean_unit(unit)

            await send_ws_event(ws, unit.service.build_session_created(session_id))

            while not stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.receive_json(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                await _dispatch_client_event(unit, session_id, raw, transport)

        except WebSocketDisconnect:
            logger.info(f"Client {session_id} disconnected from pipeline {unit.index}")
        except Exception as e:
            logger.error(f"Client {session_id} on pipeline {unit.index} error: {type(e).__name__}: {e}", exc_info=True)
        finally:
            # Hold the session reference: the send loop's snapshot will still resolve
            # to this object until we clear unit.session, so any handler output that
            # arrives during the drain window is sent to the now-closed ws (silently
            # dropped) instead of leaking to whichever client claims this unit next.
            # _release_session spawns the drain-and-release as a separate task so
            # this finally returns immediately. Awaiting here is unreliable: after
            # WebSocketDisconnect propagates, subsequent awaits in the same task
            # can be skipped/cancelled by Starlette's runner and never resume.
            _release_session(unit, session_id)

    @app.get("/v1/usage")
    async def usage_endpoint() -> dict[str, Any]:
        # Aggregate usage across the pool. Numeric fields sum; dict fields (e.g.
        # errors_by_type) merge with numeric leaves summed too, so per-unit error
        # counts don't get dropped by the first-unit's value.
        def _merge(into: dict[str, Any], src: dict[str, Any]) -> None:
            for k, v in src.items():
                if isinstance(v, (int, float)):
                    into[k] = into.get(k, 0) + v
                elif isinstance(v, dict):
                    sub = into.setdefault(k, {})
                    if isinstance(sub, dict):
                        _merge(sub, v)
                else:
                    into.setdefault(k, v)

        total: dict[str, Any] = {}
        for unit in pool:
            _merge(total, unit.service.get_usage())
        # Additive section: proxy traffic is app-level, not per-unit, so it
        # lands after the per-unit merge and never collides with unit keys.
        total["llm_proxy"] = llm_proxy_usage.model_dump()
        return total

    @app.get("/v1/pool")
    async def pool_endpoint() -> dict[str, Any]:
        now = time.monotonic()

        def _state(u: PipelineUnit) -> dict[str, Any]:
            s = u.session
            if s is None:
                return {"index": u.index, "state": "idle", "session_id": None}
            if s.released_at is None:
                return {"index": u.index, "state": "active", "session_id": s.session_id}
            # Drain wait gave up (quarantine timeout): the unit stays occupied
            # until SESSION_END actually drains — possibly forever if a handler
            # thread died. Surfaced distinctly so operators can act on it.
            if s.quarantined_at is not None:
                return {
                    "index": u.index,
                    "state": "stuck",
                    "session_id": s.session_id,
                    "draining_for_s": round(now - s.released_at, 2),
                    "stuck_for_s": round(now - s.quarantined_at, 2),
                }
            # released by client but SESSION_END hasn't drained yet → unit
            # is still occupied; surface elapsed time so operators can spot
            # stuck handlers.
            return {
                "index": u.index,
                "state": "draining",
                "session_id": s.session_id,
                "draining_for_s": round(now - s.released_at, 2),
            }

        return {
            "size": len(pool),
            "in_use": sum(1 for u in pool if u.session is not None),
            "units": [_state(u) for u in pool],
        }

    @app.post("/v1/realtime/calls")
    async def webrtc_calls_endpoint(request: Request) -> Response:
        """WebRTC SDP handshake (OpenAI GA Realtime 'calls' endpoint).

        The client POSTs an SDP offer with Content-Type: application/sdp and
        receives an SDP answer. Audio then flows over WebRTC media tracks;
        events flow over the 'oai-events' data channel using the same JSON
        protocol as the WebSocket transport.
        """
        if not WEBRTC_AVAILABLE:
            return Response(
                content="WebRTC support requires the 'webrtc' extra: pip install 'speech-to-speech[webrtc]'",
                status_code=501,
                media_type="text/plain",
            )

        if "application/sdp" not in request.headers.get("content-type", ""):
            return Response(
                content="Content-Type must be application/sdp",
                status_code=415,
                media_type="text/plain",
            )
        offer_sdp = (await request.body()).decode("utf-8")

        # Claim with a placeholder transport; the send loop tolerates a
        # transport-less snapshot until the session object below is attached.
        unit = _claim_unit(None)
        if unit is None:
            logger.warning(f"Rejected WebRTC offer: all {len(pool)} pipeline slots in use")
            return Response(
                content=build_error_event(
                    f"All {len(pool)} session slots are in use. Disconnect an existing client first.",
                    error_type="session_limit_reached",
                ).model_dump_json(),
                status_code=503,
                media_type="application/json",
            )

        pipeline_log_ctx.set(unit.index)
        try:
            session_id = unit.service.register()
            assert unit.session is not None
            unit.session.session_id = session_id
            logger.info(f"WebRTC client claiming pipeline {unit.index} (session {session_id})")

            # Defensive: drain edge queues and reset events so stale data from a
            # previous session that survived SESSION_END propagation doesn't leak.
            _clean_unit(unit)
        except Exception as e:  # noqa: BLE001
            logger.error(f"WebRTC call setup failed (pipeline {unit.index}): {type(e).__name__}: {e}")
            # No transport or drain task exists yet, so undoing the claim
            # directly is the whole release.
            unit.session = None
            return Response(content="WebRTC session setup failed", status_code=500, media_type="text/plain")

        released = False

        def _on_closed() -> None:
            # close() is idempotent but can be reached from several aiortc
            # callbacks; release the unit exactly once.
            nonlocal released
            if released:
                return
            released = True
            logger.info(f"WebRTC client {session_id} disconnected from pipeline {unit.index}")
            _release_session(unit, session_id)

        async def _on_client_event(raw: dict[str, Any]) -> None:
            assert session is not None  # callbacks only fire after setup()
            await _dispatch_client_event(unit, session_id, raw, session, transport_kind="webrtc")

        def _on_audio(pcm: bytes) -> None:
            chunks = unit.service.append_pcm(session_id, pcm, PIPELINE_SAMPLE_RATE)
            if not chunks:
                return
            rt_cfg = unit.service._state(session_id).runtime_config
            for chunk in chunks:
                unit.input_queue.put((chunk, rt_cfg))

        async def _on_open() -> None:
            assert session is not None  # callbacks only fire after setup()
            await session.send_events([unit.service.build_session_created(session_id)])
            logger.info(f"WebRTC session.created sent (session {session_id})")

        # Any failure between the claim above and a successful negotiate()
        # must release the unit, or it stays occupied forever with no peer
        # attached — the connect watchdog only exists once negotiate() ran.
        session = None
        try:
            config = rtc_configuration_from_env()
            pc = RTCPeerConnection(configuration=config) if config is not None else RTCPeerConnection()
            session = WebRTCSession(
                pc,
                on_client_event=_on_client_event,
                on_audio=_on_audio,
                on_open=_on_open,
                on_closed=_on_closed,
            )
            session.setup()
            unit.session.transport = session
        except Exception as e:  # noqa: BLE001
            logger.error(f"WebRTC session setup failed (session {session_id}): {type(e).__name__}: {e}")
            if session is not None:
                await session.close()  # fires _on_closed → _release_session
            else:
                _on_closed()
            return Response(content="WebRTC session setup failed", status_code=500, media_type="text/plain")

        try:
            answer_sdp = await session.negotiate(offer_sdp)
        except Exception as e:  # noqa: BLE001
            logger.error(f"WebRTC negotiation failed (session {session_id}): {type(e).__name__}: {e}")
            await session.close()
            return Response(content="Invalid SDP offer", status_code=400, media_type="text/plain")

        logger.info(f"WebRTC SDP answer returned (session {session_id})")
        return Response(
            content=answer_sdp,
            status_code=201,
            media_type="application/sdp",
            headers={"Location": f"/v1/realtime/calls/{session_id}"},
        )

    @app.delete("/v1/realtime/calls/{call_id}")
    async def webrtc_hangup_endpoint(call_id: str) -> Response:
        """Hang up a WebRTC call — the Location URL advertised by the POST above."""
        for unit in pool:
            session = unit.session
            if (
                session is None
                or session.session_id != call_id
                or session.released_at is not None
                or session.transport is None
                or session.transport.kind != "webrtc"
            ):
                continue
            logger.info(f"WebRTC call {call_id} hung up via DELETE (pipeline {unit.index})")
            # close() fires the session's on_closed callback, which releases
            # the unit exactly once (idempotent with aiortc's own callbacks).
            await session.transport.close()
            return Response(status_code=200)
        return Response(content="Unknown call", status_code=404, media_type="text/plain")

    async def _send_loop_for(unit: PipelineUnit) -> None:
        """Per-pipeline send loop. Polls this unit's output queues and forwards
        to the transport currently attached via unit.session.

        Per-session scratch (pending_output_item) lives on SessionState, so it
        disappears together with the transport when the session is released —
        no stale sentinel can leak into the next claim.
        """
        pipeline_log_ctx.set(unit.index)
        while not stop_event.is_set():
            try:
                # Snapshot the session once per iteration; if the route releases the
                # unit mid-iteration, we continue against the prior snapshot which is
                # consistent (its transport is still valid until close() returns).
                session = unit.session
                transport = session.transport if session is not None else None
                session_id = session.session_id if session is not None else None

                # Text events first (speech_started cancels active response).
                try:
                    text_msg = None
                    if session is not None and session_id is not None:
                        for index, pending in enumerate(session.pending_text_output_items):
                            if not _response_key_output_is_blocked(
                                unit,
                                session_id,
                                _output_response_key(pending),
                            ):
                                text_msg = session.pending_text_output_items.pop(index)
                                break
                    if text_msg is None:
                        text_msg = unit.text_output_queue.get_nowait()

                    if (
                        session is not None
                        and session_id is not None
                        and _response_key_output_is_blocked(
                            unit,
                            session_id,
                            _output_response_key(text_msg),
                        )
                    ):
                        # Response-dependent side-channel events share the same
                        # exposure barrier as audio/output events. In particular,
                        # an early tool call must never overtake response.created.
                        # Unlike the serial output hold, this list does not stall
                        # the origin response whose completion enables the claim.
                        session.pending_text_output_items.append(text_msg)
                        text_msg = None
                    if text_msg is None:
                        raise Empty
                    if isinstance(text_msg, AssistantToolCallReadyEvent):
                        generation = text_msg.cancel_generation
                        response_key = text_msg.response_key
                        if _generation_is_discardable(unit, generation):
                            continue
                        if session_id is not None and _response_key_is_obsolete(unit, session_id, response_key):
                            _discard_obsolete_response_key(unit, session_id, response_key)
                            continue
                    is_speech_start = isinstance(text_msg, SpeechStartedEvent)

                    was_in_response = False
                    was_response_pending = False
                    if is_speech_start and session_id:
                        st = unit.service._state(session_id)
                        was_in_response = st.in_response
                        was_response_pending = st.response_pending

                    if transport is not None and isinstance(text_msg, PipelineEvent) and session_id:
                        events = unit.service.dispatch_pipeline_event(session_id, text_msg)
                        if events:
                            await transport.send_events(events)

                    if isinstance(text_msg, SpeechStartedEvent) and session_id:
                        active_cfg = unit.service._state(session_id).runtime_config
                        interrupt_enabled = text_msg.interrupt_response and (
                            active_cfg is None or active_cfg.interrupt_response_enabled
                        )
                        if interrupt_enabled and transport is not None:
                            # Flush even when no response is active: the WebRTC
                            # track can still hold unplayed audio from a response
                            # whose done-sentinel was already observed —
                            # finish_response() runs on the sentinel, not when
                            # playback completes. No-op over WebSocket.
                            transport.discard_pending_audio()
                        if was_in_response or was_response_pending:
                            if interrupt_enabled:
                                unit.cancel_scope.cancel()
                                unit.service.close_pending_responses(session_id)
                                _flush_queue(unit.text_prompt_queue, preserve=_keep_pipeline_control)
                                _flush_queue(unit.output_queue, preserve=_keep_cancel_bookkeeping)
                                _flush_queue(unit.text_output_queue, preserve=_keep_user_text_event)
                                if unit.response_playing.is_set():
                                    unit.response_playing.clear()
                                logger.info(
                                    "Pipeline %d: speech during %s: cancelled, queue flushed",
                                    unit.index,
                                    "response" if was_in_response else "pending response",
                                )
                            else:
                                logger.info(
                                    f"Pipeline {unit.index}: speech during response: interrupt_response disabled, ignoring"
                                )
                except Empty:
                    pass

                try:
                    if session is not None and session.pending_output_item is not None:
                        audio_chunk = session.pending_output_item
                        session.pending_output_item = None
                    else:
                        audio_chunk = unit.output_queue.get_nowait()

                    if (
                        session is not None
                        and session_id is not None
                        and _response_key_output_is_blocked(
                            unit,
                            session_id,
                            _output_response_key(audio_chunk),
                        )
                    ):
                        # Generation and TTS may complete before the client sends
                        # response.create, or before response.created finishes
                        # sending. Keep every lifecycle event private until the
                        # response is publicly announced.
                        session.pending_output_item = audio_chunk
                        await asyncio.sleep(0.01)
                        continue

                    if isinstance(audio_chunk, TokenUsageEvent):
                        if transport is not None and session_id is not None:
                            await transport.send_events(unit.service.dispatch_pipeline_event(session_id, audio_chunk))
                        continue

                    if isinstance(audio_chunk, _RESPONSE_PIPELINE_EVENTS):
                        generation = getattr(audio_chunk, "cancel_generation", None)
                        response_key = _response_event_key(audio_chunk)
                        if _generation_is_discardable(unit, generation):
                            continue
                        if session_id is not None and _response_key_is_obsolete(unit, session_id, response_key):
                            _discard_obsolete_response_key(unit, session_id, response_key)
                            continue
                        if transport is not None and session_id is not None:
                            await transport.send_events(unit.service.dispatch_pipeline_event(session_id, audio_chunk))
                        continue

                    if _is_pipeline_end(audio_chunk):
                        if transport is not None and session_id:
                            await transport.send_events(unit.service.finish_response(session_id))
                        break

                    if _is_audio_done(audio_chunk):
                        audio_generation = _audio_generation(audio_chunk)
                        response_key = _audio_response_key(audio_chunk)
                        if _audio_cleanup_only(audio_chunk):
                            if response_key is None:
                                logger.warning("Ignoring unkeyed stale response cleanup terminal")
                                continue
                            cleaned_active_response = False
                            if session_id:
                                st = unit.service._state(session_id)
                                if st.in_response and st.current_response_key in (None, response_key):
                                    cleaned_active_response = True
                                    events = unit.service.finish_response(
                                        session_id,
                                        status="cancelled",
                                        response_key=response_key,
                                    )
                                    if transport is not None and events:
                                        await transport.send_events(events)
                                else:
                                    unit.service.close_response_key(session_id, response_key)
                                if cleaned_active_response:
                                    unit.response_playing.clear()
                                if not (st.in_response or st.response_pending):
                                    unit.should_listen.set()
                            unit.cancel_scope.response_done(audio_generation)
                            logger.info(
                                "Pipeline %d: stale response lifecycle cleaned up",
                                unit.index,
                            )
                            continue
                        if audio_generation is not None and unit.cancel_scope.is_stale(audio_generation):
                            if session_id:
                                unit.service.close_response_key(session_id, response_key)
                            unit.cancel_scope.response_done(audio_generation)
                            unit.should_listen.set()
                            logger.info(f"Pipeline {unit.index}: stale response complete, listening re-enabled")
                            continue
                        if session_id is not None and _response_key_is_obsolete(unit, session_id, response_key):
                            _discard_obsolete_response_key(unit, session_id, response_key)
                            continue
                        if transport is not None and session_id:
                            await transport.send_events(
                                unit.service.finish_response(session_id, response_key=response_key)
                            )
                        if session_id:
                            unit.service._state(session_id).clear_pending_response(response_key)
                        unit.response_playing.clear()
                        unit.cancel_scope.response_done(audio_generation)
                        unit.should_listen.set()
                        logger.info(f"Pipeline {unit.index}: response complete, listening re-enabled")
                        continue

                    # SESSION_END travels from input_queue through every handler to
                    # output_queue. Observing it here means the chain has fully reset;
                    # signal the release path so it can clear unit.session. A tag from
                    # another session means the emitting session was force-released —
                    # its late SESSION_END must not stand in for this session's drain.
                    if is_control_message(audio_chunk, SESSION_END.kind):
                        chunk_session_id = getattr(audio_chunk, "session_id", None)
                        if session is not None and chunk_session_id in (None, session.session_id):
                            session.drained.set()
                            logger.debug(f"Pipeline {unit.index}: SESSION_END drained")
                        continue

                    if is_control_message(audio_chunk):
                        continue

                    if _should_discard_audio(unit, audio_chunk):
                        continue

                    response_key = _audio_response_key(audio_chunk)
                    if session_id is not None and _response_key_is_obsolete(unit, session_id, response_key):
                        _discard_obsolete_response_key(unit, session_id, response_key)
                        continue

                    audio_chunk = _to_audio_bytes(audio_chunk)

                    audio_batch = bytearray(audio_chunk)
                    while len(audio_batch) < MAX_AUDIO_BATCH_BYTES:
                        try:
                            next_chunk = unit.output_queue.get_nowait()
                        except Empty:
                            break

                        if (
                            _is_pipeline_end(next_chunk)
                            or _is_audio_done(next_chunk)
                            or isinstance(next_chunk, PipelineEvent)
                            or is_control_message(next_chunk, SESSION_END.kind)
                        ):
                            # Only stash if we still have a session; otherwise drop it.
                            if session is not None:
                                session.pending_output_item = next_chunk
                            break

                        if _should_discard_audio(unit, next_chunk):
                            continue

                        if _audio_response_key(next_chunk) != response_key:
                            if session is not None:
                                session.pending_output_item = next_chunk
                            break

                        next_audio = _to_audio_bytes(next_chunk)
                        if len(audio_batch) + len(next_audio) > MAX_AUDIO_BATCH_BYTES:
                            if session is not None:
                                session.pending_output_item = next_chunk
                            break
                        audio_batch.extend(next_audio)

                    if not unit.response_playing.is_set():
                        unit.response_playing.set()
                        unit.should_listen.set()

                    if transport is not None and session_id:
                        await transport.send_audio_chunk(
                            unit.service,
                            session_id,
                            bytes(audio_batch),
                            response_key,
                        )
                except Empty:
                    pass

                await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pipeline {unit.index} send loop error: {e}")
                await asyncio.sleep(0.1)

    return app
