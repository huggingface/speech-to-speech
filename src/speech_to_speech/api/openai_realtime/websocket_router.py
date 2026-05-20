import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from queue import Empty, Queue
from threading import Event as ThreadingEvent
from typing import Any, Callable, TypeVar

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openai.types.realtime import (
    ConversationItemCreateEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferCommitEvent,
    ResponseCancelEvent,
    ResponseCreateEvent,
    SessionUpdateEvent,
)

from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit, SessionState
from speech_to_speech.api.openai_realtime.service import ServerEvent, build_error_event
from speech_to_speech.pipeline.control import SESSION_END, PipelineControlMessage, is_control_message
from speech_to_speech.pipeline.log_context import pipeline_log_ctx
from speech_to_speech.pipeline.events import (
    AssistantTextEvent,
    PipelineEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TokenUsageEvent,
)
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, PIPELINE_END

logger = logging.getLogger(__name__)
MAX_AUDIO_BATCH_BYTES = 6400
QItem = TypeVar("QItem")


async def _send_event(ws: WebSocket, event: ServerEvent) -> None:
    try:
        await ws.send_json(event.model_dump())
    except Exception as e:
        logger.error(f"Failed to send event to client: {e}")


async def _send_events(ws: WebSocket, events: list[ServerEvent]) -> None:
    for event in events:
        await _send_event(ws, event)


def _keep_audio_sentinel(item: Any) -> bool:
    return isinstance(item, bytes) and item == AUDIO_RESPONSE_DONE


def _keep_user_text_event(item: Any) -> bool:
    return isinstance(item, (SpeechStoppedEvent, TokenUsageEvent))


def _flush_queue(q: Queue[QItem], *, preserve: Callable[[QItem], bool] | None = None) -> None:
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
        except Empty:
            break
    if preserved:
        with q.mutex:
            for item in reversed(preserved):
                q.queue.appendleft(item)
            q.not_empty.notify(len(preserved))


def _clean_unit(unit: PipelineUnit, preserve: Callable[[Any], bool] | None = None) -> None:
    """Cancel in-flight work and flush queues for a single pipeline unit.

    The input queue is also drained so pending audio from a released session
    cannot be processed by the handlers (and thus reach the next session that
    claims this unit). SESSION_END is enqueued by the route handler *after*
    this returns to serve as the soft reset signal for stateful handlers.
    """
    unit.cancel_scope.cancel()
    _flush_queue(unit.input_queue)
    _flush_queue(unit.output_queue, preserve=preserve)
    _flush_queue(unit.text_output_queue, preserve=preserve)
    unit.response_playing.clear()
    unit.cancel_scope.reset()
    unit.should_listen.set()


def _to_audio_bytes(chunk: Any) -> bytes:
    if isinstance(chunk, PipelineControlMessage):
        raise TypeError(f"unexpected control message on audio output queue: {chunk!r}")
    if isinstance(chunk, np.ndarray) or hasattr(chunk, "tobytes"):
        return chunk.tobytes()
    return chunk


def create_app(pool: list[PipelineUnit], stop_event: ThreadingEvent) -> FastAPI:
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
            if sess is not None:
                try:
                    await sess.websocket.close()
                except Exception:
                    pass

    app = FastAPI(lifespan=lifespan)

    def _claim_unit(ws: WebSocket) -> PipelineUnit | None:
        """Atomically (between asyncio yield points) reserve the first idle unit.

        Creates a placeholder SessionState that the caller fills in with the
        session_id after RealtimeService.register().
        """
        for unit in pool:
            if unit.session is None:
                unit.session = SessionState(websocket=ws)
                return unit
        return None

    @app.websocket("/v1/realtime")
    async def realtime_endpoint(ws: WebSocket) -> None:
        await ws.accept()

        unit = _claim_unit(ws)
        if unit is None:
            logger.warning(f"Rejected connection: all {len(pool)} pipeline slots in use")
            # Stateless error event — rejection is not chargeable to any unit's usage metrics.
            await _send_event(
                ws,
                build_error_event(
                    f"All {len(pool)} session slots are in use. Disconnect an existing client first.",
                    error_type="session_limit_reached",
                ),
            )
            await ws.close(code=1008, reason="All session slots are in use")
            return

        pipeline_log_ctx.set(unit.index)
        session_id = unit.service.register()
        unit.session.session_id = session_id
        logger.info(f"Client connected to pipeline {unit.index} (session {session_id})")

        # Defensive: drain edge queues and reset events so stale data from a
        # previous session that survived SESSION_END propagation doesn't leak.
        _clean_unit(unit)

        try:
            await _send_event(ws, unit.service.build_session_created(session_id))

            while not stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.receive_json(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                event = unit.service.parse_client_event(raw)
                if event is None:
                    await _send_event(
                        ws,
                        unit.service.make_error(
                            f"Unknown or invalid event: {raw.get('type')}", "unknown_or_invalid_event"
                        ),
                    )
                    continue

                if isinstance(event, InputAudioBufferAppendEvent):
                    chunks = unit.service.handle_audio_append(session_id, event)
                    rt_cfg = unit.service._state(session_id).runtime_config
                    for chunk in chunks:
                        unit.input_queue.put((chunk, rt_cfg))

                elif isinstance(event, InputAudioBufferCommitEvent):
                    err = unit.service.handle_audio_commit(session_id)
                    if err:
                        await _send_event(ws, err)

                elif isinstance(event, SessionUpdateEvent):
                    err = unit.service.handle_session_update(session_id, event)
                    if err:
                        await _send_event(ws, err)

                elif isinstance(event, ConversationItemCreateEvent):
                    events = unit.service.handle_conversation_item_create(session_id, event)
                    if events:
                        await _send_events(ws, events)

                elif isinstance(event, ResponseCreateEvent):
                    result = unit.service.handle_response_create(session_id, event)
                    if result:
                        if result.type != "error":
                            unit.cancel_scope.new_response()
                        await _send_event(ws, result)

                elif isinstance(event, ResponseCancelEvent):
                    was_active = unit.service._state(session_id).in_response
                    if was_active:
                        unit.cancel_scope.cancel()
                    _flush_queue(unit.output_queue, preserve=_keep_audio_sentinel)
                    _flush_queue(unit.text_output_queue, preserve=_keep_user_text_event)
                    events = unit.service.handle_response_cancel(session_id)
                    if events:
                        await _send_events(ws, events)
                    unit.response_playing.clear()

        except WebSocketDisconnect:
            logger.info(f"Client {session_id} disconnected from pipeline {unit.index}")
        except Exception as e:
            logger.error(
                f"Client {session_id} on pipeline {unit.index} error: {type(e).__name__}: {e}", exc_info=True
            )
        finally:
            _clean_unit(unit)
            unit.service.unregister(session_id)
            unit.input_queue.put(SESSION_END)
            # Drop the SessionState last — once None, the send loop treats this
            # unit as idle and _claim_unit can hand it to the next client.
            unit.session = None
            logger.info(f"Pipeline {unit.index} released (session {session_id} ended)")

    @app.get("/v1/usage")
    async def usage_endpoint() -> dict[str, Any]:
        # Aggregate usage across the pool.
        total: dict[str, Any] = {}
        for unit in pool:
            usage = unit.service.get_usage()
            for k, v in usage.items():
                if isinstance(v, (int, float)):
                    total[k] = total.get(k, 0) + v
                else:
                    total.setdefault(k, v)
        return total

    @app.get("/v1/pool")
    async def pool_endpoint() -> dict[str, Any]:
        return {
            "size": len(pool),
            "in_use": sum(1 for u in pool if u.session is not None),
            "units": [
                {
                    "index": u.index,
                    "in_use": u.session is not None,
                    "session_id": u.session.session_id if u.session else None,
                }
                for u in pool
            ],
        }

    async def _send_loop_for(unit: PipelineUnit) -> None:
        """Per-pipeline send loop. Polls this unit's output queues and forwards
        to the websocket currently attached via unit.session.

        Per-session scratch (pending_output_item) lives on SessionState, so it
        disappears together with the websocket when the session is released —
        no stale sentinel can leak into the next claim.
        """
        pipeline_log_ctx.set(unit.index)
        while not stop_event.is_set():
            try:
                # Snapshot the session once per iteration; if the route releases the
                # unit mid-iteration, we continue against the prior snapshot which is
                # consistent (its websocket is still valid until ws.close() returns).
                session = unit.session
                ws = session.websocket if session is not None else None
                session_id = session.session_id if session is not None else None

                # Text events first (speech_started cancels active response).
                try:
                    text_msg = unit.text_output_queue.get_nowait()
                    is_speech_start = isinstance(text_msg, SpeechStartedEvent)

                    was_in_response = False
                    if is_speech_start and session_id:
                        was_in_response = unit.service._state(session_id).in_response

                    if unit.cancel_scope.discarding and isinstance(text_msg, AssistantTextEvent):
                        pass
                    elif ws is not None and isinstance(text_msg, PipelineEvent) and session_id:
                        events = unit.service.dispatch_pipeline_event(session_id, text_msg)
                        if events:
                            await _send_events(ws, events)

                    if is_speech_start and was_in_response:
                        active_cfg = (
                            unit.service._state(session_id).runtime_config if session_id else None
                        )
                        if active_cfg is None or active_cfg.interrupt_response_enabled:
                            unit.cancel_scope.cancel()
                            _flush_queue(unit.output_queue, preserve=_keep_audio_sentinel)
                            _flush_queue(unit.text_output_queue, preserve=_keep_user_text_event)
                            if unit.response_playing.is_set():
                                unit.response_playing.clear()
                            logger.info(
                                f"Pipeline {unit.index}: speech during response: cancelled, queue flushed"
                            )
                        else:
                            logger.info(
                                f"Pipeline {unit.index}: speech during response: "
                                f"interrupt_response disabled, ignoring"
                            )
                except Empty:
                    pass

                try:
                    if session is not None and session.pending_output_item is not None:
                        audio_chunk = session.pending_output_item
                        session.pending_output_item = None
                    else:
                        audio_chunk = unit.output_queue.get_nowait()

                    if isinstance(audio_chunk, bytes) and audio_chunk == PIPELINE_END:
                        if ws is not None and session_id:
                            await _send_events(ws, unit.service.finish_audio_response(session_id))
                        break

                    if isinstance(audio_chunk, bytes) and audio_chunk == AUDIO_RESPONSE_DONE:
                        if ws is not None and session_id:
                            await _send_events(ws, unit.service.finish_audio_response(session_id))
                        unit.response_playing.clear()
                        unit.cancel_scope.response_done()
                        unit.should_listen.set()
                        logger.info(f"Pipeline {unit.index}: response complete, listening re-enabled")
                        continue

                    if is_control_message(audio_chunk):
                        continue

                    if unit.cancel_scope.discarding:
                        continue

                    audio_chunk = _to_audio_bytes(audio_chunk)

                    audio_batch = bytearray(audio_chunk)
                    while len(audio_batch) < MAX_AUDIO_BATCH_BYTES:
                        try:
                            next_chunk = unit.output_queue.get_nowait()
                        except Empty:
                            break

                        if (
                            isinstance(next_chunk, bytes) and next_chunk in {PIPELINE_END, AUDIO_RESPONSE_DONE}
                        ) or is_control_message(next_chunk, SESSION_END.kind):
                            # Only stash if we still have a session; otherwise drop it.
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

                    if ws is not None and session_id:
                        await _send_events(
                            ws, unit.service.encode_audio_chunk(session_id, bytes(audio_batch))
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
