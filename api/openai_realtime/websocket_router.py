import asyncio
import logging
from contextlib import asynccontextmanager
from queue import Empty, Queue
from threading import Event as ThreadingEvent
from typing import Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from api.openai_realtime.service import RealtimeService, ServerEvent
from cancel_scope import CancelScope
from pipeline_control import SESSION_END, is_control_message
from pipeline_messages import AUDIO_RESPONSE_DONE, PIPELINE_END

from openai.types.realtime import (
    InputAudioBufferAppendEvent,
    InputAudioBufferCommitEvent,
    SessionUpdateEvent,
    ConversationItemCreateEvent,
    ResponseCreateEvent,
    ResponseCancelEvent,
)

logger = logging.getLogger(__name__)
MAX_AUDIO_BATCH_BYTES = 6400


async def _send_event(ws: WebSocket, event: ServerEvent) -> None:
    try:
        await ws.send_json(event.model_dump())
    except Exception as e:
        logger.error(f"Failed to send event to client: {e}")


async def _send_events(ws: WebSocket, events: list[ServerEvent]) -> None:
    for event in events:
        await _send_event(ws, event)


def _keep_audio_sentinel(item) -> bool:
    return isinstance(item, bytes) and item == AUDIO_RESPONSE_DONE


def _keep_user_text_event(item) -> bool:
    if not isinstance(item, dict):
        return False
    return item.get("type") in ("speech_stopped", "token_usage")


def create_app(
    service: RealtimeService,
    input_queue: Queue,
    output_queue: Queue,
    text_output_queue: Queue,
    should_listen: ThreadingEvent,
    response_playing: ThreadingEvent | None,
    cancel_scope: CancelScope | None,
    stop_event: ThreadingEvent,
) -> FastAPI:

    def _flush_queue(q: Queue, *, preserve: Callable | None = None) -> None:
        """Drain a queue, optionally preserving items matching *preserve*.

        Preserved items are re-inserted at the **front** of the queue
        (atomically under the queue's mutex) so they are processed before
        anything a pipeline thread may have enqueued during the drain.
        """
        preserved: list = []
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
        
    def clean_session(preserve: Callable | None = None):
        # Invalidate in-flight LLM/TTS work (cooperative cancel via is_stale), then
        # flush queues. reset() clears discarding only; generation stays bumped.
        # Blocking HTTP reads are not interrupted here; see OpenApiModelHandler.process.
        if cancel_scope:
            cancel_scope.cancel()
        _flush_queue(output_queue, preserve=preserve)
        _flush_queue(text_output_queue, preserve=preserve)
        if response_playing:
            response_playing.clear()
        if cancel_scope:
            cancel_scope.reset()
        should_listen.set()

    def _to_audio_bytes(audio_chunk) -> bytes:
        if hasattr(audio_chunk, "tobytes"):
            return audio_chunk.tobytes()
        return audio_chunk

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.websockets = {}
        app.state.active_session: str | None = None
        app.state.send_task = asyncio.create_task(_send_loop())
        yield
        app.state.send_task.cancel()
        try:
            await app.state.send_task
        except asyncio.CancelledError:
            pass
        for ws in list(app.state.websockets.values()):
            try:
                await ws.close()
            except Exception:
                pass

    app = FastAPI(lifespan=lifespan)

    @app.websocket("/v1/realtime")
    async def realtime_endpoint(ws: WebSocket):
        await ws.accept()

        if app.state.websockets:
            logger.warning("Rejected connection: a session is already active")
            await _send_event(
                ws,
                service.make_error(
                    "Only one concurrent session is supported. "
                    "Disconnect the existing client first.",
                    _type="session_limit_reached",
                ),
            )
            await ws.close(code=1008, reason="Only one concurrent session is supported")
            return

        session_id = service.register()
        app.state.websockets[session_id] = ws
        app.state.active_session = session_id
        logger.info(f"Client connected (session {session_id})")

        # Defensive: drain edge queues and reset events so stale data from a
        # previous session that survived SESSION_END propagation doesn't leak.
        clean_session()

        try:
            await _send_event(ws, service.build_session_created(session_id))

            while not stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.receive_json(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                event = service.parse_client_event(raw)
                if event is None:
                    await _send_event(
                        ws, service.make_error(f"Unknown or invalid event: {raw.get('type')}", "unknown_or_invalid_event")
                    )
                    continue

                if isinstance(event, InputAudioBufferAppendEvent):
                    chunks = service.handle_audio_append(session_id, event)
                    for chunk in chunks:
                        input_queue.put(chunk)

                elif isinstance(event, InputAudioBufferCommitEvent):
                    err = service.handle_audio_commit(session_id)
                    if err:
                        await _send_event(ws, err)

                elif isinstance(event, SessionUpdateEvent):
                    err = service.handle_session_update(session_id, event)
                    if err:
                        await _send_event(ws, err)

                elif isinstance(event, ConversationItemCreateEvent):
                    events = service.handle_conversation_item_create(session_id, event)
                    if events:
                        await _send_events(ws, events)

                elif isinstance(event, ResponseCreateEvent):
                    result = service.handle_response_create(session_id, event)
                    if result:
                        if result.type != "error" and cancel_scope:
                            cancel_scope.new_response()
                        await _send_event(ws, result)

                elif isinstance(event, ResponseCancelEvent):
                    was_active = service._state(session_id).in_response
                    if was_active and cancel_scope:
                        cancel_scope.cancel()
                    _flush_queue(output_queue, preserve=_keep_audio_sentinel)
                    _flush_queue(text_output_queue, preserve=_keep_user_text_event)
                    events = service.handle_response_cancel(session_id)
                    if events:
                        await _send_events(ws, events)
                    if response_playing:
                        response_playing.clear()

        except WebSocketDisconnect:
            logger.info(f"Client {session_id} disconnected")
        except Exception as e:
            logger.error(f"Client {session_id} error: {type(e).__name__}: {e}", exc_info=True)
        finally:
            clean_session()
            service.unregister(session_id)
            if not service._conns:
                service.runtime_config.reset()
                input_queue.put(SESSION_END)
                logger.info("Last client disconnected, reset RuntimeConfig and sent SESSION_END")
            app.state.websockets.pop(session_id, None)
            app.state.active_session = None
            logger.info(f"Client {session_id} removed")

    @app.get("/v1/usage")
    async def usage_endpoint():
        return service.get_usage()

    async def _send_loop():
        """Poll pipeline output queues and send to each connected client."""
        pending_output_item = None
        while not stop_event.is_set():
            try:
                # Process text events first (speech_started cancels active response)
                if text_output_queue:
                    try:
                        text_msg = text_output_queue.get_nowait()
                        is_speech_start = text_msg.get("type") == "speech_started"

                        # Capture response state before translate modifies it. To change when multiple sessions are supported.
                        was_in_response = False
                        if is_speech_start and app.state.active_session:
                            was_in_response = service._state(app.state.active_session).in_response

                        if cancel_scope and cancel_scope.discarding and text_msg.get("type") == "assistant_text":
                            pass
                        else:
                            for cid in service.connection_ids:
                                ws = app.state.websockets.get(cid)
                                if ws:
                                    events = service.dispatch_pipeline_event(cid, text_msg)
                                    if events:
                                        await _send_events(ws, events)

                        if is_speech_start and was_in_response:
                            if service.runtime_config.interrupt_response_enabled:
                                if cancel_scope:
                                    cancel_scope.cancel()
                                _flush_queue(output_queue, preserve=_keep_audio_sentinel)
                                _flush_queue(text_output_queue, preserve=_keep_user_text_event)
                                if response_playing and response_playing.is_set():
                                    response_playing.clear()
                                logger.info("Speech during response: cancelled, queue flushed")
                            else:
                                logger.info("Speech during response: interrupt_response disabled, ignoring")
                    except Empty:
                        pass

                try:
                    if pending_output_item is not None:
                        audio_chunk = pending_output_item
                        pending_output_item = None
                    else:
                        audio_chunk = output_queue.get_nowait()

                    if isinstance(audio_chunk, bytes) and audio_chunk == PIPELINE_END:
                        for cid in service.connection_ids:
                            ws = app.state.websockets.get(cid)
                            if ws:
                                await _send_events(ws, service.finish_audio_response(cid))
                        break

                    if isinstance(audio_chunk, bytes) and audio_chunk == AUDIO_RESPONSE_DONE:
                        for cid in service.connection_ids:
                            ws = app.state.websockets.get(cid)
                            if ws:
                                await _send_events(ws, service.finish_audio_response(cid))
                        if response_playing:
                            response_playing.clear()
                        if cancel_scope:
                            cancel_scope.response_done()
                        should_listen.set()
                        logger.info("Response complete, listening re-enabled")
                        continue

                    if is_control_message(audio_chunk, SESSION_END.kind):
                        continue

                    if cancel_scope and cancel_scope.discarding:
                        continue

                    audio_chunk = _to_audio_bytes(audio_chunk)

                    audio_batch = bytearray(audio_chunk)
                    while len(audio_batch) < MAX_AUDIO_BATCH_BYTES:
                        try:
                            next_chunk = output_queue.get_nowait()
                        except Empty:
                            break

                        if (
                            isinstance(next_chunk, bytes)
                            and next_chunk in {PIPELINE_END, AUDIO_RESPONSE_DONE}
                        ) or is_control_message(next_chunk, SESSION_END.kind):
                            pending_output_item = next_chunk
                            break

                        next_audio = _to_audio_bytes(next_chunk)
                        if len(audio_batch) + len(next_audio) > MAX_AUDIO_BATCH_BYTES:
                            pending_output_item = next_chunk
                            break
                        audio_batch.extend(next_audio)

                    if response_playing and not response_playing.is_set():
                        response_playing.set()
                        should_listen.set()

                    for cid in service.connection_ids:
                        ws = app.state.websockets.get(cid)
                        if ws:
                            await _send_events(ws, service.encode_audio_chunk(cid, bytes(audio_batch)))
                except Empty:
                    pass

                await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Send loop error: {e}")

    return app
