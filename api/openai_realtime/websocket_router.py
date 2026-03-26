import asyncio
import logging
from contextlib import asynccontextmanager
from queue import Empty, Queue
from threading import Event as ThreadingEvent

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from api.openai_realtime.service import RealtimeService, ServerEvent
from cancel_scope import CancelScope
from pipeline_control import SESSION_END

from openai.types.realtime import (
    InputAudioBufferAppendEvent,
    InputAudioBufferCommitEvent,
    SessionUpdateEvent,
    ConversationItemCreateEvent,
    ResponseCreateEvent,
    ResponseCancelEvent,
)

logger = logging.getLogger(__name__)


async def _send_event(ws: WebSocket, event: ServerEvent) -> None:
    try:
        await ws.send_json(event.model_dump())
    except Exception as e:
        logger.error(f"Failed to send event to client: {e}")


async def _send_events(ws: WebSocket, events: list[ServerEvent]) -> None:
    for event in events:
        await _send_event(ws, event)


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

    def _flush_queue(q: Queue, *, preserve_sentinel: bool = False) -> None:
        """Drain a queue.  When *preserve_sentinel* is True, any
        ``__RESPONSE_DONE__`` found during the drain is re-enqueued so
        ``_send_loop`` can still emit ``finish_audio_response`` events
        and call ``cancel_scope.response_done()``."""
        found_done = False
        while not q.empty():
            try:
                item = q.get_nowait()
                if preserve_sentinel and isinstance(item, bytes) and item == b"__RESPONSE_DONE__":
                    found_done = True
            except Empty:
                break
        if found_done:
            q.put(b"__RESPONSE_DONE__")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.websockets = {}
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
        logger.info(f"Client connected (session {session_id})")

        # Defensive: drain edge queues and reset events so stale data from a
        # previous session that survived SESSION_END propagation doesn't leak.
        for q in (output_queue, text_output_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except Empty:
                    break
        if response_playing:
            response_playing.clear()
        if cancel_scope:
            cancel_scope.reset()

        should_listen.set()

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
                    err = service.handle_session_update(event)
                    if err:
                        await _send_event(ws, err)

                elif isinstance(event, ConversationItemCreateEvent):
                    events = service.handle_conversation_item_create(session_id, event)
                    if events:
                        await _send_events(ws, events)

                elif isinstance(event, ResponseCreateEvent):
                    if cancel_scope:
                        cancel_scope.new_response()
                    err = service.handle_response_create(session_id, event)
                    if err:
                        await _send_event(ws, err)

                elif isinstance(event, ResponseCancelEvent):
                    was_active = service._state(session_id).in_response
                    if was_active and cancel_scope:
                        cancel_scope.cancel()
                    _flush_queue(output_queue, preserve_sentinel=True)
                    _flush_queue(text_output_queue)
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
            service.unregister(session_id)
            if not service._conns:
                service.runtime_config.reset()
                input_queue.put(SESSION_END)
                logger.info("Last client disconnected, reset RuntimeConfig and sent SESSION_END")
            app.state.websockets.pop(session_id, None)
            logger.info(f"Client {session_id} removed")

    @app.get("/v1/usage")
    async def usage_endpoint():
        return service.get_usage()

    async def _send_loop():
        """Poll pipeline output queues and send to each connected client."""
        while not stop_event.is_set():
            try:
                # Process text events first (speech_started cancels active response)
                if text_output_queue:
                    try:
                        text_msg = text_output_queue.get_nowait()
                        is_speech_start = text_msg.get("type") == "speech_started"

                        if cancel_scope and cancel_scope.discarding and text_msg.get("type") == "assistant_text":
                            pass
                        else:
                            for cid in service.connection_ids:
                                ws = app.state.websockets.get(cid)
                                if ws:
                                    events = service.translate_pipeline_text(cid, text_msg)
                                    if events:
                                        await _send_events(ws, events)

                        if is_speech_start and response_playing and response_playing.is_set():
                            if service.runtime_config.interrupt_response_enabled:
                                if cancel_scope:
                                    cancel_scope.cancel()
                                _flush_queue(output_queue, preserve_sentinel=True)
                                _flush_queue(text_output_queue)
                                response_playing.clear()
                                logger.info("Speech during response: cancelled, queue flushed")
                            else:
                                logger.info("Speech during response: interrupt_response disabled, ignoring")
                    except Empty:
                        pass

                try:
                    audio_chunk = output_queue.get_nowait()

                    if isinstance(audio_chunk, bytes) and audio_chunk == b"END":
                        for cid in service.connection_ids:
                            ws = app.state.websockets.get(cid)
                            if ws:
                                await _send_events(ws, service.finish_audio_response(cid))
                        break

                    if isinstance(audio_chunk, bytes) and audio_chunk == b"__RESPONSE_DONE__":
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

                    if cancel_scope and cancel_scope.discarding:
                        continue

                    if hasattr(audio_chunk, "tobytes"):
                        audio_chunk = audio_chunk.tobytes()

                    if response_playing and not response_playing.is_set():
                        response_playing.set()
                        should_listen.set()

                    for cid in service.connection_ids:
                        ws = app.state.websockets.get(cid)
                        if ws:
                            await _send_events(ws, service.encode_audio_chunk(cid, audio_chunk))
                except Empty:
                    pass

                await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Send loop error: {e}")

    return app
