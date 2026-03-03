import asyncio
import logging
from contextlib import asynccontextmanager
from queue import Empty, Queue
from threading import Event as ThreadingEvent

from fastapi import FastAPI, WebSocket, WebSocketDisconnect


from api.openai_realtime.service import RealtimeService, ServerEvent

from openai.types.realtime import (
    InputAudioBufferAppendEvent,
    InputAudioBufferCommitEvent,
    SessionUpdateEvent,
    ConversationItemCreateEvent,
    ResponseCreateEvent,
    ResponseCancelEvent,
    SessionCreatedEvent,
    RealtimeSessionCreateRequest
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
    stop_event: ThreadingEvent,
) -> FastAPI:

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
                    code="session_limit_reached",
                ),
            )
            await ws.close(code=1008, reason="Only one concurrent session is supported")
            return

        conn_id = str(id(ws))
        service.register(conn_id)
        app.state.websockets[conn_id] = ws
        logger.info(f"Client {conn_id} connected")

        should_listen.set()

        try:
            await _send_event(ws, SessionCreatedEvent(event_id=conn_id, session=RealtimeSessionCreateRequest(type="realtime")))

            while not stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.receive_json(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                event = service.parse_client_event(raw)
                if event is None:
                    await _send_event(
                        ws, service.make_error(f"Unknown or invalid event: {raw.get('type')}", "unknown_or_invalid_event", conn_id)
                    )
                    continue

                if isinstance(event, InputAudioBufferAppendEvent):
                    if should_listen.is_set():
                        chunks = service.handle_audio_append(conn_id, event)
                        for chunk in chunks:
                            input_queue.put(chunk)
                    else:
                        logger.debug(f"Client {conn_id}: skipping audio (not listening)")

                elif isinstance(event, InputAudioBufferCommitEvent):
                    err = service.handle_audio_commit(conn_id)
                    if err:
                        await _send_event(ws, err)

                elif isinstance(event, SessionUpdateEvent):
                    service.handle_session_update(event)

                elif isinstance(event, ConversationItemCreateEvent):
                    events = service.handle_conversation_item_create(conn_id, event)
                    if events:
                        await _send_events(ws, events)

                elif isinstance(event, ResponseCreateEvent):
                    err = service.handle_response_create(conn_id, event)
                    if err:
                        await _send_event(ws, err)

                elif isinstance(event, ResponseCancelEvent):
                    events = service.handle_response_cancel(conn_id)
                    if events:
                        await _send_events(ws, events)

        except WebSocketDisconnect:
            logger.info(f"Client {conn_id} disconnected")
        except Exception as e:
            logger.error(f"Client {conn_id} error: {type(e).__name__}: {e}", exc_info=True)
        finally:
            service.unregister(conn_id)
            app.state.websockets.pop(conn_id, None)
            logger.info(f"Client {conn_id} removed")
            input_queue.put(b"END")

    async def _send_loop():
        """Poll pipeline output queues and send to each connected client."""
        while not stop_event.is_set():
            try:
                try:
                    audio_chunk = output_queue.get_nowait()
                    if isinstance(audio_chunk, bytes) and audio_chunk == b"END":
                        for cid in service.connection_ids:
                            ws = app.state.websockets.get(cid)
                            if ws:
                                await _send_events(ws, service.finish_audio_response(cid))
                        break

                    if hasattr(audio_chunk, "tobytes"):
                        audio_chunk = audio_chunk.tobytes()
                    for cid in service.connection_ids:
                        ws = app.state.websockets.get(cid)
                        if ws:
                            await _send_events(ws, service.encode_audio_chunk(cid, audio_chunk))
                except Empty:
                    pass

                if text_output_queue:
                    try:
                        text_msg = text_output_queue.get_nowait()
                        for cid in service.connection_ids:
                            ws = app.state.websockets.get(cid)
                            if ws:
                                events = service.translate_pipeline_text(cid, text_msg)
                                if events:
                                    await _send_events(ws, events)
                    except Empty:
                        pass

                await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Send loop error: {e}")

    return app
