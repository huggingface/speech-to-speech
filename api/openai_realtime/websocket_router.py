import asyncio
import logging
from queue import Empty, Queue
from threading import Event as ThreadingEvent

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from api.openai_realtime.protocol import (
    InputAudioBufferAppend,
    InputAudioBufferCommit,
    SessionCreated,
    SessionUpdate,
    ServerEvent,
)
from api.openai_realtime.service import RealtimeService

logger = logging.getLogger(__name__)


async def _send_event(ws: WebSocket, event: ServerEvent) -> None:
    try:
        await ws.send_json(event.model_dump())
    except Exception as e:
        logger.error(f"Failed to send event to client: {e}")


async def _broadcast(clients: set[WebSocket], events: list[ServerEvent]) -> None:
    """Send a list of events to every connected client, in order."""
    for event in events:
        await asyncio.gather(
            *[_send_event(ws, event) for ws in clients],
            return_exceptions=True,
        )


def create_app(
    service: RealtimeService,
    input_queue: Queue,
    output_queue: Queue,
    text_output_queue: Queue,
    should_listen: ThreadingEvent,
    stop_event: ThreadingEvent,
) -> FastAPI:
    app = FastAPI()
    app.state.clients: set[WebSocket] = set()

    @app.websocket("/v1/realtime")
    async def realtime_endpoint(ws: WebSocket):
        await ws.accept()
        client_id = id(ws)
        app.state.clients.add(ws)
        logger.info(f"Client {client_id} connected")

        if len(app.state.clients) == 1:
            should_listen.set()
            logger.debug("Listening enabled (should_listen.set())")

        try:
            await _send_event(ws, SessionCreated())

            while not stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.receive_json(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                event = service.parse_client_event(raw)
                if event is None:
                    await _send_event(
                        ws, service.make_error(f"Unknown or invalid event: {raw.get('type')}")
                    )
                    continue

                if isinstance(event, InputAudioBufferAppend):
                    if should_listen.is_set():
                        chunks = service.handle_audio_append(event)
                        for chunk in chunks:
                            input_queue.put(chunk)
                        logger.debug(f"Client {client_id}: queued {len(chunks)} audio chunks")
                    else:
                        logger.debug(f"Client {client_id}: skipping audio (not listening)")

                elif isinstance(event, InputAudioBufferCommit):
                    logger.debug(f"Client {client_id}: audio buffer commit (VAD handles segmentation)")

                elif isinstance(event, SessionUpdate):
                    service.handle_session_update(event)

        except WebSocketDisconnect:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Client {client_id} error: {type(e).__name__}: {e}", exc_info=True)
        finally:
            app.state.clients.discard(ws)
            logger.info(f"Client {client_id} removed ({len(app.state.clients)} remaining)")
            if len(app.state.clients) == 0:
                input_queue.put(b"END")

    async def _send_loop():
        """Poll pipeline output queues and broadcast to all connected clients."""
        while not stop_event.is_set():
            try:
                # Audio output
                try:
                    audio_chunk = output_queue.get_nowait()
                    if isinstance(audio_chunk, bytes) and audio_chunk == b"END":
                        if app.state.clients:
                            await _broadcast(app.state.clients, service.finish_audio_response())
                        break

                    if app.state.clients:
                        if hasattr(audio_chunk, "tobytes"):
                            audio_chunk = audio_chunk.tobytes()
                        events = service.encode_audio_chunk(audio_chunk)
                        await _broadcast(app.state.clients, events)
                except Empty:
                    pass

                # Text / tool output
                if text_output_queue:
                    try:
                        text_msg = text_output_queue.get_nowait()
                        if app.state.clients:
                            events = service.translate_pipeline_text(text_msg)
                            if events:
                                await _broadcast(app.state.clients, events)
                    except Empty:
                        pass

                await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Send loop error: {e}")

    @app.on_event("startup")
    async def on_startup():
        app.state.send_task = asyncio.create_task(_send_loop())

    @app.on_event("shutdown")
    async def on_shutdown():
        task = getattr(app.state, "send_task", None)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        for ws in list(app.state.clients):
            try:
                await ws.close()
            except Exception:
                pass

    return app
