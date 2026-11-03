"""Prepare GPU models after native admission and gate early microphone input."""

import asyncio
import json
import logging


class ModelReadinessMiddleware:
    """Preserve the native session, adding readiness after session.created.

    ``prefetch`` is a blocking preparation callback: success must mean all model
    inference warmups finished. Busy sessions never request an allocation.
    """

    def __init__(self, app, prefetch):
        self.app = app
        self.prefetch = prefetch

    async def __call__(self, scope, receive, send):
        if scope["type"] != "websocket" or scope.get("path") != "/v1/realtime":
            return await self.app(scope, receive, send)
        requested = ready = closed = False
        preparation = None
        send_lock = asyncio.Lock()

        def event_type(message):
            try:
                event = json.loads(message.get("text") or message.get("bytes") or "")
                return event.get("type") if isinstance(event, dict) else None
            except (ValueError, TypeError, UnicodeDecodeError):
                return None

        async def emit(event):
            await guarded_send({"type": "websocket.send", "text": json.dumps(event)})

        async def guarded_send(message):
            nonlocal closed
            async with send_lock:
                if closed:
                    return
                await send(message)
                if message["type"] == "websocket.close":
                    closed = True

        async def prepare():
            nonlocal ready
            try:
                await asyncio.to_thread(self.prefetch)
                if not closed:
                    # Publish readiness before the next receive can admit audio.
                    await emit({"type": "speech_to_speech.models.ready"})
                    ready = True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logging.getLogger(__name__).exception("Session model preparation failed")
                await emit({"type": "speech_to_speech.models.error", "error": {"message": str(exc)}})
                await guarded_send({"type": "websocket.close", "code": 1011})

        async def observe(message):
            nonlocal requested, preparation
            await guarded_send(message)
            if (
                not closed
                and not requested
                and message["type"] == "websocket.send"
                and event_type(message) == "session.created"
            ):
                requested = True
                await emit({"type": "speech_to_speech.models.preparing"})
                preparation = asyncio.create_task(prepare())

        async def gated_receive():
            nonlocal closed
            while True:
                message = await receive()
                if message["type"] == "websocket.disconnect":
                    closed = True
                    return message
                if (
                    not ready
                    and message["type"] == "websocket.receive"
                    and event_type(message)
                    in {
                        "input_audio_buffer.append",
                        "response.create",
                    }
                ):
                    continue  # Never replay speech captured during preparation.
                return message

        try:
            await self.app(scope, gated_receive, observe)
        finally:
            closed = True
            if preparation is not None:
                preparation.cancel()
                await asyncio.gather(preparation, return_exceptions=True)
