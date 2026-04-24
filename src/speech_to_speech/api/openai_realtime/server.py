import logging
import threading
from queue import Queue
from threading import Event
from typing import cast

import uvicorn

from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.websocket_router import create_app
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.queue_types import AudioInItem, AudioOutItem, TextEventItem, TextPromptItem

logger = logging.getLogger(__name__)


class RealtimeServer:
    """
    Pipeline handler for the OpenAI Realtime API mode.

    Owns pipeline queues, exposes run() for ThreadManager, and bridges
    between FastAPI/uvicorn and the internal audio + text queues.
    """

    def __init__(
        self,
        stop_event: Event,
        input_queue: Queue[AudioInItem],
        output_queue: Queue[AudioOutItem],
        should_listen: Event,
        response_playing: Event | None = None,
        cancel_scope: CancelScope | None = None,
        text_output_queue: Queue[TextEventItem] | None = None,
        text_prompt_queue: Queue[TextPromptItem] | None = None,
        host: str = "0.0.0.0",
        port: int = 8765,
        chat_size: int = 10,
    ) -> None:
        self.stop_event = stop_event
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.text_output_queue = text_output_queue
        self.text_prompt_queue = text_prompt_queue
        self.should_listen = should_listen
        self.response_playing = response_playing
        self.cancel_scope = cancel_scope
        self.host = host
        self.port = port
        self.chat_size = chat_size

    def run(self) -> None:
        """Start the FastAPI/uvicorn server (called from a ThreadManager thread)."""
        service = RealtimeService(
            text_prompt_queue=self.text_prompt_queue,
            should_listen=self.should_listen,
            chat_size=self.chat_size,
        )
        app = create_app(
            service=service,
            input_queue=self.input_queue,
            output_queue=self.output_queue,
            text_output_queue=cast(Queue[TextEventItem], self.text_output_queue),
            should_listen=self.should_listen,
            response_playing=self.response_playing,
            cancel_scope=self.cancel_scope,
            stop_event=self.stop_event,
        )

        logger.info(f"OpenAI Realtime API server starting on ws://{self.host}:{self.port}/v1/realtime")

        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
        server = uvicorn.Server(config)

        server.install_signal_handlers = lambda: None  # type: ignore[attr-defined]

        def _watch_stop() -> None:
            self.stop_event.wait()
            server.should_exit = True

        watcher = threading.Thread(target=_watch_stop, daemon=True)
        watcher.start()

        server.run()
