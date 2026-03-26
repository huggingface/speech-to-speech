import logging
import threading
from queue import Queue
from threading import Event

import uvicorn

from api.openai_realtime.runtime_config import RuntimeConfig
from api.openai_realtime.service import RealtimeService
from api.openai_realtime.websocket_router import create_app

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
        input_queue: Queue,
        output_queue: Queue,
        should_listen: Event,
        response_playing: Event | None = None,
        cancel_response: Event | None = None,
        text_output_queue: Queue | None = None,
        text_prompt_queue: Queue | None = None,
        runtime_config: RuntimeConfig | None = None,
        host: str = "0.0.0.0",
        port: int = 8765,
    ):
        self.stop_event = stop_event
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.text_output_queue = text_output_queue
        self.text_prompt_queue = text_prompt_queue
        self.should_listen = should_listen
        self.response_playing = response_playing
        self.cancel_response = cancel_response
        self.runtime_config = runtime_config or RuntimeConfig()
        self.host = host
        self.port = port

    def run(self):
        """Start the FastAPI/uvicorn server (called from a ThreadManager thread)."""
        service = RealtimeService(
            runtime_config=self.runtime_config,
            text_prompt_queue=self.text_prompt_queue,
            should_listen=self.should_listen,
        )
        app = create_app(
            service=service,
            input_queue=self.input_queue,
            output_queue=self.output_queue,
            text_output_queue=self.text_output_queue,
            should_listen=self.should_listen,
            response_playing=self.response_playing,
            cancel_response=self.cancel_response,
            stop_event=self.stop_event,
        )

        logger.info(
            f"OpenAI Realtime API server starting on ws://{self.host}:{self.port}/v1/realtime"
        )

        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
        server = uvicorn.Server(config)

        server.install_signal_handlers = lambda: None

        def _watch_stop():
            self.stop_event.wait()
            server.should_exit = True

        watcher = threading.Thread(target=_watch_stop, daemon=True)
        watcher.start()

        server.run()
