import logging

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
        stop_event,
        input_queue,
        output_queue,
        should_listen,
        text_output_queue=None,
        text_prompt_queue=None,
        runtime_config=None,
        host="0.0.0.0",
        port=8765,
    ):
        self.stop_event = stop_event
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.text_output_queue = text_output_queue
        self.text_prompt_queue = text_prompt_queue
        self.should_listen = should_listen
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

        server.run()
