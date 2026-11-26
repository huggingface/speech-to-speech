import logging
import threading
from threading import Event

import uvicorn

from speech_to_speech.api.openai_realtime.llm_proxy import LLMProxyConfig
from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit
from speech_to_speech.api.openai_realtime.websocket_router import create_app

logger = logging.getLogger(__name__)


class RealtimeServer:
    """
    Pipeline handler for the OpenAI Realtime API mode.

    Owns a pool of isolated PipelineUnits and a single uvicorn server.
    The websocket route claims the next free unit on each accept and releases
    it on disconnect; once all units are in use, further connections are rejected.
    """

    def __init__(
        self,
        stop_event: Event,
        pool: list[PipelineUnit],
        host: str = "0.0.0.0",
        port: int = 8765,
        llm_proxy_config: LLMProxyConfig | None = None,
        context_integration=None,
    ) -> None:
        if not pool:
            raise ValueError("RealtimeServer requires at least one PipelineUnit in the pool")
        self.stop_event = stop_event
        self.startup_error = None
        self.pool = pool
        self.context_integration = context_integration
        self.host = host
        self.port = port
        self.llm_proxy_config = llm_proxy_config

    def run(self) -> None:
        """Start the FastAPI/uvicorn server (called from a ThreadManager thread)."""
        options = {"context_integration": self.context_integration} if self.context_integration is not None else {}
        app = create_app(pool=self.pool, stop_event=self.stop_event, llm_proxy_config=self.llm_proxy_config, **options)

        logger.info(
            f"OpenAI Realtime API starting on ws://{self.host}:{self.port}/v1/realtime (pool size {len(self.pool)})"
        )

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

        try:
            server.run()
            if self.context_integration is not None and not server.started and not self.stop_event.is_set():
                self.startup_error = RuntimeError("Context-enabled Realtime server failed to start")
                raise self.startup_error
        finally:
            # A bind/startup failure must also stop the model threads. During a
            # normal shutdown the event is already set by ThreadManager.
            self.stop_event.set()
