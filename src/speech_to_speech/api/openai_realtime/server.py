import logging
import threading
from socket import socket
from threading import Event

import uvicorn

from speech_to_speech.api.openai_realtime.llm_proxy import LLMProxyConfig
from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit
from speech_to_speech.api.openai_realtime.websocket_router import create_app
from speech_to_speech.config import _ID

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
        *,
        pipeline_pools: dict[str, list[PipelineUnit]] | None = None,
        llm_proxy_configs: dict[str, LLMProxyConfig | None] | None = None,
    ) -> None:
        if not pool:
            raise ValueError("RealtimeServer requires at least one PipelineUnit in the pool")
        self.stop_event = stop_event
        self.pool = pool
        self.host = host
        self.port = port
        if pipeline_pools is not None:
            if any(not isinstance(name, str) or not _ID.fullmatch(name) for name in pipeline_pools):
                raise ValueError("Use valid pipeline identifiers.")
            grouped = [unit for units in pipeline_pools.values() for unit in units]
            if (
                not pipeline_pools
                or any(not units for units in pipeline_pools.values())
                or len({id(unit) for unit in grouped}) != len(grouped)
                or len(grouped) != len(pool)
                or {id(unit) for unit in grouped} != {id(unit) for unit in pool}
                or len({unit.index for unit in grouped}) != len(grouped)
            ):
                raise ValueError("Named groups must contain exactly the disjoint, uniquely indexed aggregate pool.")
            if llm_proxy_config is not None or (
                llm_proxy_configs is not None and set(llm_proxy_configs) != set(pipeline_pools)
            ):
                raise ValueError("Named and legacy proxy configurations are incompatible.")
        elif llm_proxy_configs is not None:
            raise ValueError("Named proxy configurations require named pipeline groups.")
        self.llm_proxy_config = llm_proxy_config
        self.pipeline_pools = pipeline_pools
        self.llm_proxy_configs = llm_proxy_configs
        self.ready = Event()

    def run(self) -> None:
        """Start the FastAPI/uvicorn server (called from a ThreadManager thread)."""
        self.ready.clear()
        app = create_app(
            pool=self.pool,
            stop_event=self.stop_event,
            llm_proxy_config=self.llm_proxy_config,
            pipeline_pools=self.pipeline_pools,
            llm_proxy_configs=self.llm_proxy_configs,
        )

        logger.info(
            f"OpenAI Realtime API starting on ws://{self.host}:{self.port}/v1/realtime (pool size {len(self.pool)})"
        )

        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
        ready = self.ready
        stop_event = self.stop_event

        class ReadyServer(uvicorn.Server):
            async def startup(self, sockets: list[socket] | None = None) -> None:
                await super().startup(sockets=sockets)
                if self.started and not stop_event.is_set():
                    ready.set()

        server = ReadyServer(config)

        server.install_signal_handlers = lambda: None  # type: ignore[attr-defined]

        def _watch_stop() -> None:
            self.stop_event.wait()
            server.should_exit = True

        watcher = threading.Thread(target=_watch_stop, daemon=True)
        watcher.start()

        try:
            server.run()
        finally:
            # A bind/startup failure must also stop the model threads. During a
            # normal shutdown the event is already set by ThreadManager.
            self.ready.clear()
            self.stop_event.set()
