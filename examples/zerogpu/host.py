# ruff: noqa: I001
"""Serve the normal voice UI and real Realtime API alongside ZeroGPU callbacks."""

import spaces  # noqa: F401  # ZeroGPU must intercept CUDA before speech_to_speech imports.

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Event

import gradio as gr
from fastapi import Request
from fastapi.staticfiles import StaticFiles

from speech_to_speech.api.openai_realtime.websocket_router import create_app
from speech_to_speech.arguments_classes.module_arguments import ModuleArguments
from speech_to_speech.arguments_classes.vad_arguments import VADHandlerArguments
from speech_to_speech.s2s_pipeline import _build_pipeline_unit
from speech_to_speech.utils.thread_manager import ThreadManager


def build_unit(selections, stop_event):
    stt, llm, tts = selections
    unit = _build_pipeline_unit(
        index=0,
        stop_event=stop_event,
        module_kwargs=ModuleArguments(
            stt=stt.name,
            llm_backend=llm.name,
            tts=tts.name,
            enable_live_transcription=False,
        ),
        vad_handler_kwargs=VADHandlerArguments(smart_turn=True),
        stt_backend=stt,
        llm_backend=llm,
        tts_backend=tts,
    )
    return unit


def create_host(unit, gpu_blocks, stop_event: Event, frontend_dir=None, prefetch=None):
    """Models initialize first; permanent CPU pipeline threads start in lifespan."""
    app = create_app(pool=[unit], stop_event=stop_event)
    if prefetch is not None:
        from speech_to_speech.zerogpu.readiness import ModelReadinessMiddleware

        app.add_middleware(ModelReadinessMiddleware, prefetch=prefetch)
    original_lifespan = app.router.lifespan_context
    manager = ThreadManager(unit.handlers)

    @asynccontextmanager
    async def lifespan(app):
        manager.start()
        try:
            async with original_lifespan(app):
                yield
        finally:
            from speech_to_speech.zerogpu import runtime

            runtime.stop()
            await asyncio.to_thread(manager.stop)

    app.router.lifespan_context = lifespan

    @app.get("/api/config")
    def config(request: Request):
        # HF forwards HTTPS; SPACE_HOST is the canonical host in hosted mode.
        host = os.getenv("SPACE_HOST") or request.url.netloc
        scheme = "wss" if os.getenv("SPACE_HOST") or request.url.scheme == "https" else "ws"
        return {
            "search": False,
            "lb": False,
            "allowDirect": True,
            "modelReadiness": True,
            "s2sUrl": f"{scheme}://{host}/v1/realtime",
            "rtc": False,
            "iceServers": [],
            "auth": False,
            "startupGreeting": os.getenv("STARTUP_GREETING", ""),
        }

    @app.get("/api/me")
    def me():
        return {"enabled": False}

    # Register GPU callbacks with Gradio, while the normal UI owns the root URL.
    # Spaces enables SSR by default. Its Node proxy claims the public port,
    # conflicting with our own Uvicorn host; this diagnostic UI needs no SSR.
    app = gr.mount_gradio_app(app, gpu_blocks, path="/gpu", ssr_mode=False)
    frontend = Path(frontend_dir or Path(__file__).parent / "frontend")
    app.mount("/", StaticFiles(directory=frontend, html=True), name="voice-demo")
    return app
