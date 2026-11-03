# ruff: noqa: I001, E402
"""ZeroGPU host for the repository's normal hands-free realtime voice demo."""

import spaces  # noqa: F401  # Must be first: CUDA model initialization uses ZeroGPU emulation.

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Event

import gradio as gr
import nltk
import uvicorn
from fastapi import Request
from fastapi.staticfiles import StaticFiles

from speech_to_speech.api.openai_realtime.websocket_router import create_app
from speech_to_speech.arguments_classes.module_arguments import ModuleArguments
from speech_to_speech.arguments_classes.vad_arguments import VADHandlerArguments
from speech_to_speech.s2s_pipeline import _build_pipeline_unit
from speech_to_speech.utils.thread_manager import ThreadManager
from speech_to_speech.zerogpu.readiness import ModelReadinessMiddleware

from speech_to_speech.zerogpu import runtime
from speech_to_speech.backend_registry import STT_BACKENDS, LLM_BACKENDS, TTS_BACKENDS, BackendSelection

worker = runtime.initialize(
    stt_model=os.getenv("STT_MODEL", runtime.DEFAULT_STT_MODEL),
    llm_model=os.getenv("LLM_MODEL", runtime.DEFAULT_LLM_MODEL),
    tts_backend=os.getenv("TTS_BACKEND", "qwen3"),
    tts_model=os.getenv("TTS_MODEL"),
    tts_voice=os.getenv("TTS_VOICE"),
)


def backend_selections():
    selections = []
    for registry, kind in ((STT_BACKENDS, "stt"), (LLM_BACKENDS, "llm"), (TTS_BACKENDS, "tts")):
        spec = registry["zerogpu"]
        config = spec.normalize(spec.config_type())
        config["model_name"] = runtime.get_model_ids()[kind]
        if kind == "tts":
            config["backend"] = runtime.get_tts_config()["backend"]
            config["voice"] = runtime.get_tts_config()["voice"]
        selections.append(BackendSelection(spec, config))
    return tuple(selections)


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


def create_host(unit, gpu_blocks, stop_event: Event):
    """Models initialize first; permanent CPU pipeline threads start in lifespan."""
    app = create_app(pool=[unit], stop_event=stop_event)
    app.add_middleware(ModelReadinessMiddleware, prefetch=runtime.prepare)
    original_lifespan = app.router.lifespan_context
    manager = ThreadManager(unit.handlers)

    @asynccontextmanager
    async def lifespan(app):
        manager.start()
        try:
            async with original_lifespan(app):
                yield
        finally:
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
    # conflicting with our own Uvicorn host; the callback mount needs no SSR.
    app = gr.mount_gradio_app(app, gpu_blocks, path="/gpu", ssr_mode=False)
    frontend = Path(__file__).parent / "frontend"
    app.mount("/", StaticFiles(directory=frontend, html=True), name="voice-demo")
    return app


logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
nltk.download("punkt_tab", quiet=True, raise_on_error=True)

# Gradio discovers decorated functions through registered events, even though
# the real conversation uses the existing WebSocket API and CPU handler queues.
with gr.Blocks() as gpu_blocks:
    gr.Markdown("Open [the voice conversation](/) to talk.")
    with gr.Column(visible=False):
        lease_id = gr.Textbox()
        result = gr.JSON()
        lease = gr.Button("Inference worker")
        lease.click(worker.lease_worker, lease_id, result, api_name=False)

gpu_blocks.queue()
stop_event = Event()
unit = build_unit(backend_selections(), stop_event)
from prefetch import enable_speech_prefetch

enable_speech_prefetch(unit, runtime.prefetch)
# Mounted FastAPI does not call Blocks.launch, which normally initializes ZeroGPU.
# Experimental private API; all models must be loaded before this call.
from spaces.config import Config

if Config.zero_gpu:
    from spaces.zero import startup

    startup()
app = create_host(unit, gpu_blocks, stop_event)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860"))))
