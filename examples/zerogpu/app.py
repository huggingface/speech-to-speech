# ruff: noqa: I001, E402
"""ZeroGPU host for the repository's normal hands-free realtime voice demo."""

import spaces  # noqa: F401  # Must be first: CUDA model initialization uses ZeroGPU emulation.

import logging
import os
from threading import Event

import gradio as gr
import nltk
import uvicorn

from speech_to_speech.zerogpu import runtime
from speech_to_speech.backend_registry import STT_BACKENDS, LLM_BACKENDS, TTS_BACKENDS, BackendSelection

worker = runtime.initialize(
    stt_model=os.getenv("STT_MODEL", runtime.DEFAULT_STT_MODEL),
    llm_model=os.getenv("LLM_MODEL", runtime.DEFAULT_LLM_MODEL),
    tts_model=os.getenv("TTS_MODEL", runtime.DEFAULT_TTS_MODEL),
    tts_speaker=os.getenv("TTS_SPEAKER", "Aiden"),
)


def backend_selections():
    selections = []
    for registry, kind in ((STT_BACKENDS, "stt"), (LLM_BACKENDS, "llm"), (TTS_BACKENDS, "tts")):
        spec = registry["zerogpu"]
        config = spec.normalize(spec.config_type())
        config["model_name"] = runtime.get_model_ids()[kind]
        if kind == "tts":
            config["speaker"] = runtime.get_model_ids()["speaker"]
        selections.append(BackendSelection(spec, config))
    return tuple(selections)


from host import build_unit, create_host

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
nltk.download("punkt_tab", quiet=True, raise_on_error=True)

# Gradio discovers decorated functions through registered events, even though
# the real conversation uses the existing WebSocket API and CPU handler queues.
with gr.Blocks() as gpu_blocks:
    gr.Markdown("GPU diagnostics. Open [the voice conversation](/) to talk.")
    with gr.Column(visible=False):
        audio = gr.JSON()
        messages = gr.JSON()
        kwargs = gr.JSON()
        cancel_id = gr.Textbox()
        text = gr.Textbox()
        result = gr.JSON()
        transcribe = gr.Button("Transcribe")
        generate = gr.Button("Generate")
        synthesize = gr.Button("Synthesize")
        transcribe.click(worker.transcribe_once, audio, text, api_name=False)
        generate.click(worker.tokens_once, [messages, kwargs, cancel_id], text, api_name=False)
        synthesize.click(worker.synthesize_once, text, result, api_name=False)
        lease = gr.Button("Inference worker")
        lease.click(worker.lease_worker, cancel_id, result, api_name=False)
    info = gr.Button("Inspect GPU")
    info.click(worker.gpu_info, None, gr.JSON(), api_name="gpu_info")

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
app = create_host(unit, gpu_blocks, stop_event, prefetch=runtime.prepare)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860"))))
