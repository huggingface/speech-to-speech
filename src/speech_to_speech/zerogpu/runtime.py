"""Explicitly initialized, shared ZeroGPU inference runtime.

Importing this module does not import CUDA/model libraries or load weights.
A Hugging Face deployment must import ``spaces`` before other CUDA libraries,
then call :func:`initialize` before starting its ZeroGPU workers. The runtime
owns inference only; native handlers retain conversation and cancellation state.
"""

from importlib import import_module
from threading import Lock

DEFAULT_STT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_LLM_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

_worker = None
_settings = None
_initialize_lock = Lock()


def initialize(
    *, stt_model=DEFAULT_STT_MODEL, llm_model=DEFAULT_LLM_MODEL, tts_model=DEFAULT_TTS_MODEL, tts_speaker="Aiden"
):
    """Load all three models once, without inference outside a GPU allocation.

    Returns the worker module so the deployment can register its decorated
    ``lease_worker`` and diagnostic GPU entrypoints with Gradio. Repeating the
    same initialization is harmless; changing models requires a process restart.
    Supported model families are Parakeet TDT, Transformers causal LMs, and
    Qwen3 TTS CustomVoice. Model identifiers alone do not add other backends.
    """
    global _worker, _settings
    settings = dict(stt_model=stt_model, llm_model=llm_model, tts_model=tts_model, tts_speaker=tts_speaker)
    if any(not isinstance(value, str) or not value.strip() for value in settings.values()):
        raise ValueError("ZeroGPU model identifiers and speaker must be nonempty strings")
    with _initialize_lock:
        if _worker is not None:
            if settings != _settings:
                raise RuntimeError(
                    "ZeroGPU runtime is already initialized with different models; restart to change them"
                )
            return _worker
        try:
            worker = import_module("speech_to_speech.zerogpu._worker")
        except ModuleNotFoundError as exc:
            if exc.name == "spaces":
                raise RuntimeError(
                    "ZeroGPU requires the optional spaces package and a Hugging Face ZeroGPU deployment"
                ) from exc
            raise
        worker.initialize_models(stt_model, llm_model, tts_model, tts_speaker)
        _settings = settings
        _worker = worker
        return worker


def _require_worker():
    if _worker is None:
        raise RuntimeError(
            "ZeroGPU runtime is not initialized; call speech_to_speech.zerogpu.runtime.initialize() before building the pipeline"
        )
    return _worker


def get_model_ids():
    """Return a copy of the initialized model identifiers and default speaker."""
    _require_worker()
    return {
        "stt": _settings["stt_model"],
        "llm": _settings["llm_model"],
        "tts": _settings["tts_model"],
        "speaker": _settings["tts_speaker"],
    }


def get_tokenizer():
    """Tokenizer used by the shared GPU LLM (for native history management)."""
    return _require_worker().llm_tokenizer


def get_tts_speakers():
    """Speakers supported by the initialized native Qwen3 handler."""
    return _require_worker().tts._supported_speakers()


def prepare():
    """Wait for real STT, LLM and TTS warmup in the shared bounded GPU lease."""
    return _require_worker().prepare()


def prefetch():
    """Request an allocation without waiting, for native speech-start hooks."""
    return _require_worker().prefetch()


def transcribe(audio):
    return _require_worker().transcribe(audio)


def tokens(messages, gen_kwargs, cancel_id=None):
    return _require_worker().tokens(messages, gen_kwargs, cancel_id)


def synthesize_chunks(text, language="english", speaker=None):
    """Stream native 16 kHz PCM16 chunks from the allocated GPU."""
    return _require_worker().synthesize_chunks(text, language, speaker)


def cancel_generation(cancel_id):
    return _require_worker().cancel_generation(cancel_id)


def cleanup_cancel(cancel_id):
    return _require_worker().cleanup_cancel(cancel_id)


def stop():
    """Ask the current allocation to retire; already accepted work may drain."""
    if _worker is not None:
        _worker.stop()
