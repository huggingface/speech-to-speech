# ruff: noqa: I001
"""Library-default Parakeet STT, loaded without running CUDA before a lease.

Requires nano-parakeet==0.2.1. Only model setup/warmup policy is adapted;
transcription remains the library's Parakeet implementation.
"""

import spaces  # noqa: F401  # Establish ZeroGPU CUDA emulation before importing torch.

import os
from queue import Queue
from threading import Event
from types import MethodType
from unittest.mock import patch

import numpy as np
import torch

from speech_to_speech.STT.parakeet_tdt_handler import ParakeetTDTSTTHandler


DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"


def _skip_graph_capture(self, device):
    # nano-parakeet falls back to its native eager decoder when graph is None.
    # Do not replace the decoding implementation or move it onto CPU.
    return None


class DeferredParakeet(ParakeetTDTSTTHandler):
    def warmup(self):
        # Handler.setup calls this outside a GPU lease.
        pass

    def _setup_nano_parakeet(self, model_name):
        from nano_parakeet import ParakeetTDT, from_pretrained

        self.backend = "nano_parakeet"
        # The upstream loader independently calls model.warmup(). Suppress just
        # that eager inference, scoped to synchronous startup model loading.
        # Explicit float32 also avoids a real GPU capability query at startup.
        with patch.object(ParakeetTDT, "warmup", lambda self: None):
            self.model = from_pretrained(model_name=model_name, device="cuda", dtype=torch.float32)
        if os.getenv("PARAKEET_CUDA_GRAPHS", "0") != "1":
            self.model._build_decode_graph = MethodType(_skip_graph_capture, self.model)
        self.device = "cuda"


def create_stt(model_name=DEFAULT_MODEL):
    return DeferredParakeet(
        Event(),
        Queue(),
        Queue(),
        setup_kwargs={"model_name": model_name, "device": "cuda", "enable_live_transcription": False},
    )


def warmup_stt(handler):
    """Call once in each allocated GPU worker, before dispatching its requests.

    This warms preprocessing/encoder/decoder CUDA kernels. It cannot eliminate
    scheduler wait and consumes real allocated GPU time. Exceptions propagate.
    """
    assert_cuda(handler)
    handler.model.warmup(duration_s=1.0)
    torch.cuda.synchronize()


def assert_cuda(handler):
    if not torch.cuda.is_available() or next(handler.model.parameters()).device.type != "cuda":
        raise RuntimeError("Parakeet STT requires the allocated CUDA GPU; no CPU fallback")


def transcribe(handler, audio):
    """Use the library's nano-parakeet inference method without swallowing errors."""
    assert_cuda(handler)
    with torch.inference_mode():
        return handler._process_nano_parakeet(np.asarray(audio, dtype=np.float32))
