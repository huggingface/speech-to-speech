"""Qwen3's native library synthesis path with lease-local CUDA graphs.

The default GGML runtime owns CUDA outside PyTorch and cannot use ZeroGPU's
startup emulation. This adapter selects the library's torch backend instead.
Graph capture happens only after a real allocation is active.
"""

from contextlib import contextmanager
from queue import Queue
from threading import Event

import numpy as np
import torch

from speech_to_speech.TTS.qwen3_tts_handler import DEFAULT_MODEL, Qwen3TTSHandler


@contextmanager
def _qwen_rope_compatibility():
    """Scope qwen-tts-hf's legacy registry fallback to the loading operation.

    qwen-tts-hf 0.1.1.post1 registers a global default initializer expecting
    config.rope_theta. Transformers 5.20 Mimi moved that field into
    rope_parameters, but its generic weight initialization consults this
    registry. Dispatch Mimi to its own native initializer during TTS loading.
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.models.mimi.modeling_mimi import MimiRotaryEmbedding

    original = ROPE_INIT_FUNCTIONS.get("default")
    if original is None:
        yield
        return

    def dispatch(config, device=None, seq_len=None, layer_type=None):
        if getattr(config, "model_type", None) == "mimi" and hasattr(config, "rope_parameters"):
            inv_freq, scale = MimiRotaryEmbedding.compute_default_rope_parameters(config)
            return inv_freq.to(device=device) if device is not None else inv_freq, scale
        return original(config, device=device, seq_len=seq_len, layer_type=layer_type)

    ROPE_INIT_FUNCTIONS["default"] = dispatch
    try:
        yield
    finally:
        ROPE_INIT_FUNCTIONS["default"] = original


class LeaseQwenTTS(Qwen3TTSHandler):
    def warmup(self):
        # Handler setup runs before an actual GPU allocation.
        pass

    def _setup_faster(self, model_name, dtype, attn_implementation, backend):
        if backend != "torch":
            raise ValueError("ZeroGPU Qwen3 requires the PyTorch backend")
        from faster_qwen3_tts import FasterQwen3TTS
        from qwen_tts import Qwen3TTSModel

        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        # CPU-first loading avoids accelerate's device-map CUDA initialization.
        with _qwen_rope_compatibility():
            base = Qwen3TTSModel.from_pretrained(
                model_name,
                device_map="cpu",
                dtype=self.dtype,
                attn_implementation=attn_implementation,
            )
        base.model.to("cuda").eval()
        base.device = torch.device("cuda")
        codec = base.model.speech_tokenizer
        codec.model.to("cuda").eval()
        codec.device = torch.device("cuda")
        self.model = FasterQwen3TTS(
            base_model=base,
            predictor_graph=None,
            talker_graph=None,
            device="cuda",
            dtype=self.dtype,
        )

    def initialize_lease(self):
        if self.model.predictor_graph is not None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3 graph initialization requires a real GPU allocation")
        from faster_qwen3_tts.predictor_graph import PredictorGraph
        from faster_qwen3_tts.talker_graph import TalkerGraph

        talker = self.model.model.model.talker
        config = self.model.model.model.config.talker_config
        predictor = talker.code_predictor
        self.model.predictor_graph = PredictorGraph(
            predictor,
            predictor.model.config,
            config.hidden_size,
            device="cuda",
            dtype=self.dtype,
        )
        self.model.talker_graph = TalkerGraph(
            talker.model,
            config,
            device="cuda",
            dtype=self.dtype,
            max_seq_len=self.model.max_seq_len,
        )
        # Graphs cannot cross forks/device contexts; spaces 0.51.3 may reuse
        # an idle child with its existing context and valid captured graphs.
        self.model.warmup(prefill_len=100)


def load_tts(model_name=DEFAULT_MODEL, speaker="Aiden"):
    return LeaseQwenTTS(
        Event(),
        Queue(),
        Queue(),
        setup_kwargs={
            "should_listen": Event(),
            "model_name": model_name,
            "device": "cuda",
            "dtype": "bfloat16",
            "backend": "torch",
            "attn_implementation": "eager",
            "speaker": speaker,
            "language": "english",
            "streaming_chunk_size": 4,
            "max_new_tokens": 512,
        },
    )


def gpu_models(handler):
    """Both talker and codec must be CUDA-backed."""
    base = handler.model.model.model
    return {"tts": base, "tts_codec": base.speech_tokenizer.model}


def initialize_tts_graphs(handler):
    """Capture graphs exclusively, before any model lane issues CUDA work."""
    handler.initialize_lease()


def synthesize_chunks16k(handler, text, language="english", speaker=None):
    """Yield native Qwen handler PCM16 chunks; never wait for a whole sentence."""
    handler.initialize_lease()
    if speaker:
        handler.speaker = speaker
    for model in gpu_models(handler).values():
        if next(model.parameters()).device.type != "cuda":
            raise RuntimeError("Qwen3 TTS CPU fallback is disabled")
    with torch.inference_mode():
        for chunk in handler._process_custom_voice(text, language):
            yield np.asarray(chunk, dtype=np.int16)
