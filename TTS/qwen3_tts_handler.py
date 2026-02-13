import logging
from time import perf_counter
from pathlib import Path
import numpy as np
from baseHandler import BaseHandler

logger = logging.getLogger(__name__)

QWEN3_TTS_ROOT = Path("/home/andi/Documents/qwen3-tts")
MODELS_DIR = QWEN3_TTS_ROOT / "models"
DEFAULT_REF_AUDIO = str(QWEN3_TTS_ROOT / "ref_audio.wav")
DEFAULT_REF_TEXT = "This is a reference audio sample for voice cloning."
DEFAULT_TEXT = "Hello from the speech to speech benchmark. This is a latency test."


class Qwen3TTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        model_name="Qwen3-TTS-12Hz-0.6B-Base",
        device="cuda",
        dtype="auto",
        attn_implementation="flash_attention_2",
        ref_audio=DEFAULT_REF_AUDIO,
        ref_text=DEFAULT_REF_TEXT,
        language="English",
    ):
        self.should_listen = should_listen
        self.model_name = model_name
        self.device = device
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.language = language
        self.attn_implementation = attn_implementation
        self.dtype = dtype

        # Import qwen_tts from local project
        import sys
        qwen_site = QWEN3_TTS_ROOT / ".venv" / "lib" / "python3.10" / "site-packages"
        if qwen_site.exists() and str(qwen_site) not in sys.path:
            sys.path.insert(0, str(qwen_site))
        qwen_streaming = Path("/home/andi/Documents/Qwen3-TTS-streaming")
        if qwen_streaming.exists() and str(qwen_streaming) not in sys.path:
            sys.path.insert(0, str(qwen_streaming))
        if str(QWEN3_TTS_ROOT) not in sys.path:
            sys.path.insert(0, str(QWEN3_TTS_ROOT))

        import torch
        from qwen_tts import Qwen3TTSModel

        if dtype == "auto":
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif isinstance(dtype, str):
            self.dtype = getattr(torch, dtype)
        else:
            self.dtype = dtype

        model_path = MODELS_DIR / model_name
        logger.info(f"Loading Qwen3TTS model: {model_path}")
        self.model = Qwen3TTSModel.from_pretrained(
            str(model_path),
            device_map="cuda:0" if device == "cuda" and torch.cuda.is_available() else "cpu",
            dtype=self.dtype,
            attn_implementation=self.attn_implementation,
        )

    def process(self, llm_sentence):
        import torch

        if isinstance(llm_sentence, tuple):
            llm_sentence, _language_code = llm_sentence
        if not llm_sentence:
            llm_sentence = DEFAULT_TEXT

        start = perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        wavs, sr = self.model.generate_voice_clone(
            text=llm_sentence,
            language=self.language,
            ref_audio=self.ref_audio,
            ref_text=self.ref_text,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        audio = wavs[0]
        audio_int16 = (audio * 32768).astype(np.int16)

        logger.info(f"Qwen3TTS generated {len(audio_int16)/sr:.2f}s audio in {perf_counter()-start:.2f}s")

        # Yield as a single chunk
        yield audio_int16
        self.should_listen.set()

    def cleanup(self):
        try:
            import torch
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
