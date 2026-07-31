from __future__ import annotations

import logging
from typing import Any, Iterator

from rich.console import Console

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

logger = logging.getLogger(__name__)
console = Console()

# ISO 639-1 (plus 'yue' for Cantonese, not part of the standard) -> the language
# name Qwen3-ASR expects/returns, per https://huggingface.co/Qwen/Qwen3-ASR-1.7B
LANGUAGE_NAME_BY_CODE = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "hu": "Hungarian",
    "mk": "Macedonian",
    "ro": "Romanian",
}
CODE_BY_LANGUAGE_NAME = {name.lower(): code for code, name in LANGUAGE_NAME_BY_CODE.items()}


def _patch_check_model_inputs() -> None:
    """qwen_asr's vendored modeling code calls ``@check_model_inputs()`` (transformers==4.57.6's
    optional-arg decorator factory). transformers>=5 made ``func`` a required positional arg,
    raising ``TypeError: check_model_inputs() missing 1 required positional argument``.
    """
    import transformers.utils.generic as generic

    if getattr(generic.check_model_inputs, "_qwen3_asr_compat", False):
        return
    original = generic.check_model_inputs

    def compat(func=None, *, tie_last_hidden_states=True):
        if func is None:
            return lambda f: original(f)
        return original(func)

    compat._qwen3_asr_compat = True
    generic.check_model_inputs = compat


def _patch_rope_default() -> None:
    """transformers>=5 dropped the "default" key from ROPE_INIT_FUNCTIONS (replaced by the
    rope_parameters-based RotaryEmbeddingConfigMixin API), but qwen_asr's vendored rotary
    embedding still looks it up by that key, raising ``KeyError: 'default'``. Re-register it
    using the transformers==4.57.6 implementation qwen_asr was written against.
    """
    import torch
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def compute_default_rope_parameters(config=None, device=None, seq_len=None):
        base = config.rope_theta
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        attention_factor = 1.0
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    ROPE_INIT_FUNCTIONS["default"] = compute_default_rope_parameters


def _patch_qwen3_asr_config_init() -> None:
    """qwen_asr's ``Qwen3ASRConfig.__init__`` assigns ``self.thinker_config`` *after* calling
    ``super().__init__(**kwargs)``. transformers>=5's ``PretrainedConfig.__init__`` now eagerly
    validates token ids via ``self.get_text_config()``, which qwen_asr overrides to read
    ``self.thinker_config`` — raising ``AttributeError`` because it isn't set yet. Reorder so the
    sub-config exists before the base class validates.
    """
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
        Qwen3ASRConfig,
        Qwen3ASRThinkerConfig,
    )
    from transformers.configuration_utils import PretrainedConfig

    if getattr(Qwen3ASRConfig.__init__, "_qwen3_asr_compat", False):
        return

    def compat_init(self, thinker_config=None, support_languages=None, **kwargs):
        if thinker_config is None:
            thinker_config = {}
        self.thinker_config = Qwen3ASRThinkerConfig(**thinker_config)
        self.support_languages = support_languages
        PretrainedConfig.__init__(self, **kwargs)

    compat_init._qwen3_asr_compat = True
    Qwen3ASRConfig.__init__ = compat_init


def _apply_transformers_5x_compat() -> None:
    """Best-effort compatibility shims for qwen-asr==0.0.6 running under transformers>=5.

    qwen-asr hard-pins transformers==4.57.6; this project pins transformers==5.6.2 on macOS
    (needed by Qwen3-TTS/mlx-audio), so both can't be installed together and qwen-asr must be
    force-installed with --no-deps. These patch three independent transformers 4->5 breaking
    changes qwen_asr hasn't adapted to yet. See src/speech_to_speech/STT/README.md for details.
    Each patch checks the current state first, so it's a no-op once qwen-asr fixes it upstream.
    """
    _patch_check_model_inputs()
    _patch_rope_default()

    import qwen_asr  # noqa: F401  (loads the vendored modeling code check_model_inputs patches)

    _patch_qwen3_asr_config_init()


class Qwen3ASRSTTHandler(BaseSTTHandler):
    """
    Handles Speech To Text generation using Qwen3-ASR (an audio-LLM ASR model).
    """

    def setup(
        self,
        model_name: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        language: str | None = None,
        max_new_tokens: int = 256,
        max_inference_batch_size: int = 1,
        gen_kwargs: dict[str, Any] = {},
    ) -> None:
        try:
            import torch

            _apply_transformers_5x_compat()
            from qwen_asr import Qwen3ASRModel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Qwen3-ASR STT requires the optional 'qwen3-asr' extra. "
                "Install it with `pip install speech-to-speech[qwen3-asr]`."
            ) from exc

        self.start_language = language
        self.last_language_code = language if language != "auto" else None
        self.language_name = self._code_to_language_name(self.last_language_code)

        self.model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=getattr(torch, torch_dtype),
            device_map=device,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )
        self.warmup()

    def _code_to_language_name(self, language_code: str | None) -> str | None:
        if language_code is None:
            return None
        language_name = LANGUAGE_NAME_BY_CODE.get(language_code)
        if language_name is None:
            raise ValueError(
                f"Unsupported --qwen3_asr_language {language_code!r}. "
                f"Supported codes: {sorted(LANGUAGE_NAME_BY_CODE)}, or 'auto'."
            )
        return language_name

    def warmup(self) -> None:
        import numpy as np

        logger.info(f"Warming up {self.__class__.__name__}")
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1s of silence at 16kHz
        self.model.transcribe(audio=(dummy_audio, 16000), language=self.language_name)

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        if vad_audio.mode == "progressive":
            # Re-running full Qwen3-ASR inference on every progressive/partial VAD chunk while
            # the user is still speaking would stall the pipeline for seconds at a time. Only
            # transcribe once the turn is final.
            return

        logger.debug("infering Qwen3-ASR...")

        results = self.model.transcribe(audio=(vad_audio.audio, 16000), language=self.language_name)
        result = results[0]
        pred_text = result.text
        language_code = CODE_BY_LANGUAGE_NAME.get(result.language.lower(), result.language) if result.language else None

        if language_code is not None:
            self.last_language_code = language_code

        logger.debug("finished Qwen3-ASR inference")
        console.print(f"[yellow]USER: {pred_text}")
        logger.debug(f"Language Code Qwen3-ASR: {language_code}")

        if self.start_language == "auto" and language_code is not None:
            language_code += "-auto"

        yield Transcription(
            text=pred_text,
            language_code=language_code,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
        )
