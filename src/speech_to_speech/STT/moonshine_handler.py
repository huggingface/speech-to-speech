"""English speech to text with Moonshine checkpoints through Transformers.

Moonshine is a family of small encoder-decoder models built for edge devices:
``moonshine-tiny`` (27M) and ``moonshine-base`` (61M), plus the newer
``moonshine-streaming-*`` checkpoints (34M / 123M / 245M). Unlike Whisper, the
encoder takes variable-length audio without 30 s padding, so short turns are cheap.
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Iterator, Optional

import numpy as np
import torch
import transformers

from speech_to_speech.pipeline.console import console
from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler
from speech_to_speech.STT.qwen3_asr_handler import resolve_torch_dtype
from speech_to_speech.utils.utils import TORCH_DEVICES, resolve_device

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "moonshine-ai/moonshine-base"
SAMPLE_RATE = 16000
# The model card bounds decoding by audio length to stop hallucination loops.
DEFAULT_TOKENS_PER_SECOND = 6.5
# Very short turns ("yes", "no") would otherwise get a limit below one word.
MIN_MAX_LENGTH = 8


def model_class_for(config: Any) -> type[Any]:
    """Pick the Transformers class matching a Moonshine config.

    Streaming checkpoints use a different architecture class; both expose the
    standard seq2seq ``generate`` API.
    """
    if getattr(config, "model_type", "") == "moonshine_streaming":
        return transformers.MoonshineStreamingForConditionalGeneration
    return transformers.MoonshineForConditionalGeneration


def max_length_for(num_samples: int, tokens_per_second: float) -> int:
    return max(MIN_MAX_LENGTH, int(num_samples * tokens_per_second / SAMPLE_RATE))


class MoonshineSTTHandler(BaseSTTHandler):
    """Moonshine is English-only, so every final turn reports ``en``."""

    def setup(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        torch_dtype: str = "auto",
        tokens_per_second: float = DEFAULT_TOKENS_PER_SECOND,
        num_threads: Optional[int] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        if tokens_per_second <= 0:
            raise ValueError(f"tokens_per_second must be positive, got {tokens_per_second}")
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        logger.info("Loading Moonshine STT model: %s", model_name)
        self.device = resolve_device(device, TORCH_DEVICES, "Moonshine STT")
        self.torch_dtype = resolve_torch_dtype(torch_dtype, self.device)
        self.tokens_per_second = tokens_per_second
        self.gen_kwargs = dict(gen_kwargs or {})
        self.start_language = "en"
        self.last_language = "en"

        self.processor = transformers.AutoProcessor.from_pretrained(model_name)
        config = transformers.AutoConfig.from_pretrained(model_name)
        model = model_class_for(config).from_pretrained(model_name, dtype=self.torch_dtype)
        self.model = model.to(self.device).eval()
        self.warmup()

    def warmup(self) -> None:
        logger.info("Warming up %s", self.__class__.__name__)
        start = perf_counter()
        self._transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
        logger.info("%s: warmed up! time: %.3f s", self.__class__.__name__, perf_counter() - start)

    def _transcribe(self, audio: np.ndarray) -> str:
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = inputs.to(self.device, self.torch_dtype)
        gen_kwargs = {"max_length": max_length_for(len(audio), self.tokens_per_second), **self.gen_kwargs}
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
        return self.processor.decode(generated_ids[0], skip_special_tokens=True).strip()

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        audio = np.asarray(vad_audio.audio, dtype=np.float32)
        start = perf_counter()
        text = self._transcribe(audio)
        logger.debug(
            "Moonshine %s transcription took %.3f s for %.2f s of audio",
            vad_audio.mode,
            perf_counter() - start,
            len(audio) / SAMPLE_RATE,
        )

        if vad_audio.mode == "progressive":
            yield PartialTranscription(
                text=text,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
            )
            return

        console.print(f"[yellow]USER: {text}")
        self._record_final_stt(vad_audio, perf_counter() - start)
        yield Transcription(
            text=text,
            language_code="en",
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.speech_end_at_s,
        )
