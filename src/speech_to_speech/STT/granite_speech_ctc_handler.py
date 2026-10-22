"""Speech to text with Granite Speech 5.0 TurboCTC through Transformers.

TurboCTC is a 470M conformer encoder with a CTC head: decoding is a single
non-autoregressive pass, so latency barely grows with transcript length. The
checkpoint covers English, French, German, Spanish, Portuguese, Italian and
Japanese and writes punctuated, cased text. CTC has no language token, so the
handler reports the configured language rather than a detected one.
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

DEFAULT_MODEL = "ibm-granite/granite-speech-5.0-470m-turboctc"
SAMPLE_RATE = 16000
SUPPORTED_LANGUAGES = ("en", "fr", "de", "es", "pt", "it", "ja")


class GraniteSpeechCTCSTTHandler(BaseSTTHandler):
    def setup(
        self,
        model_name: str = DEFAULT_MODEL,
        revision: Optional[str] = None,
        device: str = "auto",
        torch_dtype: str = "auto",
        language: str = "en",
        num_threads: Optional[int] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Granite Speech TurboCTC supports {', '.join(SUPPORTED_LANGUAGES)}; got {language!r}")
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        logger.info("Loading Granite Speech TurboCTC STT model: %s", model_name)
        self.device = resolve_device(device, TORCH_DEVICES, "Granite Speech TurboCTC")
        self.torch_dtype = resolve_torch_dtype(torch_dtype, self.device)
        self.gen_kwargs = dict(gen_kwargs or {})
        self.start_language = language
        self.last_language = language

        self.processor = transformers.AutoProcessor.from_pretrained(model_name, revision=revision)
        model = transformers.AutoModelForCTC.from_pretrained(model_name, revision=revision, dtype=self.torch_dtype)
        self.model = model.to(self.device).eval()
        self.warmup()

    def warmup(self) -> None:
        logger.info("Warming up %s", self.__class__.__name__)
        start = perf_counter()
        self._transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
        logger.info("%s: warmed up! time: %.3f s", self.__class__.__name__, perf_counter() - start)

    def _transcribe(self, audio: np.ndarray) -> str:
        inputs = self.processor([audio], sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = inputs.to(self.device, self.torch_dtype)
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **self.gen_kwargs)
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        audio = np.asarray(vad_audio.audio, dtype=np.float32)
        start = perf_counter()
        text = self._transcribe(audio)
        logger.debug(
            "Granite TurboCTC %s transcription took %.3f s for %.2f s of audio",
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
            language_code=self.start_language,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.speech_end_at_s,
        )
